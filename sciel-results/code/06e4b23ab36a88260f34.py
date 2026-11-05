import random
import time
import sys
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for 120s limit
    
    # Helper to check time
    def time_up():
        return time.time() - start_time > time_limit
    
    best_result = None
    best_scope = float('inf')
    
    # Precompute a table of conflict masks for efficiency
    # For a given current_row and used_diffs, we want to quickly find valid next elements
    # We'll use a different approach: iterate candidates and check conflicts via bitmask
    
    def solve_attempt(target_scope, rng_seed):
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        used_diffs = 0
        rows = []
        
        def _backtrack(row_idx, current_row, current_max_val):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                # All rows constructed
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0], current_max_val):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            upper_bound = min(target_scope - 1, best_scope - 1)
            
            if last_val >= upper_bound:
                return False
            
            candidates = []
            
            # Generate candidates with conflict checking
            for x in range(last_val + 1, upper_bound + 1):
                # Check conflicts using bitmask
                new_mask = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                # Cost heuristic: prefer larger differences (less scarce)
                cost = 0.0
                for a in current_row:
                    d = x - a
                    cost += 1.0 / d
                
                # Add noise for exploration
                noise = rng.random() * 0.05
                
                candidates.append((cost + noise, x, new_mask))
                
                # Limit candidates
                if len(candidates) > 15:
                    break
            
            if not candidates:
                return False
            
            # Sort by cost
            candidates.sort(key=lambda item: item[0])
            
            # Try top candidates with randomization
            top_n = min(4, len(candidates))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            remaining = candidates[top_n:]
            
            for cost, x, new_mask in top_candidates + remaining:
                old_used = used_diffs
                used_diffs = used_diffs | new_mask
                
                if _backtrack(row_idx, current_row + [x], max(current_max_val, x)):
                    return True
                
                used_diffs = old_used
            
            return False
        
        return _backtrack(0, [0], 0)

    def _local_search(rows, max_iter=20000):
        """
        Local search to reduce scope by moving elements within rows.
        """
        nonlocal best_result, best_scope
        rng = random.Random(42)
        
        if not rows:
            return
        
        current_rows = [r[:] for r in rows]
        current_scope = max(r[-1] for r in current_rows)
        
        # Build used differences set
        used_diffs = set()
        for row in current_rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        improved = True
        iters = 0
        while improved and iters < max_iter and not time_up():
            improved = False
            iters += 1
            
            # Try to reduce the maximum element
            max_row_idx = max(range(len(current_rows)), key=lambda i: current_rows[i][-1])
            row = current_rows[max_row_idx]
            
            # Try to move each element to a smaller valid position
            for pos in range(len(row) - 1, 0, -1):  # From right to left
                old_val = row[pos]
                # Try smaller values
                for new_val in range(row[pos - 1] + 1, old_val):
                    # Check if moving this element creates conflicts
                    # Remove old differences
                    old_diffs = set()
                    for i in range(len(row)):
                        if i != pos:
                            d = abs(row[pos] - row[i])
                            old_diffs.add(d)
                    
                    # Check if new value creates conflicts
                    new_val_valid = True
                    new_diffs = set()
                    for i in range(len(row)):
                        if i != pos:
                            d = abs(new_val - row[i])
                            if d in used_diffs and d not in old_diffs:
                                new_val_valid = False
                                break
                            new_diffs.add(d)
                    
                    if not new_val_valid:
                        continue
                    
                    # Check that new_diffs don't conflict with each other (they shouldn't since row is sorted)
                    # But we need to ensure the row remains strictly increasing
                    if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
                        continue
                    
                    # Apply the change
                    row[pos] = new_val
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    
                    new_scope = max(r[-1] for r in current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        improved = True
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_result = [r[:] for r in current_rows]
                    break  # Move to next position
        
        # Update best if improved
        final_scope = max(r[-1] for r in current_rows)
        if final_scope < best_scope:
            best_scope = final_scope
            best_result = [r[:] for r in current_rows]
    
    # Phase 1: Initial greedy construction
    if time.time() - start_time < 5.0:
        for g_seed in range(15):
            rows = _greedy_construct(n_rows, k, g_seed)
            if rows:
                scope = max(r[-1] for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_result = [r[:] for r in rows]
            if time.time() - start_time > 5.0:
                break
    
    # Phase 2: Local search to improve
    if best_result is not None:
        _local_search(best_result, max_iter=50000)
    
    # Phase 3: Backtracking to find better solutions
    seed_counter = 0
    while time.time() - start_time < time_limit:
        if best_result is not None:
            target = best_scope
        else:
            target = 200
        
        solve_attempt(target, seed_counter)
        seed_counter += 1
        
        # Periodically run local search
        if seed_counter % 5 == 0 and best_result is not None:
            _local_search(best_result, max_iter=20000)
    
    if best_result is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
        
    return best_result

def _greedy_construct(n_rows, k, seed):
    """
    Fast greedy construction to provide an initial upper bound.
    Uses a simple heuristic: pick the smallest valid x.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
        
    return rows
