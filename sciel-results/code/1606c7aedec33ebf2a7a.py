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
    
    def time_up():
        return time.time() - start_time > time_limit
    
    best_result = None
    best_scope = float('inf')
    
    def get_used_diffs_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def solve_attempt(target_scope, rng_seed):
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        used_diffs = 0
        rows = []
        
        def _backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            upper_bound = target_scope - 1
            
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                conflict = False
                new_mask = 0
                for a in current_row:
                    d = x - a
                    if d >= 256:
                        continue
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                cost = 0.0
                for a in current_row:
                    d = x - a
                    cost += 1.0 / d
                
                noise = rng.random() * 0.01
                candidates.append((cost + noise, x, new_mask))
                
                if len(candidates) > 15:
                    break
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda item: item[0])
            
            top_n = min(4, len(candidates))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            remaining_candidates = candidates[top_n:]
            
            for cost, x, new_mask in top_candidates + remaining_candidates:
                old_used = used_diffs
                used_diffs = used_diffs | new_mask
                
                if _backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used
            
            return False
        
        return _backtrack(0, [0])

    def local_search_improve(rows, max_iters=5000):
        """Try to reduce scope by moving elements to smaller values."""
        nonlocal best_scope, best_result
        if rows is None:
            return None
        
        current = [r[:] for r in rows]
        current_scope = max(r[-1] for r in current)
        used = get_used_diffs_set(current)
        
        for _ in range(max_iters):
            if time_up():
                break
            
            # Find the row with the maximum last element
            max_row_idx = max(range(n_rows), key=lambda i: current[i][-1])
            max_row = current[max_row_idx]
            
            if max_row[-1] <= 105:
                break
            
            # Try to reduce elements in this row
            improved = False
            for pos in range(k, 0, -1):  # Try from the end backwards
                if pos == 0:
                    continue
                old_val = max_row[pos]
                
                # Try smaller values
                for new_val in range(max_row[pos-1] + 1, old_val):
                    # Check if moving this element to new_val is valid
                    # Remove old differences involving this element
                    removed = set()
                    for i in range(len(max_row)):
                        if i != pos:
                            removed.add(abs(max_row[pos] - max_row[i]))
                    
                    # Tentatively update
                    max_row[pos] = new_val
                    new_diffs = set()
                    for i in range(len(max_row)):
                        if i != pos:
                            new_diffs.add(abs(max_row[pos] - max_row[i]))
                    
                    # Check conflicts
                    conflict = False
                    for d in new_diffs:
                        if d in used and d not in removed:
                            conflict = True
                            break
                    
                    if not conflict:
                        # Valid move
                        used -= removed
                        used |= new_diffs
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try swapping elements between rows
                for r1 in range(n_rows):
                    for r2 in range(r1+1, n_rows):
                        for p1 in range(1, len(current[r1])):
                            for p2 in range(1, len(current[r2])):
                                v1 = current[r1][p1]
                                v2 = current[r2][p2]
                                if v1 == v2:
                                    continue
                                
                                # Try swapping
                                current[r1][p1] = v2
                                current[r2][p2] = v1
                                
                                if is_valid(current):
                                    new_scope = max(r[-1] for r in current)
                                    if new_scope < current_scope:
                                        current_scope = new_scope
                                        used = get_used_diffs_set(current)
                                        improved = True
                                        break
                                else:
                                    current[r1][p1] = v1
                                    current[r2][p2] = v2
                                
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        if max(r[-1] for r in current) < best_scope:
            best_scope = max(r[-1] for r in current)
            best_result = [r[:] for r in current]
        
        return current

    # Phase 1: Initial greedy construction
    if time.time() - start_time < 3.0:
        for g_seed in range(20):
            rows = _greedy_construct(n_rows, k, g_seed)
            if rows:
                scope = max(r[-1] for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_result = [r[:] for r in rows]
            if time.time() - start_time > 3.0:
                break

    # Phase 2: Backtracking to find better solutions
    seed_counter = 0
    while time.time() - start_time < time_limit - 10.0:
        if best_result is not None:
            target = best_scope
        else:
            target = 200
        
        solve_attempt(target, seed_counter)
        seed_counter += 1
        
        # Run local search periodically
        if seed_counter % 3 == 0 and best_result is not None:
            local_search_improve(best_result, max_iters=2000)

    # Phase 3: Final local search with more iterations
    if best_result is not None and time.time() - start_time < time_limit - 2.0:
        local_search_improve(best_result, max_iters=10000)

    if best_result is None:
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
