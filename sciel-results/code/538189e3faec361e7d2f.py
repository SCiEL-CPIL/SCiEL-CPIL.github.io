import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 118.0  # Leave buffer for 120s limit
    
    def time_up():
        return time.time() - start_time > time_limit
    
    # Number of differences per row: C(k+1, 2)
    diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Greedy construction to get an initial upper bound
    def greedy_construct(target_scope, seed=0):
        rng = random.Random(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                found = False
                for x in range(row[-1] + 1, target_scope + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows
    
    # Local search to reduce scope
    def local_search(rows, target_scope, seed=0, max_iter=5000):
        rng = random.Random(seed)
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        best_scope = max(r[-1] for r in rows)
        current_rows = [r[:] for r in rows]
        no_improve_count = 0
        temp = 1.0
        cooling = 0.999
        
        for it in range(max_iter):
            if time_up():
                break
            if no_improve_count > 2000:
                break
            
            # Pick a random row to modify
            row_idx = rng.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            
            # Compute diffs used by other rows
            other_diffs = set()
            for i, r in enumerate(current_rows):
                if i == row_idx:
                    continue
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        other_diffs.add(r[b] - r[a])
            
            # Try to rebuild this row with smaller max
            old_max = row[-1]
            if old_max <= 1:
                continue
            
            # Try decreasing the last element
            for new_last in range(old_max - 1, max(old_max - 10, 1), -1):
                # Check if we can complete the row ending at new_last
                test_row = row[:-1] + [new_last]
                if not all(test_row[i] < test_row[i+1] for i in range(len(test_row)-1)):
                    continue
                
                new_diffs = set()
                valid = True
                for i in range(len(test_row)):
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        if d in other_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                
                if valid:
                    # Accept with some probability
                    delta = new_last - old_max  # negative means improvement
                    if delta < 0 or rng.random() < pow(2.718281828, delta / temp):
                        current_rows[row_idx] = test_row
                        best_scope = min(best_scope, max(r[-1] for r in current_rows))
                        no_improve_count = 0
                        break
            
            if current_rows[row_idx][-1] != old_max:
                # Recompute used_diffs if needed (lazy)
                pass
            
            temp *= cooling
        
        return current_rows
    
    # Backtracking with frame-based approach
    def backtrack_solve(target_scope, seed=0):
        rng = random.Random(seed)
        used_diffs = 0
        rows = []
        max_diff = target_scope  # max possible difference
        
        # Precompute valid differences bitmask
        valid_mask = (1 << (max_diff + 1)) - 1  # bits 0 to max_diff
        
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
            upper_bound = min(target_scope - 1, best_scope - 1) if best_scope < float('inf') else target_scope - 1
            
            if last_val >= upper_bound:
                return False
            
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                # Check conflicts using bitmask
                conflict = False
                new_mask = 0
                for a in current_row:
                    d = x - a
                    if d > max_diff:
                        continue
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                # Heuristic: prefer candidates that leave more options
                # Count how many future differences would be blocked
                cost = 0.0
                for a in current_row:
                    d = x - a
                    cost += 1.0 / d
                
                noise = rng.random() * 0.05
                candidates.append((cost + noise, x, new_mask))
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda item: item[0])
            
            # Try top candidates with randomization
            top_n = min(8, len(candidates))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            remaining = candidates[top_n:]
            
            for cost, x, new_mask in top_candidates + remaining:
                old_used = used_diffs
                used_diffs = used_diffs | new_mask
                
                if _backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used
            
            return False
        
        return _backtrack(0, [0])
    
    best_result = None
    best_scope = float('inf')
    
    # Phase 1: Greedy to get initial solution
    for seed in range(50):
        if time_up():
            break
        result = greedy_construct(200, seed)
        if result is not None:
            scope = max(r[-1] for r in result)
            if scope < best_scope:
                best_scope = scope
                best_result = result
    
    # Phase 2: Local search to improve
    if best_result is not None and not time_up():
        for seed in range(100):
            if time_up():
                break
            improved = local_search(best_result, best_scope, seed, max_iter=3000)
            if improved is not None:
                scope = max(r[-1] for r in improved)
                if scope < best_scope:
                    best_scope = scope
                    best_result = improved
    
    # Phase 3: Backtracking to find optimal or near-optimal
    if not time_up():
        target = best_scope if best_result is not None else 150
        for seed in range(200):
            if time_up():
                break
            found = backtrack_solve(target, seed)
            if found and best_scope < target:
                # Found better solution, try to improve further
                target = best_scope
    
    # Final polish: try to reduce scope by 1
    if best_result is not None and not time_up():
        polish_target = best_scope
        for seed in range(100):
            if time_up():
                break
            found = backtrack_solve(polish_target, seed)
            if found and best_scope < polish_target:
                polish_target = best_scope
    
    if best_result is None:
        # Ultimate fallback
        result = greedy_construct(300, 42)
        if result is None:
            # Very conservative fallback
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                    x += 1
                rows.append(row)
            return rows
        return result
    
    return best_result
