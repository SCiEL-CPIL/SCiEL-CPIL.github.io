import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Row length is k + 1
    row_len = k + 1
    diffs_per_row = row_len * (row_len - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for M
    # We need total_diffs distinct positive integers <= M
    # So M >= total_diffs
    M_lower = total_diffs
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)

    # Backtracking solver
    def solve(M, time_limit, seed=0):
        random.seed(seed)
        global_used = 0
        rows = []
        
        # Pre-compute nothing, just use bitsets
        
        def backtrack(row_idx):
            if time.time() > time_limit:
                raise TimeoutError()
            
            if row_idx == n_rows:
                return True
            
            # Build row row_idx
            # We try to build the row from left to right (increasing values)
            # Start with 0
            current_row = [0]
            local_used = 0
            
            def build_row(pos, last_val):
                if time.time() > time_limit:
                    raise TimeoutError()
                
                if pos == row_len:
                    # Row complete
                    # Check global conflict (should be checked during build, but double check)
                    if local_used & global_used:
                        return False
                    rows.append(list(current_row))
                    # Recurse
                    res = backtrack(row_idx + 1)
                    if res:
                        return True
                    rows.pop()
                    return False
                
                # Try next values
                # Heuristic: try values close to last_val + 1 first?
                # Or randomize?
                # For minimizing M, smaller values are generally better.
                # But we might need to skip some to avoid conflicts.
                
                # Range of v
                start_v = last_val + 1
                end_v = M
                
                # To avoid getting stuck in one path, we can shuffle the candidates
                # but keep a bias towards smaller values.
                # Simple: just iterate 1..M. If it fails, it fails.
                # Randomized restarts help here.
                
                for v in range(start_v, end_v + 1):
                    # Check conflicts with current_row
                    new_diffs = 0
                    conflict = False
                    for u in current_row:
                        d = v - u
                        if (local_used >> d) & 1:
                            conflict = True
                            break
                        if (global_used >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    
                    if conflict:
                        continue
                    
                    # Try this value
                    current_row.append(v)
                    res = build_row(pos + 1, v)
                    if res:
                        return True
                    current_row.pop()
                
                return False
            
            return build_row(1, 0)

        try:
            if backtrack(0):
                return rows
        except TimeoutError:
            pass
        except RecursionError:
            pass
        return None

    # Main search loop
    best_rows = None
    best_max = float('inf')
    
    # Try M from lower bound upwards
    # We limit the number of M values we try to avoid spending too much time on infeasible small M
    # However, if M is infeasible, backtracking should prune quickly? Not necessarily.
    # So we allocate a time slice per M.
    
    current_M = M_lower
    # Max M to try? Maybe M_lower + 50?
    # If we don't find it, we fall back to a greedy solution.
    
    max_M_to_try = M_lower + 100
    
    while current_M <= max_M_to_try and time.time() < deadline - 5:
        # Allocate time for this M
        # If current_M is close to lower bound, give it more time?
        # Or just fixed slice?
        # Let's give 2 seconds per M initially.
        time_slice = 2.0
        # If we have a lot of time left, increase slice
        remaining = deadline - time.time()
        if remaining > 20:
            time_slice = min(remaining / 10, 5.0)
        
        t_start = time.time()
        t_limit = t_start + time_slice
        
        # Try multiple seeds for this M
        found = False
        for seed in range(5):
            if time.time() > t_limit:
                break
            sol = solve(current_M, t_limit, seed=seed)
            if sol:
                best_rows = sol
                best_max = current_M
                found = True
                break
        
        if found:
            # We found a solution for current_M.
            # Can we try current_M - 1? No, we started from lower bound.
            # So this is the first M we found?
            # Wait, if we started from M_lower, and found a solution, it's the best possible (assuming M_lower is tight).
            # If M_lower was too low, we would have failed for M_lower, M_lower+1, etc.
            # So the first success is the minimum M found by this search.
            # We can break and try to improve with local search?
            # Or just return.
            # Let's break and do a quick local search to see if we can reduce max entry further?
            # No, the constraint is "largest entry as small as possible".
            # If we found a solution with max entry M, we can't have a solution with max entry < M 
            # if we've already proven M-1 is infeasible?
            # We didn't prove it, we just timed out.
            # So we should try to see if M-1 is possible with more time?
            # But we are moving UP from M_lower.
            # So if we succeeded at M, we haven't tried M-1?
            # No, we tried M_lower, M_lower+1, ..., M-1 and failed (timed out).
            # So M is the best we found.
            
            # We can try to polish the solution with local search to see if we can reduce the max entry of the *current* rows to allow a smaller global M?
            # No, the global M is the max of all rows.
            # If we reduce a row's max, the global max might drop.
            # But the validity condition is about differences.
            # Local search: try to replace a row with a "smaller" one.
            
            break
        else:
            current_M += 1
    
    if best_rows is None:
        # Fallback: Greedy
        def greedy():
            used = 0
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < row_len:
                    x += 1
                    new_diffs = 0
                    conflict = False
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    if not conflict:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            return rows
        
        best_rows = greedy()
        best_max = get_max_entry(best_rows)
        
    # Final Validation
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != row_len:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    if not validate(best_rows):
        # Should not happen, but fallback
        def greedy():
            used = 0
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < row_len:
                    x += 1
                    new_diffs = 0
                    conflict = False
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    if not conflict:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            return rows
        best_rows = greedy()

    return best_rows
