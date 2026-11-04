import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # 1. Greedy Baseline
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    # 2. DFS Solver with Bitsets
    def dfs_solve(M, time_limit, seed):
        random.seed(seed)
        global_deadline = time.time() + time_limit
        used_diffs = 0  # Bitset of used differences
        rows = []
        
        def solve_row(row_idx):
            if time.time() > global_deadline:
                raise TimeoutError()
            
            if row_idx == n_rows:
                return True
            
            # Generator for valid rows given current global_used_diffs
            def generate_rows(global_used_mask):
                def build(pos, vals, local_mask):
                    if time.time() > global_deadline:
                        return
                    
                    if pos == k + 1:
                        yield (vals, local_mask)
                        return
                    
                    last = vals[-1]
                    # Candidates: last+1 to M
                    candidates = list(range(last + 1, M + 1))
                    # Randomize to explore different branches
                    random.shuffle(candidates)
                    
                    for v in candidates:
                        new_mask = 0
                        conflict = False
                        # Check internal conflicts and global conflicts
                        for u in vals:
                            d = v - u
                            if (local_mask >> d) & 1:
                                conflict = True
                                break
                            if (global_used_mask >> d) & 1:
                                conflict = True
                                break
                            new_mask |= (1 << d)
                        
                        if conflict:
                            continue
                        
                        vals.append(v)
                        # Recurse
                        yield from build(pos + 1, vals, local_mask | new_mask)
                        vals.pop()
                
                # Start with [0]
                yield from build(1, [0], 0)

            # Try each generated row
            for row, row_mask in generate_rows(used_diffs):
                old_used = used_diffs
                used_diffs = old_used | row_mask
                rows.append(row)
                
                if solve_row(row_idx + 1):
                    return True
                
                # Backtrack
                rows.pop()
                used_diffs = old_used
                
            return False

        try:
            if solve_row(0):
                return [list(r) for r in rows]
            return None
        except TimeoutError:
            return None

    # 3. Search for better M
    # Lower bound: total distinct differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # We try M from lower_bound to best_max - 1
    # We use small time slices for each M
    # If we find a solution, we update best and break (since we are going up)
    
    # To ensure we don't spend all time on M=lower_bound if it's hard,
    # we limit attempts.
    
    m_values = list(range(lower_bound, best_max))
    
    # Shuffle M values? 
    # If we go ascending, the first found is the minimum.
    # But if lower_bound is infeasible, we waste time.
    # However, infeasibility is often detected quickly (no rows generated).
    # So ascending is good.
    
    for M in m_values:
        if time.time() > deadline - 5.0:
            break
        
        # Try a few seeds for this M
        found = False
        for i in range(2): # 2 attempts
            if time.time() > deadline - 5.0:
                break
            seed = M * 1000 + i
            sol = dfs_solve(M, 1.5, seed)
            if sol:
                new_max = get_max_entry(sol)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = sol
                found = True
                break
        
        if found:
            # Found a solution for this M. Since we iterate ascending, this is the min M found.
            # We can stop searching for smaller M? 
            # We already tried smaller M and failed (or ran out of time).
            # If we failed due to time, a smaller M might still have a solution we missed.
            # But given the time budget, we assume the first found is good enough.
            # To be safe, we can continue to check if there's time?
            # No, we want to minimize. If we found M=125, we don't need to find M=126.
            # We DO want to know if M=124 works.
            # Since we iterate ascending, we tried 112...124 before 125.
            # If they failed, we assume they don't work or we didn't find them.
            # So we can break.
            break

    return best_rows
