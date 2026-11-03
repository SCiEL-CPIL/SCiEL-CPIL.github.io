import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    
    # 1. Greedy Baseline
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(max(r) for r in best_rows)
    
    deadline = time.time() + 110.0
    
    # 2. Backtracking Search
    # Try to find a solution with scope M
    def backtrack_search(M, time_limit):
        used = set()
        rows = []
        start_time = time.time()
        
        # We need to find n_rows rows.
        # To break symmetry, we can enforce that the max element of row i is <= max element of row i+1.
        # Or simply that the rows are generated in a specific order.
        # Let's just generate them.
        
        def dfs(rows_left, last_row_max):
            if time.time() - start_time > time_limit:
                return False, None
            
            if rows_left == 0:
                return True, rows[:]
            
            # Try to construct the next row
            # The row must have max element <= M
            # And to break symmetry, max element >= last_row_max? 
            # Actually, if we sort rows by max element, we can enforce non-decreasing max.
            
            # We need to pick 5 numbers x1..x5 such that 0 < x1 < ... < x5 <= M
            # And differences are valid.
            
            # Optimization: 
            # The available differences are those in [1, M] not in `used`.
            # We need 15 of them for this row.
            
            # Let's try to build the row greedily with backtracking on the points.
            # x1, x2, x3, x4, x5
            
            # We can iterate x1 from 1 to M
            # But that's slow.
            # Better: Pick a candidate set of differences?
            
            # Alternative: Iterate over possible x5 (max element) from last_row_max to M.
            # For a fixed x5, we need to find x1..x4.
            
            # This is still complex.
            
            # Let's try a simpler heuristic: 
            # Generate a random valid row (Golomb ruler) within [0, M] that doesn't conflict with `used`.
            # If found, add it and recurse.
            # If not, try a different one?
            # This is essentially a randomized backtracking.
            
            # To make it deterministic and fast, let's try to find ANY valid row.
            # We can iterate x5 from M down to last_row_max.
            # For each x5, try to find x4 < x5, etc.
            
            # Given the time constraint, let's implement a randomized restart backtracking.
            
            for _ in range(100): # 100 random attempts to pick a row
                # Pick random x5
                # Actually, let's just pick a random set of 5 points from 1..M
                # Check if it forms a valid row (internal diffs distinct, external diffs unused)
                # If yes, add to used and recurse.
                
                # This is too slow if M is large.
                pass
            
            return False, None
            
        # This simple backtracking is hard to optimize quickly in Python.
        # Let's rely on SA.
        
        return None

    # 3. Simulated Annealing
    def get_cost(rows):
        diffs = []
        max_val = 0
        for r in rows:
            max_val = max(max_val, r[-1])
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.append(r[j] - r[i])
        unique_diffs = len(set(diffs))
        collisions = len(diffs) - unique_diffs
        return collisions * 10000 + max_val, max_val, collisions

    def simulate_annealing(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        current_cost, current_max, current_collisions = get_cost(rows)
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max
        
        T = 100.0
        cooling = 0.9999
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            lower = rows[r_idx][i_idx - 1] + 1
            upper_limit = best_max + 20 
            if i_idx < row_len - 1:
                upper = rows[r_idx][i_idx + 1] - 1
                upper = min(upper, upper_limit)
            else:
                upper = upper_limit
            
            if lower > upper:
                continue
                
            new_val = random.randint(lower, upper)
            if new_val == rows[r_idx][i_idx]:
                continue
                
            old_val = rows[r_idx][i_idx]
            rows[r_idx][i_idx] = new_val
            
            new_cost, new_max, new_collisions = get_cost(rows)
            
            if new_cost < current_cost:
                current_cost = new_cost
                current_max = new_max
                current_collisions = new_collisions
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_max = new_max
                    best_local_rows = [r[:] for r in rows]
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if random.random() < 0.5:
                        current_cost = new_cost
                        current_max = new_max
                        current_collisions = new_collisions
                else:
                    if delta < 1000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    
                    if random.random() < p:
                        current_cost = new_cost
                        current_max = new_max
                        current_collisions = new_collisions
                    else:
                        rows[r_idx][i_idx] = old_val
            
            T *= cooling
            if T < 0.1:
                T = 0.1
                
        return best_local_rows, best_local_max, best_local_cost

    # Main Loop
    while time.time() < deadline:
        # Initialize with random rows
        init_rows = []
        # Use best_max as bound
        bound = best_max
        if bound < k:
            bound = k + 10
            
        for _ in range(n_rows):
            if bound >= k:
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)
            else:
                init_rows.append([0, 1, 2, 3, 4, 5]) # Dummy
                
        sa_time = 5.0
        if time.time() + sa_time > deadline:
            sa_time = deadline - time.time()
            if sa_time <= 0:
                break
                
        local_rows, local_max, local_cost = simulate_annealing(init_rows, sa_time)
        
        if local_cost < 10000: # Valid
            if local_max < best_max:
                best_max = local_max
                best_rows = local_rows
                if best_max <= 112:
                    # Found a very good solution
                    # We can break or continue
                    pass

    return best_rows
