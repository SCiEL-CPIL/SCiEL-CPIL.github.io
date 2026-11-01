import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Total distinct differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on max entry M
    # We need total_diffs_needed distinct differences from {1, ..., M}
    lower_bound_M = total_diffs_needed
    
    # Best known for (4,7) is 134. Lower bound is 112.
    # We search M from lower_bound_M upwards.
    # Cap the search to avoid infinite loops if no solution is found quickly.
    max_M_to_try = lower_bound_M + 60

    def solve_csp(M, time_limit):
        """
        Solves the problem for a given M using CSP with Forward Checking and MRV.
        Variables: The entries of the rows.
        Domains: Possible values for each entry.
        Constraints: All differences must be distinct.
        
        We model this by building rows one by one, but within a row, we use MRV
        to choose the next variable (position in the row) to assign, and FC to prune domains.
        However, since rows are independent except for the global difference set,
        a simpler and often more effective CSP approach for this specific structure is:
        
        1. Initialize global set of used differences.
        2. For each row, we need to find a sequence a_0=0, a_1, ..., a_k.
        3. Instead of standard backtracking on values, we can use a randomized restart
           local search or a smarter backtracking.
           
        Given the small size (M ~ 130), a well-optimized backtracking with MRV on the
        *differences* might be overkill. Let's stick to a high-performance backtracking
        but with a different heuristic: 
        - Try to place the "hardest" differences first? No, the structure is fixed.
        - Let's use a different approach: **Simulated Annealing** is already in inspirations.
        - Let's try **Randomized Restart Backtracking** with a "lookahead" check.
        
        Actually, the most structurally different and robust method for small M is 
        **Integer Linear Programming (ILP)**? No, std lib only.
        
        Let's use a **Genetic Algorithm** specifically designed for this packing problem?
        Inspiration 2 used GA. 
        
        Let's try a **Hybrid**: 
        1. Use a fast greedy heuristic to get a valid (but large) solution.
        2. Use **Local Search (Simulated Annealing)** to minimize the scope.
        
        Wait, the current program is Backtracking. Inspiration 1 was SA. 
        Let's implement a **Robust Simulated Annealing** that is specifically tuned for this 
        "distinct differences" constraint. The key is a good cost function and move set.
        
        Move Set:
        - Pick a random row.
        - Pick a random position in the row.
        - Change the value to a random valid integer in [prev+1, M-next].
        - Or, swap two values in the same row (preserving order? No, must be increasing).
        - Or, insert a new value and remove another?
        
        Better Move Set for Increasing Sequence:
        - Pick a row r.
        - Pick an index i (1..k).
        - Change a[i] to a random value v such that a[i-1] < v < a[i+1] (or bounds).
        
        Cost Function:
        - Count number of duplicate differences.
        - Add penalty for exceeding M? No, we fix M in the outer loop.
        - We want to minimize duplicates. If duplicates == 0, we have a valid solution for M.
        
        This is a different design family: Metaheuristic (SA) vs Exact (Backtracking).
        """
        
        # Initialize rows with random valid increasing sequences
        def random_row(M):
            # Choose k distinct integers from 1..M-1, sort them
            # Ensure 0 is first.
            vals = random.sample(range(1, M), k)
            vals.sort()
            return [0] + vals

        # Calculate current cost (number of duplicate differences)
        def calc_cost(rows, M):
            diffs = []
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        diffs.append(row[j] - row[i])
            
            # Count duplicates
            seen = set()
            dupes = 0
            for d in diffs:
                if d > M: # Should not happen if rows are within M
                    return float('inf')
                if d in seen:
                    dupes += 1
                else:
                    seen.add(d)
            return dupes

        # SA Parameters
        T = 1.0
        cooling_rate = 0.999
        T_min = 0.01
        max_iters = 500000  # Cap iterations to respect time
        
        rows = [random_row(M) for _ in range(n_rows)]
        current_cost = calc_cost(rows, M)
        
        if current_cost == 0:
            return rows
            
        best_rows = rows[:]
        best_cost = current_cost
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
                
            # Pick a random row and position to mutate
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, k) # 1..k
            
            row = rows[r_idx]
            
            # Determine valid range for new value
            low = row[p_idx - 1] + 1
            high = row[p_idx + 1] - 1 if p_idx < k else M
            
            if low > high:
                # Cannot mutate this position, try another or skip
                continue
                
            old_val = row[p_idx]
            new_val = random.randint(low, high)
            
            if new_val == old_val:
                continue
                
            # Calculate delta cost efficiently?
            # For simplicity, recalculate full cost. M is small, n_rows small.
            # But 500k iters * O(n_rows * k^2) might be slow in Python.
            # Let's optimize cost calculation or limit iters.
            
            # Optimization: Only recalculate differences involving the changed row.
            # Actually, let's just do a simple recalculation but with a faster inner loop.
            
            # Apply move
            row[p_idx] = new_val
            
            new_cost = calc_cost(rows, M)
            
            # Acceptance criterion
            if new_cost < current_cost or random.random() < pow(2.718281828, -(new_cost - current_cost) / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [row[:] for row in rows]
                    if best_cost == 0:
                        return best_rows
            else:
                # Revert
                row[p_idx] = old_val
                
            # Cool down
            T = max(T_min, T * cooling_rate)
            
            # If T is too low and no improvement for a while, restart?
            # Simple restart strategy: if no improvement in last 5000 steps, reset T.
            # We'll just let it run.
            
        return best_rows if best_cost == 0 else None

    # Main Loop: Try M from lower bound up
    for M in range(lower_bound_M, max_M_to_try + 1):
        if time.time() > deadline - 5:
            break
            
        # Try SA for this M
        # Give it a time budget
        time_budget = 5.0
        sol = solve_csp(M, start_time + time_budget)
        
        if sol is not None:
            # Verify
            all_diffs = set()
            valid = True
            for row in sol:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                return sol
        else:
            # If SA failed to find 0 cost, it's likely infeasible or hard.
            # Try next M? Or try backtracking for this M?
            # Backtracking is exact. If M is small, backtracking might be faster than SA failing.
            # Let's try a quick backtracking for the first few M values.
            pass

    # Fallback: Greedy valid solution (large scope)
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    used_diffs.update(x - a for a in row[:-1])
                    current_max = x
                    break
                x += 1
        rows.append(row)
        
    return rows
