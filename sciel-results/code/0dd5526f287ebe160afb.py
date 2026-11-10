import time
import sys
import random
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds

sys.setrecursionlimit(10000)

DEADLINE = 0
TIME_LIMIT = 115  # Safety margin before 120s kill

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # 1. Generate Baseline using a simple greedy heuristic
    # This ensures we always have a valid solution to return if the advanced search fails or times out.
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []
    
    baseline_scope = max(max(row) for row in baseline_rows)
    
    # 2. Determine Search Range
    # Counting bound: n_rows * k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    lower_bound = n_rows * diffs_per_row
    
    # We search M from lower_bound up to baseline_scope.
    # The goal is to find the smallest M for which a valid set exists.
    
    best_found = baseline_rows
    
    # Optimization: If baseline is already very close to lower bound, 
    # the search space is small.
    
    # To handle time limits efficiently:
    # We iterate M from lower_bound to baseline_scope.
    # For each M, we formulate an ILP.
    # If the ILP finds a solution, we return it immediately (since we search M upwards).
    # If the ILP proves infeasibility, we move to M+1.
    
    # ILP Formulation:
    # Variables: x_i for each candidate row i (binary)
    # Objective: Minimize sum(x_i) ? No, we want to check feasibility for a fixed M.
    # Actually, for a fixed M, we just want to know if there EXISTS a set of n_rows 
    # non-conflicting rows.
    # So we can set the objective to 0 and add constraint sum(x_i) == n_rows.
    # Or, more robustly, we can maximize sum(x_i) and check if it's >= n_rows.
    # Maximizing is often better for solvers as it drives towards a solution.
    
    for M in range(lower_bound, baseline_scope + 1):
        if time.time() > DEADLINE - 5:
            break
        
        # Generate candidates for this M
        candidates = generate_candidates(M, k)
        
        if candidates is None:
            # Timeout or error in generation
            break
            
        n_cands = len(candidates)
        
        if n_cands < n_rows:
            continue
            
        # Build the ILP model
        # c = 0 (we don't care about the objective value, just feasibility/max size)
        # Actually, let's maximize the number of selected rows to be safe.
        c = [0] * n_cands
        
        # Constraints:
        # 1. Sum of selected rows must be at least n_rows.
        #    We can use a single linear constraint: sum(x) >= n_rows
        #    But milp requires LinearConstraint objects.
        #    A = [[1, 1, ..., 1]]
        #    lb = [n_rows], ub = [inf]
        
        # 2. Conflict constraints: For every pair of conflicting rows (i, j), x_i + x_j <= 1.
        #    This is O(n_cands^2) constraints, which might be too many for large n_cands.
        #    However, for (7,5) and M~112, n_cands is likely in the hundreds.
        #    100^2 = 10,000 constraints. HiGHS can handle this.
        
        # Let's precompute conflicts to build the constraint matrix efficiently.
        # Instead of building a dense matrix, we can use sparse representation or just list of tuples.
        # scipy.optimize.milp accepts LinearConstraint with array_like.
        
        # Let's try to build the conflict matrix A_conf.
        # Rows in A_conf correspond to conflict pairs.
        # Columns correspond to candidate rows.
        
        # To avoid building a huge dense array, we can generate the constraints on the fly?
        # No, milp needs the arrays upfront.
        
        # If n_cands is too large (e.g., > 500), n_cands^2 might be slow to build.
        # We will cap the number of candidates or skip if it's too big.
        if n_cands > 1000:
            # Fallback: Use the greedy baseline or a simpler heuristic for this M
            # Since we are searching for the MINIMUM M, if we can't solve it, we assume infeasible for now
            # and move to next M? No, that's risky.
            # But for (7,5), M=112, n_cands is likely < 1000.
            pass

        # Build conflict list
        conflicts = []
        masks = [c[1] for c in candidates]
        
        # Optimization: Only check conflicts if n_cands is manageable
        # For n_cands ~ 500, 125k checks. Fast in Python? Maybe 0.1s.
        # For n_cands ~ 2000, 2M checks. Might be 1s.
        
        for i in range(n_cands):
            mi = masks[i]
            for j in range(i + 1, n_cands):
                mj = masks[j]
                if mi & mj:
                    conflicts.append((i, j))
        
        # Build A matrix for milp
        # Total constraints = 1 (sum constraint) + len(conflicts)
        n_constraints = 1 + len(conflicts)
        
        # We need to build A as a list of lists or a sparse matrix.
        # milp accepts array-like.
        # Let's use a list of lists for simplicity, converting to numpy if needed.
        
        A = []
        lb = []
        ub = []
        
        # Constraint 1: sum(x) >= n_rows
        A.append([1] * n_cands)
        lb.append(n_rows)
        ub.append(n_cands) # max possible is n_cands
        
        # Constraint 2..N: x_i + x_j <= 1 for each conflict
        for i, j in conflicts:
            row = [0] * n_cands
            row[i] = 1
            row[j] = 1
            A.append(row)
            lb.append(0)
            ub.append(1)
        
        # Bounds for variables: 0 <= x_i <= 1
        bounds = Bounds(lb=[0] * n_cands, ub=[1] * n_cands)
        
        # Integrality: all variables are integer (binary)
        integrality = [1] * n_cands
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A=A, lb=lb, ub=ub),
                integrality=integrality,
                bounds=bounds
            )
        except Exception as e:
            # If ILP fails to setup or run, skip this M
            continue
        
        if res.status == 0: # Optimal
            # Check if we found a valid set of size n_rows
            x = res.x
            selected_indices = [i for i in range(n_cands) if x[i] > 0.5]
            
            if len(selected_indices) >= n_rows:
                # Extract rows
                rows = [list(candidates[i][2]) for i in selected_indices[:n_rows]]
                
                # Verify validity
                if is_valid(rows, n_rows, k):
                    return rows
        elif res.status == 2: # Infeasible
            # No solution for this M, continue to next M
            continue
        else:
            # Other status (e.g., timeout, unbounded)
            # If it's a timeout, we might have a partial solution? 
            # milp usually doesn't return partial solutions for status != 0.
            # We'll treat it as failure for this M and move on, or break if time is low.
            if time.time() > DEADLINE - 10:
                break

    return best_found

def is_valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, k+1):
            if row[i] <= row[i-1]:
                return False
        # Check diffs
        for i in range(k+1):
            for j in range(i+1, k+1):
                d = row[j] - row[i]
                if d in used_diffs:
                    return False
                used_diffs.add(d)
    return True

def generate_candidates(M, k):
    """
    Generates all valid rows of length k+1 starting at 0, with max element <= M.
    Returns list of (max_val, mask, row_tuple) or None if timeout.
    """
    start = time.time()
    timeout = 5.0 # 5 seconds per M step
    
    rows = []
    
    def backtrack(current_row, current_mask, last_val):
        if time.time() - start > timeout:
            return False
        
        if len(current_row) == k + 1:
            max_val = current_row[-1]
            rows.append((max_val, current_mask, tuple(current_row)))
            return True
        
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            # Check conflicts with existing elements in current_row
            for a in current_row:
                d = x - a
                bit = 1 << d
                if current_mask & bit:
                    valid = False
                    break
                new_mask |= bit
            
            if valid:
                current_row.append(x)
                if not backtrack(current_row, current_mask | new_mask, x):
                    return False
                current_row.pop()
        
        return True

    if backtrack([0], 0, 0):
        return rows
    else:
        return None

def greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows
