"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    
    # Greedy construction: add numbers in order, skip if it creates a 3-AP
    selected = []
    selected_set = set()
    for x in range(1, N + 1):
        can_add = True
        for a in selected:
            c = 2 * x - a
            if c > x and c in selected_set:
                can_add = False
                break
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in selected_set and b != a and b != x:
                    can_add = False
                    break
        if can_add:
            selected.append(x)
            selected_set.add(x)
    
    def compute_weight(sol):
        return sum(Fraction(1, x) for x in sol)
    
    current_weight = compute_weight(selected)
    best_weight = current_weight
    best_solution = list(selected)
    
    # 1. Quick Local Search to improve baseline
    # Precompute conflicts for fast validity check
    # conflict[i][j] = True if i,j are part of a 3-AP with some k in [1,N]
    # Actually, for local search, we just need to check if adding x to set S creates an AP.
    # We can precompute for each x, the pairs (a,b) in S that would form an AP with x.
    
    # Simple local search: single swaps
    improved = True
    while improved and time.time() < deadline - 30: # Reserve 30s for MILP
        improved = False
        sol_set = set(best_solution)
        
        # Try adding a new element and removing an old one (Swap)
        # Or just removing one and greedily filling? 
        # Let's stick to swaps for speed.
        for idx in range(len(best_solution)):
            if time.time() > deadline - 30:
                break
            removed = best_solution[idx]
            for new_elem in range(1, N + 1):
                if new_elem == removed or new_elem in sol_set:
                    continue
                # Check if swap is valid
                # New set: (best_solution - {removed}) U {new_elem}
                # We need to check if new_elem forms AP with any pair in (best_solution - {removed})
                # Or if any pair in (best_solution - {removed}) forms AP with new_elem
                # Essentially, check if new_elem can be added to (best_solution - {removed})
                trial_set = set(best_solution)
                trial_set.remove(removed)
                
                valid = True
                for a in trial_set:
                    # a, new_elem, c
                    c = 2 * new_elem - a
                    if c > new_elem and c in trial_set:
                        valid = False
                        break
                    # a, b, new_elem
                    s = a + new_elem
                    if s % 2 == 0:
                        b = s // 2
                        if b in trial_set and b != a and b != new_elem:
                            valid = False
                            break
                    # new_elem, b, a
                    c = 2 * a - new_elem
                    if c < a and c in trial_set:
                        valid = False
                        break
                if valid:
                    trial_weight = best_weight - Fraction(1, removed) + Fraction(1, new_elem)
                    if trial_weight > best_weight:
                        best_weight = trial_weight
                        best_solution.remove(removed)
                        best_solution.append(new_elem)
                        sol_set = set(best_solution)
                        improved = True
    
    # 2. MILP Formulation
    # Variables x_1 ... x_N
    # Max sum(1/i * x_i)
    # Subject to x_a + x_b + x_c <= 2 for all 3-APs (a,b,c)
    # x_i in {0,1}
    
    if time.time() < deadline - 5:
        try:
            # Build constraint matrix
            # Rows: constraints, Cols: variables
            rows = []
            cols = []
            data = []
            
            # Generate all 3-APs
            # a, a+d, a+2d
            # a in 1..N, d in 1..N//2
            # a+2d <= N
            for d in range(1, N // 2 + 1):
                max_a = N - 2 * d
                for a in range(1, max_a + 1):
                    b = a + d
                    c = a + 2 * d
                    # Indices are 0-based in the array
                    idx_a = a - 1
                    idx_b = b - 1
                    idx_c = c - 1
                    
                    rows.append(len(data))
                    cols.append(idx_a)
                    data.append(1)
                    
                    rows.append(len(data))
                    cols.append(idx_b)
                    data.append(1)
                    
                    rows.append(len(data))
                    cols.append(idx_c)
                    data.append(1)
            
            A = sparse.csr_matrix((data, (rows, cols)), shape=(len(data) // 3, N))
            b_vec = np.full(len(data) // 3, 2.0)
            
            # Objective: Minimize -sum(1/i * x_i)
            c_vec = -np.array([1.0 / i for i in range(1, N + 1)])
            
            # Bounds: 0 <= x_i <= 1
            bounds = Bounds(lb=0, ub=1)
            
            # Integrality: 1 for all (binary)
            integrality = np.ones(N)
            
            # Linear Constraint: A @ x <= b
            # milp expects LinearConstraint(A, lb, ub)
            # Here A @ x <= b  =>  A @ x is between -inf and b
            constraints = LinearConstraint(A, lb=-np.inf, ub=b_vec)
            
            # Run MILP
            # time_limit in seconds
            time_left = deadline - time.time() - 2
            if time_left > 1:
                res = milp(c=c_vec, constraints=constraints, integrality=integrality, bounds=bounds, 
                           options={"time_limit": time_left, "mip_rel_gap": 0.01})
                
                if res.status == 0: # Optimal
                    x_sol = res.x
                    # Extract solution
                    milp_solution = [i + 1 for i in range(N) if x_sol[i] > 0.5]
                    if milp_solution:
                        milp_weight = compute_weight(milp_solution)
                        if milp_weight > best_weight:
                            best_weight = milp_weight
                            best_solution = milp_solution
                            
        except Exception:
            pass # If MILP fails, keep local search result
    
    # Final verification
    result = sorted(best_solution)
    result_set = set(result)
    # Quick check for 3-APs
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            c = 2 * b - a
            if c > b and c in result_set:
                # Invalid, fallback to greedy
                result = sorted(selected)
                break
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if c > a and c < b and c in result_set:
                    result = sorted(selected)
                    break
            c = 2 * a - b
            if c < a and c in result_set:
                result = sorted(selected)
                break
        else:
            continue
        break
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
