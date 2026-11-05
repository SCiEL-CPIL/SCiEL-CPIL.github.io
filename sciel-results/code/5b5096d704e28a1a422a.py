import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute reciprocals
    recip = [0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i

    # Precompute all 3-APs for fast conflict checking
    ap_conflicts = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            if c > n:
                break
            # (a, b, c) is a 3-AP
            ap_conflicts[a].add(b)
            ap_conflicts[a].add(c)
            ap_conflicts[b].add(a)
            ap_conflicts[b].add(c)
            ap_conflicts[c].add(a)
            ap_conflicts[c].add(b)

    def can_add_to(x, s):
        """Check if x can be added to set s without creating a 3-AP."""
        for y in ap_conflicts[x]:
            if s[y]:
                return False
        return True

    def greedy_fill_array(s):
        """Add all valid elements in increasing order (prioritize small numbers)."""
        for x in range(1, n + 1):
            if not s[x]:
                if can_add_to(x, s):
                    s[x] = True
        return s

    def get_sum_array(s):
        return sum(recip[i] for i in range(1, n + 1) if s[i])

    def local_search_array(s, time_limit=10.0):
        """Local search: try removing one element and greedily refilling."""
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            current_sum = get_sum_array(s)
            
            # Try removing one element and refilling
            for rem in range(1, n + 1):
                if not s[rem]:
                    continue
                
                trial = s[:]
                trial[rem] = False
                greedy_fill_array(trial)
                new_sum = get_sum_array(trial)
                
                if new_sum > current_sum + 1e-12:
                    s = trial
                    current_sum = new_sum
                    improved = True
                    break
            
        return s

    def polish_array(s, time_limit=5.0):
        """Final polish: try both adding and removing elements."""
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            current_sum = get_sum_array(s)
            
            # Try adding elements
            for add in range(1, n + 1):
                if s[add]:
                    continue
                if can_add_to(add, s):
                    trial = s[:]
                    trial[add] = True
                    new_sum = get_sum_array(trial)
                    if new_sum > current_sum + 1e-12:
                        s = trial
                        current_sum = new_sum
                        improved = True
                        break
            
            # Try removing elements and refilling
            if not improved:
                for rem in range(1, n + 1):
                    if not s[rem]:
                        continue
                    trial = s[:]
                    trial[rem] = False
                    greedy_fill_array(trial)
                    new_sum = get_sum_array(trial)
                    if new_sum > current_sum + 1e-12:
                        s = trial
                        current_sum = new_sum
                        improved = True
                        break
        
        return s

    # 1. Initial Greedy
    in_set = [False] * (n + 1)
    greedy_fill_array(in_set)
    best_set = in_set[:]
    best_sum = get_sum_array(best_set)
    
    # 2. Local Search on Initial
    improved_set = local_search_array(best_set[:], time_limit=15.0)
    if get_sum_array(improved_set) > best_sum:
        best_set = improved_set
        best_sum = get_sum_array(best_set)
        
    # 3. Randomized Restart / Perturbation
    random.seed(42)
    
    while time.time() < start_time + 30.0:
        # Perturb best_set
        perturbed = best_set[:]
        
        # Remove 1 to 4 random elements
        num_removals = random.randint(1, 4)
        elements_in_set = [i for i in range(1, n + 1) if perturbed[i]]
        if not elements_in_set:
            break
            
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed[rem_val] = False
            elements_in_set.pop(idx)
        
        # Refill
        greedy_fill_array(perturbed)
        
        # Local Search
        improved_set = local_search_array(perturbed, time_limit=5.0)
        
        new_sum = get_sum_array(improved_set)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_set = improved_set

    # --- MILP Attempt ---
    try:
        # Generate constraints for all 3-APs
        rows = []
        cols = []
        vals = []
        b_vec = []
        
        for a in range(1, n + 1):
            for d in range(1, n):
                c = a + 2 * d
                if c > n:
                    break
                b = a + d
                # Constraint: x_a + x_b + x_c <= 2
                row_idx = len(b_vec)
                rows.extend([row_idx, row_idx, row_idx])
                cols.extend([a - 1, b - 1, c - 1])
                vals.extend([1.0, 1.0, 1.0])
                b_vec.append(2.0)
        
        from scipy.sparse import csr_matrix
        A = csr_matrix((vals, (rows, cols)), shape=(len(b_vec), n))
        
        # Objective: Minimize -sum(recip[i] * x_i)
        c_obj = np.array([-recip[i] for i in range(1, n + 1)])
        
        # Bounds: x_i in [0, 1]
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x_i are integers (binary)
        integrality = np.ones(n)
        
        # Constraints: A x <= b
        constraint = LinearConstraint(A, lb=-np.inf, ub=np.array(b_vec))
        
        # Solve with remaining time
        milp_time_limit = max(1.0, deadline - time.time() - 8.0)
        
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, 
                   bounds=bounds, options={"time_limit": milp_time_limit, "presolve": "default"})
        
        if res.status == 0 and res.x is not None:
            milp_set = [False] * (n + 1)
            for i in range(n):
                if res.x[i] > 0.5:
                    milp_set[i + 1] = True
            
            # Validate MILP result
            s_set = set(i for i in range(1, n+1) if milp_set[i])
            valid = True
            for x in s_set:
                for y in ap_conflicts[x]:
                    if y in s_set and x != y:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Use MILP solution as starting point for local search
                milp_polished = local_search_array(milp_set[:], time_limit=5.0)
                milp_polished = polish_array(milp_polished, time_limit=3.0)
                milp_sum = get_sum_array(milp_polished)
                if milp_sum > best_sum:
                    best_set = milp_polished
                    best_sum = milp_sum
                    
    except Exception:
        pass

    # Final polish on best solution
    final_set = polish_array(best_set[:], time_limit=3.0)
    final_sum = get_sum_array(final_set)
    if final_sum > best_sum:
        best_set = final_set
        best_sum = final_sum

    # Extract result
    result = [i for i in range(1, n + 1) if best_set[i]]
    
    # Validation
    s_set = set(result)
    for x in s_set:
        for y in ap_conflicts[x]:
            if y in s_set and x != y:
                raise AssertionError("3-AP found")
                    
    return result
