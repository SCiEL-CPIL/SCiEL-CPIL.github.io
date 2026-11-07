import math
import numpy as np
import time
import sys

def construct():
    n = 256
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # 5s safety margin
    
    # 1. Generate Primes and Prime Powers
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Prime Powers Solution (Baseline)
    pp_solution = [1]
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        pp_solution.append(power)
        
    pp_sum = sum(pp_solution)
    
    # 2. Attempt MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        num_vars = n
        # Objective: Maximize sum -> Minimize -sum
        c = -np.arange(1, n + 1, dtype=float)
        
        # Conflicts
        conflicts = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if math.gcd(i, j) > 1:
                    conflicts.append((i, j))
        
        num_constraints = len(conflicts)
        
        # Sparse matrix construction for efficiency
        rows = []
        cols = []
        vals = []
        
        for k, (i, j) in enumerate(conflicts):
            rows.append(k)
            cols.append(i - 1)
            vals.append(1.0)
            
            rows.append(k)
            cols.append(j - 1)
            vals.append(1.0)
            
        from scipy.sparse import csr_matrix
        A = csr_matrix((vals, (rows, cols)), shape=(num_constraints, num_vars))
        
        b_lower = np.zeros(num_constraints)
        b_upper = np.ones(num_constraints)
        
        linear_constraint = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(np.zeros(num_vars), np.ones(num_vars))
        
        # Integrality
        integrality = np.ones(num_vars)
        
        # Warm Start: Prime Powers
        init_sol = np.zeros(num_vars)
        for val in pp_solution:
            init_sol[val - 1] = 1.0
            
        # Check time
        if time.time() - start_time > timeout - 10.0:
            return [int(x) for x in pp_solution]
            
        result = milp(
            c, 
            constraints=linear_constraint, 
            integrality=integrality, 
            bounds=bounds,
            initial_solution=init_sol,
            options={'time_limit': 60, 'output_flag': 0}
        )
        
        best_solution = [int(x) for x in pp_solution]
        best_sum = pp_sum
        
        if result.status == 0:
            sol = result.x
            selected = [int(i + 1) for i in range(num_vars) if sol[i] > 0.5]
            # Verify validity just in case
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if math.gcd(selected[i], selected[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid and sum(selected) > best_sum:
                best_solution = selected
                best_sum = sum(selected)
                
        # 3. Local Search Polish
        # Try to improve the best solution by swapping or adding/removing elements
        if time.time() - start_time < timeout - 5.0:
            current_sol = set(best_solution)
            current_sum = best_sum
            improved = True
            while improved and time.time() - start_time < timeout - 2.0:
                improved = False
                
                # Try adding a new number
                for new_val in range(1, n + 1):
                    if new_val in current_sol:
                        continue
                    # Check if it conflicts with any existing
                    conflict = False
                    for existing in current_sol:
                        if math.gcd(new_val, existing) > 1:
                            conflict = True
                            break
                    if not conflict:
                        current_sol.add(new_val)
                        current_sum += new_val
                        improved = True
                        break
                
                # Try swapping one element for another
                if not improved:
                    # Sort by value descending to prioritize replacing small numbers
                    sorted_sol = sorted(current_sol, reverse=True)
                    for i, old_val in enumerate(sorted_sol):
                        # Try replacing old_val with a larger value
                        # We can only replace if the new value doesn't conflict with others
                        # and the new value is larger than old_val
                        for new_val in range(old_val + 1, n + 1):
                            if new_val in current_sol:
                                continue
                            # Check conflicts with all except old_val
                            conflict = False
                            for existing in current_sol:
                                if existing == old_val:
                                    continue
                                if math.gcd(new_val, existing) > 1:
                                    conflict = True
                                    break
                            if not conflict:
                                current_sol.remove(old_val)
                                current_sol.add(new_val)
                                current_sum += new_val - old_val
                                improved = True
                                break
                        if improved:
                            break
                
                # Try removing an element if it allows adding multiple larger ones? (Too complex, skip for now)
                # Or try swapping two elements for two others? (Too complex)
                
            if current_sum > best_sum:
                best_solution = list(current_sol)
                best_sum = current_sum
                
        return [int(x) for x in best_solution]
            
    except Exception as e:
        # Fallback to Prime Powers if anything goes wrong
        return [int(x) for x in pp_solution]
