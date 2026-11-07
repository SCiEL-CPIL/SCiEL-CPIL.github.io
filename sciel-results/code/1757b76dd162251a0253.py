import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We stop at 85s.
    deadline = start_time + 85.0
    
    n = 256
    
    # 1. Precompute factor masks
    # Primes up to 256
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    factor_mask = [0] * (n + 1)
    for i in range(2, n + 1):
        mask = 0
        x = i
        for p in primes:
            if p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_bit[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_bit[x])
        factor_mask[i] = mask
    
    # factor_mask[1] = 0 (1 is coprime to all)

    def is_compatible(val, current_set):
        fm = factor_mask[val]
        if fm == 0:
            return True
        for v in current_set:
            if (fm & factor_mask[v]) != 0:
                return False
        return True

    # 2. Fallback Local Search
    def local_search(initial_set, time_limit):
        best_set = set(initial_set)
        best_sum = sum(best_set)
        candidates = list(range(n, 0, -1))
        end_time = time.time() + time_limit
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            
            # Phase 1: Add
            for val in candidates:
                if time.time() >= end_time:
                    break
                if val in best_set:
                    continue
                if is_compatible(val, best_set):
                    best_set.add(val)
                    best_sum += val
                    improved = True
            
            # Phase 2: Swap 1-for-1
            for remove_val in list(best_set):
                if time.time() >= end_time:
                    break
                reduced_set = best_set - {remove_val}
                reduced_sum = best_sum - remove_val
                
                for add_val in candidates:
                    if time.time() >= end_time:
                        break
                    if add_val in reduced_set:
                        continue
                    if is_compatible(add_val, reduced_set):
                        new_sum = reduced_sum + add_val
                        if new_sum > best_sum:
                            best_set = reduced_set
                            best_set.add(add_val)
                            best_sum = new_sum
                            improved = True
                            break
                            
            # Phase 3: Swap 1-for-2
            if time.time() < end_time:
                for remove_val in list(best_set):
                    if time.time() >= end_time:
                        break
                    reduced_set = best_set - {remove_val}
                    reduced_sum = best_sum - remove_val
                    
                    for add1 in candidates:
                        if time.time() >= end_time:
                            break
                        if add1 in reduced_set:
                            continue
                        if not is_compatible(add1, reduced_set):
                            continue
                        
                        temp_set = reduced_set | {add1}
                        temp_sum = reduced_sum + add1
                        
                        for add2 in candidates:
                            if time.time() >= end_time:
                                break
                            if add2 in temp_set:
                                continue
                            if not is_compatible(add2, temp_set):
                                continue
                            
                            new_sum = temp_sum + add2
                            if new_sum > best_sum:
                                best_set = temp_set
                                best_set.add(add2)
                                best_sum = new_sum
                                improved = True
                                break

        return best_set, best_sum

    # Generate initial solutions for fallback
    init_set_1 = {1}
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        if is_compatible(power, init_set_1):
            init_set_1.add(power)
            
    init_set_2 = {1}
    for val in range(n, 1, -1):
        if is_compatible(val, init_set_2):
            init_set_2.add(val)
            
    best_overall_set = set()
    best_overall_sum = 0

    # Run a quick fallback search to have a baseline result ready
    # We allocate 10s for this baseline to ensure we have a decent answer if MILP fails
    t_fallback_baseline = 10.0
    if time.time() < deadline - t_fallback_baseline:
        s1, sum1 = local_search(init_set_1, t_fallback_baseline)
        s2, sum2 = local_search(init_set_2, t_fallback_baseline)
        
        if sum1 > best_overall_sum:
            best_overall_set = s1
            best_overall_sum = sum1
        if sum2 > best_overall_sum:
            best_overall_set = s2
            best_overall_sum = sum2

    # 3. Try Exact ILP Solver
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Identify conflicting pairs
        conflicts = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if (factor_mask[i] & factor_mask[j]) != 0:
                    conflicts.append((i, j))
        
        num_vars = n
        num_constraints = len(conflicts)
        
        if num_constraints > 0:
            # Build A matrix
            # Using sparse matrix for efficiency
            import scipy.sparse as sp
            
            row_idx = []
            col_idx = []
            data = []
            
            for k, (i, j) in enumerate(conflicts):
                row_idx.append(k)
                col_idx.append(i - 1)
                data.append(1)
                
                row_idx.append(k)
                col_idx.append(j - 1)
                data.append(1)
                
            A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(num_constraints, num_vars))
            
            b_lower = np.zeros(num_constraints)
            b_upper = np.ones(num_constraints)
            
            linear_constraint = LinearConstraint(A, b_lower, b_upper)
            bounds = Bounds(np.zeros(num_vars), np.ones(num_vars))
            
            # Objective: maximize sum(i * x_i) -> minimize sum(-i * x_i)
            c = -np.arange(1, n + 1, dtype=float)
            
            integrality = np.ones(num_vars)
            
            # Time limit for MILP: use remaining time minus safety margin
            time_left = deadline - time.time() - 2.0 # 2s safety margin
            if time_left > 1.0:
                result = milp(c, constraints=linear_constraint, integrality=integrality, bounds=bounds, options={"time_limit": time_left})
                
                if result.status == 0 or result.status == 2: # 0: Optimal, 2: Iteration limit (best available)
                    sol = result.x
                    selected = []
                    for i in range(num_vars):
                        if sol[i] > 0.5:
                            selected.append(i + 1)
                    
                    # Ensure 1 is included if possible (it always is in optimal unless excluded by constraint, but 1 has no constraints)
                    if 1 not in selected:
                        selected.append(1)
                        
                    new_sum = sum(selected)
                    if new_sum > best_overall_sum:
                        best_overall_set = set(selected)
                        best_overall_sum = new_sum

    except Exception as e:
        # If MILP fails, we rely on the fallback result
        pass

    # If we still have time, do a few more random restarts with local search to polish
    if time.time() < deadline - 5.0:
        remaining_time = deadline - time.time() - 5.0
        # Do a few quick restarts
        restarts = 3
        t_per_restart = remaining_time / restarts
        for _ in range(restarts):
            if time.time() >= deadline - 5.0:
                break
            rand_set = {1}
            cands = list(range(n, 1, -1))
            random.shuffle(cands)
            for val in cands:
                if is_compatible(val, rand_set):
                    rand_set.add(val)
            
            rs, rsum = local_search(rand_set, t_per_restart)
            if rsum > best_overall_sum:
                best_overall_set = rs
                best_overall_sum = rsum

    # Final check: ensure non-empty and valid types
    result_list = [int(x) for x in best_overall_set]
    if not result_list:
        result_list = [1]
        
    return result_list
