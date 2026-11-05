import time
import random
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        if not subset:
            return True
        s_set = set(subset)
        n = len(subset)
        for i in range(n):
            a = subset[i]
            for j in range(i, n):
                c = a + subset[j]
                if c in s_set:
                    return False
        return True

    def build_constraints(elements):
        """Build all constraint triples (i,j,k) where elements[i]+elements[j]=elements[k]."""
        n = len(elements)
        val_to_idx = {v: i for i, v in enumerate(elements)}
        constraint_set = set()
        
        for i in range(n):
            for j in range(i, n):  # Only need i <= j to avoid duplicates
                c = elements[i] + elements[j]
                if c in val_to_idx:
                    k = val_to_idx[c]
                    # Store sorted tuple for deduplication
                    constraint_set.add(tuple(sorted((i, j, k))))
        
        return list(constraint_set)

    def solve_ilp(elements, time_budget):
        """Solve the sum-free subset problem exactly using ILP."""
        n = len(elements)
        if n == 0:
            return []
        
        constraints = build_constraints(elements)
        
        if not constraints:
            return elements[:]
        
        m = len(constraints)
        
        # Build sparse constraint matrix
        rows = []
        cols = []
        data = []
        
        for idx, (i, j, k) in enumerate(constraints):
            rows.extend([idx, idx, idx])
            cols.extend([i, j, k])
            data.extend([1, 1, 1])
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(m, n))
        
        # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
        c = -np.ones(n)
        
        # Constraints: A x <= 2
        constraints_obj = LinearConstraint(A, -np.inf, 2)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: x_i must be 0 or 1
        integrality = np.ones(n)
        
        # Solve with time limit
        try:
            result = milp(
                c=c,
                constraints=constraints_obj,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": time_budget}
            )
            
            if result.status == 0 or result.status == 2:  # Optimal or feasible
                x = result.x
                # Extract selected elements
                selected = [elements[i] for i in range(n) if x[i] > 0.5]
                return selected
            else:
                # If no solution found, return empty
                return []
        except Exception:
            return []

    def greedy_max_abs(S):
        S_sorted = sorted(S, key=lambda x: abs(x), reverse=True)
        C = []
        C_set = set()
        for x in S_sorted:
            if x not in C_set:
                # Check if adding x maintains sum-free property
                valid = True
                if 2 * x in C_set:
                    valid = False
                else:
                    for c in C:
                        if x + c in C_set:
                            valid = False
                            break
                        if x - c in C_set:
                            valid = False
                            break
                if valid:
                    C.append(x)
                    C_set.add(x)
        return C

    def greedy_min_abs(S):
        S_sorted = sorted(S, key=lambda x: abs(x))
        C = []
        C_set = set()
        for x in S_sorted:
            if x not in C_set:
                valid = True
                if 2 * x in C_set:
                    valid = False
                else:
                    for c in C:
                        if x + c in C_set:
                            valid = False
                            break
                        if x - c in C_set:
                            valid = False
                            break
                if valid:
                    C.append(x)
                    C_set.add(x)
        return C

    def greedy_value(S, reverse=True):
        S_sorted = sorted(S, key=lambda x: x, reverse=reverse)
        C = []
        C_set = set()
        for x in S_sorted:
            if x not in C_set:
                valid = True
                if 2 * x in C_set:
                    valid = False
                else:
                    for c in C:
                        if x + c in C_set:
                            valid = False
                            break
                        if x - c in C_set:
                            valid = False
                            break
                if valid:
                    C.append(x)
                    C_set.add(x)
        return C

    def local_search(S, initial_subset, time_limit):
        C = list(initial_subset)
        C_set = set(C)
        remaining = [x for x in S if x not in C_set]
        
        if not is_sum_free(C):
            return []

        best_size = len(C)
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
            improved = False
            
            # Try to insert any element
            for x in list(remaining):
                if time.time() - start_time > time_limit:
                    break
                # Check if x can be added
                if x in C_set:
                    continue
                valid = True
                if 2 * x in C_set:
                    valid = False
                else:
                    for c in C:
                        if x + c in C_set:
                            valid = False
                            break
                        if x - c in C_set:
                            valid = False
                            break
                if valid:
                    C.append(x)
                    C_set.add(x)
                    remaining.remove(x)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue

            # Try swap: Remove a, Add x
            found_move = False
            for i, a in enumerate(C):
                if time.time() - start_time > time_limit:
                    break
                if found_move:
                    break
                
                C_set.discard(a)
                C_no_a = C[:i] + C[i+1:]
                
                added_any = False
                added_elements = []
                
                for x in list(remaining):
                    if time.time() - start_time > time_limit:
                        break
                    if x in C_set:
                        continue
                    valid = True
                    if 2 * x in C_set:
                        valid = False
                    else:
                        for c in C_no_a:
                            if x + c in C_set:
                                valid = False
                                break
                            if x - c in C_set:
                                valid = False
                                break
                    if valid:
                        C_no_a.append(x)
                        C_set.add(x)
                        remaining.remove(x)
                        added_any = True
                        added_elements.append(x)
                        
                if added_any:
                    C = C_no_a
                    found_move = True
                    no_improve_count = 0
                else:
                    C_set.add(a)
            
            if not found_move:
                no_improve_count += 1
        
        return C

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        
        # Generate structural candidates
        S_odd = [x for x in S if x % 2 != 0]
        S_mod1 = [x for x in S if x % 3 == 1]
        S_mod2 = [x for x in S if x % 3 == 2]
        
        M = max(S)
        m = min(S)
        S_upper = [x for x in S if x > M / 2.0]
        S_lower = [x for x in S if x < m / 2.0]
        
        S_greedy_max = greedy_max_abs(S)
        S_greedy_min = greedy_min_abs(S)
        S_greedy_val_desc = greedy_value(S, reverse=True)
        S_greedy_val_asc = greedy_value(S, reverse=False)
        
        S_pos = [x for x in S if x > 0]
        S_neg = [x for x in S if x < 0]
        
        S_sorted_val = sorted(S)
        half = len(S_sorted_val) // 2
        S_top_half = S_sorted_val[half:]
        S_bottom_half = S_sorted_val[:half]
        
        abs_vals = sorted(set(abs(x) for x in S))
        if abs_vals:
            S_mag_75 = [x for x in S if abs(x) > abs_vals[len(abs_vals)*3//4]]
            S_mag_50 = [x for x in S if abs(x) > abs_vals[len(abs_vals)//2]]
            S_mag_25 = [x for x in S if abs(x) > abs_vals[len(abs_vals)//4]]
        else:
            S_mag_75 = []
            S_mag_50 = []
            S_mag_25 = []
            
        candidates = [S_odd, S_mod1, S_mod2, S_upper, S_lower, 
                      S_greedy_max, S_greedy_min, S_greedy_val_desc, S_greedy_val_asc,
                      S_pos, S_neg, S_top_half, S_bottom_half, S_mag_75, S_mag_50, S_mag_25]
        
        valid_candidates = []
        for cand in candidates:
            if is_sum_free(cand):
                valid_candidates.append(cand)
        
        if not valid_candidates:
            valid_candidates = [S_odd]
        
        best = max(valid_candidates, key=len)
        
        # Local search improvement
        if time.time() - start_time < DEADLINE - 10:
            improved = local_search(S, best, DEADLINE - 5)
            if len(improved) > len(best):
                best = improved
        
        # Now try ILP polish if we have time
        elapsed = time.time() - start_time
        ilp_time = min(remaining_time - 5, 30)  # Cap ILP at 30s, leave 5s safety
        if ilp_time > 2:
            ilp_result = solve_ilp(S, ilp_time)
            if ilp_result and is_sum_free(ilp_result) and len(ilp_result) > len(best):
                best = ilp_result
        
        # Final safety check
        if not is_sum_free(best):
            best = S_odd
        
        results.append(best)
    
    return results
