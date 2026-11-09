"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    N = 256
    
    # 1. Precompute prime factors and compatibility
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    # Build conflict graph using bitsets
    # conflict[i] is a bitset of all j that conflict with i (including i itself)
    conflict = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if i == j:
                conflict[i] |= (1 << i)
            elif prime_factors[i] & prime_factors[j]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # weights[i] = i (the value of including integer i)
    weights = list(range(N + 1))
    
    def mask_to_list(mask):
        return [i for i in range(N + 1) if mask & (1 << i)]
    
    def greedy_initial(order):
        """Greedy construction with given order."""
        result_mask = 0
        for num in order:
            if (result_mask & conflict[num]) == 0:
                result_mask |= (1 << num)
        return result_mask
    
    # 2. Multiple greedy initializations
    best_sum = 0
    best_mask = 0
    
    # Descending order
    m1 = greedy_initial(range(N, 0, -1))
    s1 = sum(weights[i] for i in range(N+1) if m1 & (1 << i))
    if s1 > best_sum:
        best_sum = s1
        best_mask = m1
    
    # Try a few different heuristics
    # Sort by weight/degree ratio
    degrees = [bin(conflict[i] & ((1 << (N+1)) - 1)).count('1') - 1 for i in range(N+1)]
    ratio_order = sorted(range(N, 0, -1), key=lambda x: weights[x] / max(1, degrees[x]), reverse=True)
    m2 = greedy_initial(ratio_order)
    s2 = sum(weights[i] for i in range(N+1) if m2 & (1 << i))
    if s2 > best_sum:
        best_sum = s2
        best_mask = m2
    
    # 3. Local search
    current_mask = best_mask
    current_sum = best_sum
    
    improved = True
    pass_count = 0
    max_passes = 2000
    
    while improved and pass_count < max_passes and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Try 1-for-0 insertions
        for num in range(N, 0, -1):
            if current_mask & (1 << num):
                continue
            if (current_mask & conflict[num]) == 0:
                current_mask |= (1 << num)
                current_sum += num
                improved = True
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_mask = current_mask
        
        # Try 1-for-1 swaps
        if not improved:
            for num_out in range(1, N + 1):
                if not (current_mask & (1 << num_out)):
                    continue
                mask_without_out = current_mask & ~(1 << num_out)
                for num_in in range(N, num_out, -1):
                    if current_mask & (1 << num_in):
                        continue
                    if (mask_without_out & conflict[num_in]) == 0:
                        new_sum = current_sum - num_out + num_in
                        if new_sum > current_sum:
                            current_mask = mask_without_out | (1 << num_in)
                            current_sum = new_sum
                            improved = True
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_mask = current_mask
                        break
        
        # Try 2-for-1 swaps (remove 2, add 1)
        if not improved:
            for num_out1 in range(1, N + 1):
                if not (current_mask & (1 << num_out1)):
                    continue
                for num_out2 in range(num_out1 + 1, N + 1):
                    if not (current_mask & (1 << num_out2)):
                        continue
                    mask_without = current_mask & ~(1 << num_out1) & ~(1 << num_out2)
                    for num_in in range(N, max(num_out1, num_out2), -1):
                        if current_mask & (1 << num_in):
                            continue
                        if (mask_without & conflict[num_in]) == 0:
                            new_sum = current_sum - num_out1 - num_out2 + num_in
                            if new_sum > current_sum:
                                current_mask = mask_without | (1 << num_in)
                                current_sum = new_sum
                                improved = True
                                if current_sum > best_sum:
                                    best_sum = current_sum
                                    best_mask = current_mask
                            break
        
        # Try 1-for-2 swaps (remove 1, add 2)
        if not improved:
            for num_out in range(1, N + 1):
                if not (current_mask & (1 << num_out)):
                    continue
                mask_without_out = current_mask & ~(1 << num_out)
                added_any = False
                for num_in1 in range(N, num_out, -1):
                    if current_mask & (1 << num_in1):
                        continue
                    if (mask_without_out & conflict[num_in1]) != 0:
                        continue
                    mask_with_in1 = mask_without_out | (1 << num_in1)
                    for num_in2 in range(num_in1 - 1, num_out, -1):
                        if current_mask & (1 << num_in2):
                            continue
                        if (mask_with_in1 & conflict[num_in2]) != 0:
                            continue
                        new_sum = current_sum - num_out + num_in1 + num_in2
                        if new_sum > current_sum:
                            current_mask = mask_with_in1 | (1 << num_in2)
                            current_sum = new_sum
                            improved = True
                            added_any = True
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_mask = current_mask
                            break
                    if added_any:
                        break
                if added_any:
                    break
        
        # Try 2-for-2 swaps
        if not improved and time.time() < deadline - 10:
            for num_out1 in range(1, N + 1):
                if not (current_mask & (1 << num_out1)):
                    continue
                for num_out2 in range(num_out1 + 1, N + 1):
                    if not (current_mask & (1 << num_out2)):
                        continue
                    mask_without = current_mask & ~(1 << num_out1) & ~(1 << num_out2)
                    found = False
                    for num_in1 in range(N, max(num_out1, num_out2), -1):
                        if current_mask & (1 << num_in1):
                            continue
                        if (mask_without & conflict[num_in1]) != 0:
                            continue
                        mask_with_in1 = mask_without | (1 << num_in1)
                        for num_in2 in range(num_in1 - 1, max(num_out1, num_out2), -1):
                            if current_mask & (1 << num_in2):
                                continue
                            if (mask_with_in1 & conflict[num_in2]) != 0:
                                continue
                            new_sum = current_sum - num_out1 - num_out2 + num_in1 + num_in2
                            if new_sum > current_sum:
                                current_mask = mask_with_in1 | (1 << num_in2)
                                current_sum = new_sum
                                improved = True
                                found = True
                                if current_sum > best_sum:
                                    best_sum = current_sum
                                    best_mask = current_mask
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
    
    # 4. ILP polish step if time permits
    remaining = deadline - time.time()
    if remaining > 15:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import scipy.sparse as sp
            
            # Variables x[i] for i in 1..N
            # Maximize sum(i * x[i])
            # Subject to: for each conflicting pair (i,j), x[i] + x[j] <= 1
            # x[i] in {0, 1}
            
            n_vars = N  # indices 0..N-1 represent numbers 1..N
            
            # Objective: maximize sum(i * x[i]) for i=1..N
            # In milp form: minimize -sum(i * x[i])
            c = np.array([-i for i in range(1, N + 1)], dtype=float)
            
            # Constraints: for each edge in conflict graph, x[i-1] + x[j-1] <= 1
            # Build constraint matrix
            rows = []
            cols = []
            vals = []
            
            for i in range(1, N + 1):
                # Get conflicting nodes
                cm = conflict[i]
                m = cm
                while m:
                    lowbit = m & (-m)
                    j = lowbit.bit_length() - 1
                    if j > i:  # Only add each pair once
                        rows.append(len(rows))
                        cols.append(i - 1)
                        vals.append(1.0)
                        rows.append(len(rows))
                        cols.append(j - 1)
                        vals.append(1.0)
                    m ^= lowbit
            
            if rows:
                A = sp.csr_matrix((vals, (rows, cols)), shape=(len(rows) // 2, n_vars))
                ub = np.ones(len(rows) // 2)
                lb = np.full(len(rows) // 2, -np.inf)
                cons = LinearConstraint(A, lb, ub)
            else:
                cons = None
            
            integrality = np.ones(n_vars)
            bounds = Bounds(0, 1)
            
            result = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
            
            if result.status == 0 and result.x is not None:
                ilp_mask = 0
                ilp_sum = 0
                for i in range(N):
                    if result.x[i] > 0.5:
                        num = i + 1
                        ilp_mask |= (1 << num)
                        ilp_sum += num
                if ilp_sum > best_sum:
                    best_sum = ilp_sum
                    best_mask = ilp_mask
        except Exception:
            pass
    
    # Final local search polish
    current_mask = best_mask
    current_sum = best_sum
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for num in range(N, 0, -1):
            if current_mask & (1 << num):
                continue
            if (current_mask & conflict[num]) == 0:
                current_mask |= (1 << num)
                current_sum += num
                improved = True
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_mask = current_mask
        if not improved:
            break
    
    # Convert mask to list
    final_set = mask_to_list(best_mask)
    final_set = [int(x) for x in final_set]
    
    return final_set

# EVOLVE-BLOCK-END
