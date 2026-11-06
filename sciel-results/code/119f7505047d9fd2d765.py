"""Combined program: Exact MILP with Bitset-Optimized Conflict Graph and Robust Heuristic Fallback."""
# EVOLVE-BLOCK-START
import time
from math import lcm
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit (5s buffer)
    
    # 1. Define the set of candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # 2. Precompute incompatibility (conflict) pairs using Bitsets for speed
    # conflict[i] is an integer bitmask where bit j is set if candidates[i] and candidates[j] conflict
    # i.e., lcm(candidates[i], candidates[j]) <= 256
    conflict = [0] * n
    
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if lcm(a, b) <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # 3. Try to solve exactly using scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Weights: 1/x
        weights = np.array([1.0 / x for x in candidates])
        
        # Variables: x_i in {0, 1}
        integrality = np.ones(n)
        
        # Objective: Maximize sum(w_i * x_i) -> Minimize -sum(w_i * x_i)
        c = -weights
        
        # Build constraint matrix from bitsets
        # We need to convert bitsets to a list of (i, j) conflicts for the sparse matrix
        # Or we can iterate through the bitsets to build the COO matrix directly
        
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            mask = conflict[i]
            # Iterate through set bits in mask
            # Only consider j > i to avoid duplicates, or just all j?
            # The constraint is x_i + x_j <= 1.
            # If we include both (i,j) and (j,i), it's redundant but valid.
            # To save memory, we can just generate unique pairs.
            # Since we built conflict symmetrically, we can just iterate i and check bits j > i.
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    rows.append(len(rows)) # Wait, we need row index for each constraint
                    # Actually, let's just count conflicts first or build lists properly.
                    pass
        
        # Let's rebuild the lists properly
        rows = []
        cols = []
        data = []
        constraint_idx = 0
        for i in range(n):
            mask = conflict[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    # Constraint: x_i + x_j <= 1
                    rows.extend([constraint_idx, constraint_idx])
                    cols.extend([i, j])
                    data.extend([1, 1])
                    constraint_idx += 1
        
        if constraint_idx > 0:
            A = sp.coo_matrix((data, (rows, cols)), shape=(constraint_idx, n)).tocsr()
            b_ub = np.ones(constraint_idx)
            constraints = [
                LinearConstraint(A, ub=b_ub)
            ]
        else:
            constraints = []
            
        bounds = Bounds(lb=0, ub=1)
        
        # Check time before solving
        if time.time() > deadline - 10:
            raise TimeoutError("Time limit approaching before MILP")
            
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:  # Optimal solution found
            # Extract solution
            x_sol = res.x
            selected_indices = np.where(x_sol > 0.5)[0]
            selected_values = [candidates[i] for i in selected_indices]
            
            # Verify validity
            valid = True
            sel_set = set(selected_values)
            for i in range(len(selected_values)):
                a = selected_values[i]
                for j in range(i + 1, len(selected_values)):
                    b = selected_values[j]
                    if lcm(a, b) <= 256:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and selected_values:
                return selected_values
            else:
                # Fallback if verification fails
                pass
                
    except Exception as e:
        # Fallback to heuristic if scipy fails or is not available
        pass

    # 4. Fallback: Greedy + Advanced Local Search using Bitsets
    
    # Greedy: Sort by weight descending (smallest x first)
    selected_bits = 0
    selected_indices = []
    
    for i in range(n):
        if not (conflict[i] & selected_bits):
            selected_indices.append(i)
            selected_bits |= (1 << i)
    
    current_sum = sum(1.0 / candidates[i] for i in selected_indices)
    sel_set = set(selected_indices)
    weights = {i: 1.0 / candidates[i] for i in range(n)}
    
    # Precompute compatibility masks: compat[i] has bit j set if i and j are compatible (no conflict)
    # Note: conflict[i] has bit j set if they conflict.
    # So compat[i] = ~conflict[i] & ((1 << n) - 1)
    full_mask = (1 << n) - 1
    compat = [0] * n
    for i in range(n):
        compat[i] = full_mask ^ conflict[i]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Try to add any single node
        for j in range(n):
            if (selected_bits >> j) & 1:
                continue
            if not (conflict[j] & selected_bits):
                selected_bits |= (1 << j)
                sel_set.add(j)
                current_sum += weights[j]
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try 1-swap
        found_swap = False
        for i in list(sel_set):
            trial_bits = selected_bits & ~(1 << i)
            # Find best j compatible with trial_bits
            best_w = 0
            best_j = -1
            # Iterate candidates
            # Optimization: We can iterate over set bits of (compat_with_trial)
            # But trial_bits changes.
            # Just loop j
            for j in range(n):
                if (trial_bits >> j) & 1:
                    continue
                if not (conflict[j] & trial_bits):
                    if weights[j] > best_w:
                        best_w = weights[j]
                        best_j = j
            if best_j != -1 and best_w > weights[i] + 1e-12:
                selected_bits &= ~(1 << i)
                selected_bits |= (1 << best_j)
                sel_set.remove(i)
                sel_set.add(best_j)
                current_sum += best_w - weights[i]
                improved = True
                found_swap = True
                break
        if found_swap:
            continue
            
        # 3. Try 2-swap
        if time.time() < deadline - 10:
            best_swap_gain = 0
            best_i = -1
            best_k = -1
            best_j = -1
            best_l = -1
            
            sel_list = list(sel_set)
            len_sel = len(sel_list)
            for idx_i in range(len_sel):
                i = sel_list[idx_i]
                for idx_k in range(idx_i + 1, len_sel):
                    k = sel_list[idx_k]
                    trial_bits = selected_bits & ~(1 << i) & ~(1 << k)
                    removed_weight = weights[i] + weights[k]
                    
                    best_pair_w = 0
                    bj = -1
                    bl = -1
                    
                    # Find best pair (j, l)
                    for j in range(n):
                        if (trial_bits >> j) & 1:
                            continue
                        if (conflict[j] & trial_bits):
                            continue
                        
                        # j is compatible with trial_bits
                        # Find best l compatible with trial_bits and j
                        # l must not be j
                        # l must not be in trial_bits
                        # l must not conflict with trial_bits
                        # l must not conflict with j
                        
                        # Mask of valid l:
                        # ~trial_bits (not in selected)
                        # & ~conflict[j] (compatible with j)
                        # & ~conflict_with_trial (compatible with trial) -> This is hard to precompute for dynamic trial_bits
                        # Instead, just check l
                        for l in range(n):
                            if l == j:
                                continue
                            if (trial_bits >> l) & 1:
                                continue
                            if (conflict[l] & trial_bits):
                                continue
                            if (conflict[j] & (1 << l)):
                                continue
                            
                            pair_w = weights[j] + weights[l]
                            if pair_w > best_pair_w:
                                best_pair_w = pair_w
                                bj = j
                                bl = l
                                
                    gain = best_pair_w - removed_weight
                    if gain > best_swap_gain + 1e-12:
                        best_swap_gain = gain
                        best_i = i
                        best_k = k
                        best_j = bj
                        best_l = bl
                        
            if best_j != -1 and best_l != -1:
                selected_bits &= ~(1 << best_i)
                selected_bits &= ~(1 << best_k)
                selected_bits |= (1 << best_j)
                selected_bits |= (1 << best_l)
                sel_set.remove(best_i)
                sel_set.remove(best_k)
                sel_set.add(best_j)
                sel_set.add(best_l)
                current_sum += best_swap_gain
                improved = True
                continue
                
    return [candidates[i] for i in sel_set]

# EVOLVE-BLOCK-END
