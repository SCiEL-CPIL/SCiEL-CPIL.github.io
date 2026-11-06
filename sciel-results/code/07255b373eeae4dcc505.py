import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # 1. Precompute APs and incidence structure
    ap_indices = []
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_indices.append(terms)
                
    num_aps = len(ap_indices)
    
    # Build reverse index
    aps_containing = [[] for _ in range(N)]
    for j, terms in enumerate(ap_indices):
        for i in terms:
            aps_containing[i].append(j)
    
    # Precompute AP lengths for weighting
    ap_lengths = np.array([len(terms) for terms in ap_indices], dtype=np.int32)

    def compute_initial_sums(signs):
        S = np.zeros(num_aps, dtype=np.float64)
        for j, terms in enumerate(ap_indices):
            S[j] = np.sum(signs[terms])
        return S

    def get_hard_max(S):
        return int(np.max(np.abs(S)))

    def calc_soft_obj_fast(S, beta):
        """Fast soft objective using log-sum-exp with shift"""
        abs_S = np.abs(S)
        max_val = np.max(abs_S)
        if max_val == 0:
            return 0.0
        shifted = abs_S / beta - max_val / beta
        # Use log-sum-exp trick
        sum_exp = np.sum(np.exp(shifted))
        return beta * (max_val / beta + np.log(sum_exp))

    def polish(signs, S, rng, time_limit):
        """Local search polish: try flipping indices that reduce hard max"""
        hard_max = get_hard_max(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Find APs with max absolute sum
            abs_S = np.abs(S)
            max_val = np.max(abs_S)
            if max_val <= 1:
                break
            # Get indices of APs achieving max
            max_aps = np.where(abs_S == max_val)[0]
            # For each such AP, try flipping one of its terms
            for ap_idx in max_aps[:20]:  # Limit to avoid too much work
                terms = ap_indices[ap_idx]
                for i in terms:
                    old_sign = signs[i]
                    delta_s = -2.0 * old_sign
                    affected = aps_containing[i]
                    S[affected] += delta_s
                    new_max = get_hard_max(S)
                    if new_max < hard_max:
                        signs[i] = -old_sign
                        hard_max = new_max
                        improved = True
                    else:
                        S[affected] -= delta_s
                    if improved:
                        break
                if improved:
                    break
        return signs, S, hard_max

    best_signs = None
    best_hard_max = N

    # Generate diverse initial solutions
    def thue_morse():
        return np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(N)], dtype=np.int8)

    def random_init(rng):
        return np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)

    # Create initial candidates
    candidates = []
    candidates.append(thue_morse())
    for s in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
        rng = random.Random(s)
        candidates.append(random_init(rng))

    # Evaluate all candidates and keep best
    for signs in candidates:
        S = compute_initial_sums(signs)
        hmax = get_hard_max(S)
        if hmax < best_hard_max:
            best_hard_max = hmax
            best_signs = signs.copy()
        if time.time() > deadline - 10:
            break

    # SA with improved parameters
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Start from current best or random
        if seed == 42:
            signs = best_signs.copy()
        elif seed < 4:
            signs = thue_morse()
        else:
            signs = random_init(rng)
            
        S = compute_initial_sums(signs)
        current_hard_max = get_hard_max(S)
        
        if current_hard_max < best_hard_max:
            best_hard_max = current_hard_max
            best_signs = signs.copy()
            
        # SA Parameters - tuned for faster convergence
        beta = 0.5
        T = 3.0
        T_min = 0.05
        beta_min = 0.1
        
        current_obj = calc_soft_obj_fast(S, beta)
        
        # Track progress for adaptive cooling
        no_improve_count = 0
        last_improve_hard_max = current_hard_max
        
        steps = 0
        while time.time() < deadline - 3:
            steps += 1
            
            # Pick random index to flip
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            delta_s = -2.0 * old_sign
            
            affected_aps = aps_containing[i]
            S[affected_aps] += delta_s
            
            new_obj = calc_soft_obj_fast(S, beta)
            delta_obj = new_obj - current_obj
            
            accept = False
            if delta_obj < 0:
                accept = True
            else:
                ratio = delta_obj / T
                if ratio < 50:
                    accept = rng.random() < math.exp(-ratio)
                    
            if accept:
                signs[i] = -old_sign
                current_obj = new_obj
                
                # Periodic hard max check
                if steps % 50 == 0:
                    current_hard_max = get_hard_max(S)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                        no_improve_count = 0
                    elif current_hard_max == last_improve_hard_max:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                        last_improve_hard_max = current_hard_max
                    
                    # Adaptive cooling: accelerate if stuck
                    if no_improve_count > 20:
                        T *= 0.8
                        beta *= 0.95
                        no_improve_count = 0
            else:
                S[affected_aps] -= delta_s
                
            # Regular annealing
            if steps % 500 == 0:
                T *= 0.97
                beta *= 0.99
                if T < T_min:
                    T = T_min
                if beta < beta_min:
                    beta = beta_min
                    
            if best_hard_max <= 1:
                break

    # Final polish phase
    if best_signs is not None and time.time() < deadline - 2:
        best_signs, S, _ = polish(best_signs.copy(), compute_initial_sums(best_signs), random.Random(999), deadline - 1)
        final_hmax = get_hard_max(S)
        if final_hmax < best_hard_max:
            best_hard_max = final_hmax

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
