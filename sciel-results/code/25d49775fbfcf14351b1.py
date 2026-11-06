import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # 1. Precompute APs and Incidence Matrix
    # We include all APs of length >= 2.
    # Note: Length 1 APs have sum +/- 1, so max is at least 1.
    
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)
                
    num_aps = len(ap_terms)
    
    # Build M (num_aps x N)
    M = np.zeros((num_aps, N), dtype=np.float32)
    pos_to_aps = [[] for _ in range(N)]
    
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)
            
    # Convert pos_to_aps to numpy arrays for indexing
    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]
    
    # Compute C = M.T @ M
    C = M.T @ M
    K = np.diag(C).astype(np.float32)
    
    # Helper to get hard max
    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))
    
    best_signs = None
    best_hard_max = N # Worst case
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Initialize with random signs
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.float32)
        
        # Initial sums
        sums = M @ signs
        
        # Initial G = C @ signs
        G = C @ signs
        
        current_hard_max = get_hard_max(sums)
        if current_hard_max < best_hard_max:
            best_hard_max = current_hard_max
            best_signs = signs.copy()
            
        # SA Parameters
        T = 10.0
        T_min = 0.1
        
        steps = 0
        best_local_hard_max = current_hard_max
        best_local_signs = signs.copy()
        
        while time.time() < deadline - 2:
            # Choose move type: 70% single flip, 30% double flip
            use_double = rng.random() < 0.3
            
            if use_double:
                k1 = rng.randint(0, N - 1)
                k2 = rng.randint(0, N - 1)
                if k1 == k2:
                    use_double = False
                else:
                    delta1 = -2.0 * signs[k1]
                    delta2 = -2.0 * signs[k2]
                    
                    # Compute cross term: sum over APs containing both k1 and k2
                    # This is C[k1, k2]
                    cross = C[k1, k2]
                    
                    # delta_J = 2*(delta1*G[k1] + delta2*G[k2]) + delta1^2*K[k1] + delta2^2*K[k2] + 2*delta1*delta2*cross
                    delta_J = 2.0 * (delta1 * G[k1] + delta2 * G[k2]) + (delta1 * delta1) * K[k1] + (delta2 * delta2) * K[k2] + 2.0 * delta1 * delta2 * cross
                    
                    accept = False
                    if delta_J < 0:
                        accept = True
                    else:
                        if delta_J / T > 100:
                            accept = False
                        else:
                            accept = rng.random() < math.exp(-delta_J / T)
                            
                    if accept:
                        # Apply moves
                        signs[k1] = -signs[k1]
                        signs[k2] = -signs[k2]
                        
                        # Update sums
                        idxs1 = pos_to_aps_arr[k1]
                        idxs2 = pos_to_aps_arr[k2]
                        sums[idxs1] += delta1
                        sums[idxs2] += delta2
                        
                        # Update G
                        G += delta1 * C[:, k1] + delta2 * C[:, k2]
                        
                        steps += 1
                        
                        # Check hard max periodically
                        if steps % 50 == 0:
                            current_hard_max = get_hard_max(sums)
                            if current_hard_max < best_local_hard_max:
                                best_local_hard_max = current_hard_max
                                best_local_signs = signs.copy()
                            if current_hard_max < best_hard_max:
                                best_hard_max = current_hard_max
                                best_signs = signs.copy()
            else:
                # Single flip
                k = rng.randint(0, N - 1)
                delta = -2.0 * signs[k]
                
                delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
                
                accept = False
                if delta_J < 0:
                    accept = True
                else:
                    if delta_J / T > 100:
                        accept = False
                    else:
                        accept = rng.random() < math.exp(-delta_J / T)
                        
                if accept:
                    signs[k] = -signs[k]
                    
                    idxs = pos_to_aps_arr[k]
                    sums[idxs] += delta
                    
                    G += delta * C[:, k]
                    
                    steps += 1
                    
                    if steps % 50 == 0:
                        current_hard_max = get_hard_max(sums)
                        if current_hard_max < best_local_hard_max:
                            best_local_hard_max = current_hard_max
                            best_local_signs = signs.copy()
                        if current_hard_max < best_hard_max:
                            best_hard_max = current_hard_max
                            best_signs = signs.copy()
                            
            # Annealing
            if steps % 500 == 0:
                T *= 0.97
                if T < T_min:
                    T = T_min
                    
            # Early stop if we hit 1
            if best_hard_max <= 1:
                break
        
        # Polish phase: exact local search to reduce hard max
        if time.time() < deadline - 1:
            polish_signs = best_local_signs.copy()
            polish_sums = M @ polish_signs
            polish_hard_max = get_hard_max(polish_sums)
            
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for k in range(N):
                    delta = -2.0 * polish_signs[k]
                    idxs = pos_to_aps_arr[k]
                    # Try flip
                    old_sums = polish_sums.copy()
                    polish_sums[idxs] += delta
                    new_hard_max = get_hard_max(polish_sums)
                    if new_hard_max < polish_hard_max:
                        polish_signs[k] = -polish_signs[k]
                        polish_hard_max = new_hard_max
                        improved = True
                        if polish_hard_max < best_hard_max:
                            best_hard_max = polish_hard_max
                            best_signs = polish_signs.copy()
                            if best_hard_max <= 1:
                                break
                    else:
                        # Revert
                        polish_sums = old_sums
                if best_hard_max <= 1:
                    break
    
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
