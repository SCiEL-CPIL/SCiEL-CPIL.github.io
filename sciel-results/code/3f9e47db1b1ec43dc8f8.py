import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Heuristic Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    fac_order = np.argsort(-flow_sums, kind="stable")
    loc_order = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Calculate initial cost
    B_perm = b[np.ix_(p, p)]
    cost = np.sum(a * B_perm)
    
    best_p = p.copy()
    best_cost = cost
    
    start_time = time.time()
    time_limit = 115
    
    # Tabu Search Parameters
    tabu_tenure = 100
    tabu_list = {}
    current_iter = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        best_delta = np.inf
        best_i = -1
        best_k = -1
        
        # Evaluate all 2-swaps using vectorized calculation per i
        for i in range(n):
            u = p[i]
            # Indices k > i
            K = np.arange(i + 1, n)
            if len(K) == 0:
                continue
            V = p[K]
            
            # M[k', j] = B[V[k'], p[j]]
            M = b[np.ix_(V, p)]
            W = b[u, p]
            A_i = a[i]
            A_K = a[K]
            
            # Term 1: sum_j A_ij * B_vk_pj
            T1 = M @ A_i
            # Term 2: sum_j A_ij * B_u_pj (constant for fixed i)
            T2 = np.dot(A_i, W)
            # Term 3: sum_j A_kj * B_vk_pj
            T3 = np.sum(A_K * M, axis=1)
            # Term 4: sum_j A_kj * B_u_pj
            T4 = A_K @ W
            
            S = T1 - T2 - T3 + T4
            
            # Corrections
            # Corr_i: (A_ii - A_ki) * (B_vk_u - B_uu)
            Corr_i = (a[i, i] - A_K[:, i]) * (b[V, u] - b[u, u])
            # Corr_k: (A_ik - A_kk) * (B_vk_vk - B_u_vk)
            Corr_k = (A_i[K] - np.diag(A_K)) * (b[V, V] - b[u, V])
            
            # DiffPair
            DP1 = A_i[K] * (b[V, u] - b[u, V])
            DP2 = A_K[:, i] * (b[u, V] - b[V, u])
            DP3 = a[i, i] * (b[V, V] - b[u, u])
            DP4 = np.diag(A_K) * (b[u, u] - b[V, V])
            
            DP = DP1 + DP2 + DP3 + DP4
            
            Deltas = S - Corr_i - Corr_k + DP
            
            # Check tabu status
            # We need to check which k's are tabu
            # tabu_list keys are (i, k) with i < k
            # Here i is fixed, k varies in K
            # We can create a mask
            tabu_mask = np.zeros(len(K), dtype=bool)
            # Efficient check: 
            # Instead of looping K, we can check if (i, k) is in tabu_list
            # Since K is contiguous, we can just loop or use a dictionary lookup
            # For speed, we might just ignore tabu in the vectorized part and filter later?
            # No, we need to find the best non-tabu.
            
            # Let's find the min delta overall
            min_idx = np.argmin(Deltas)
            min_k = K[min_idx]
            min_delta = Deltas[min_idx]
            
            # Check if this best move is tabu
            key = (i, min_k)
            is_tabu = key in tabu_list and tabu_list[key] > current_iter
            
            if not is_tabu:
                if min_delta < best_delta:
                    best_delta = min_delta
                    best_i = i
                    best_k = min_k
            else:
                # If the best is tabu, check if it satisfies aspiration
                if cost + min_delta < best_cost:
                    if min_delta < best_delta:
                        best_delta = min_delta
                        best_i = i
                        best_k = min_k
                else:
                    # Find next best non-tabu
                    # This is expensive to do with vectorization.
                    # For now, if the best is tabu and not aspirating, 
                    # we might need to loop or just accept a suboptimal move?
                    # Better: Mask out tabu moves and find min again.
                    # But we don't have a vectorized tabu mask easily.
                    # Given N=150, we can just loop K to find best non-tabu if needed.
                    # But this case should be rare if tenure is small.
                    # Let's just do a simple loop for this specific i if the best is tabu.
                    for idx, k in enumerate(K):
                        key_k = (i, k)
                        if key_k in tabu_list and tabu_list[key_k] > current_iter:
                            continue
                        if Deltas[idx] < best_delta:
                            best_delta = Deltas[idx]
                            best_i = i
                            best_k = k

        if best_i == -1:
            # No valid move found
            break
            
        # Apply move
        p[best_i], p[best_k] = p[best_k], p[best_i]
        cost += best_delta
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            break
            
        # Update Tabu List
        tabu_list[(best_i, best_k)] = current_iter + tabu_tenure
        current_iter += 1

    return best_p.tolist()
