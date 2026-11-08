import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n, m, p) with all entries 1
    # We represent T as a flat array of length n*m*p
    # Index mapping: (i, j, k) -> i*m*p + j*p + k
    
    total_size = n * m * p
    target = np.ones(total_size, dtype=np.uint8)
    
    U_list = []
    V_list = []
    W_list = []
    
    current_target = target.copy()
    
    # Helper to get a random sparse binary vector
    def random_sparse_vector(size, sparsity=0.1):
        vec = np.zeros(size, dtype=np.uint8)
        num_ones = max(1, int(size * sparsity))
        indices = np.random.choice(size, num_ones, replace=False)
        vec[indices] = 1
        return vec

    # Pre-compute a set of candidate rank-1 tensors to avoid full random search every step
    # We will use a mix of structured and random candidates
    
    candidates = []
    # Structured candidates based on schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m, dtype=np.uint8)
                v = np.zeros(m*p, dtype=np.uint8)
                w = np.zeros(p*n, dtype=np.uint8)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                candidates.append((u, v, w))
                
    # Add some random structured candidates
    for _ in range(100):
        u = random_sparse_vector(n*m, 0.2)
        v = random_sparse_vector(m*p, 0.2)
        w = random_sparse_vector(p*n, 0.2)
        candidates.append((u, v, w))

    max_rank = 80
    start_time = time.time()
    
    for rank in range(max_rank):
        if time.time() - start_time > 110:
            break
            
        best_score = np.inf
        best_u = None
        best_v = None
        best_w = None
        
        # Evaluate a subset of candidates to save time
        # Shuffle candidates
        np.random.shuffle(candidates)
        num_eval = min(200, len(candidates))
        
        for idx in range(num_eval):
            u, v, w = candidates[idx]
            
            # Compute the rank-1 tensor S = u \otimes v \otimes w
            # S[i, j, k] = u[i, j] * v[j, k] * w[k, i]
            # Flatten S
            # u is (n*m), v is (m*p), w is (p*n)
            # We need to broadcast and multiply
            # Reshape u to (n, m, 1), v to (1, m, p), w to (p, 1, n)
            # Then multiply and sum? No, we just need the product element-wise.
            # u_reshaped = u.reshape(n, m, 1)
            # v_reshaped = v.reshape(1, m, p)
            # w_reshaped = w.reshape(p, 1, n)
            # S_3d = u_reshaped * v_reshaped * w_reshaped
            # S_flat = S_3d.flatten()
            
            # Optimization: Only compute if potential is good
            # Score = weight(current_target XOR S_flat)
            
            u_reshaped = u.reshape(n, m, 1)
            v_reshaped = v.reshape(1, m, p)
            w_reshaped = w.reshape(p, 1, n)
            
            S_3d = (u_reshaped * v_reshaped * w_reshaped).astype(np.uint8)
            S_flat = S_3d.flatten()
            
            new_target = current_target ^ S_flat
            score = np.sum(new_target)
            
            if score < best_score:
                best_score = score
                best_u = u
                best_v = v
                best_w = w
                best_new_target = new_target
                
        if best_u is None:
            break
            
        if best_score == 0:
            # Found exact decomposition
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            break
        else:
            # Add the best candidate
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            current_target = best_new_target
            
        # Regenerate some candidates based on current target?
        # For simplicity, we stick with the initial set + randoms
        if rank % 10 == 0:
            for _ in range(50):
                u = random_sparse_vector(n*m, 0.2)
                v = random_sparse_vector(m*p, 0.2)
                w = random_sparse_vector(p*n, 0.2)
                candidates.append((u, v, w))

    # If we didn't reach 0, we have a partial solution or a bad one.
    # The task requires an EXACT decomposition.
    # If we failed to find one, we must return a valid one.
    # Since we can't reuse the current design, we must generate a valid one.
    # The schoolbook decomposition is always valid and has rank n*m*p.
    # Rank 3*4*6 = 72.
    # If our greedy search failed, we fallback to schoolbook.
    
    if np.sum(current_target) != 0:
        # Fallback to schoolbook
        U_list = []
        V_list = []
        W_list = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n*m, dtype=np.uint8)
                    v = np.zeros(m*p, dtype=np.uint8)
                    w = np.zeros(p*n, dtype=np.uint8)
                    u[i*m + j] = 1
                    v[j*p + k] = 1
                    w[k*n + i] = 1
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
                    
    # Convert lists of arrays to the required format
    # U: (n*m, rank)
    rank = len(U_list)
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    for r in range(rank):
        U[:, r] = U_list[r]
        V[:, r] = V_list[r]
        W[:, r] = W_list[r]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
