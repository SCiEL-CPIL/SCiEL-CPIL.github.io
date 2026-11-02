import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Define the target tensor T_target
    # T_target[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) else 0
    # We will represent it implicitly in the cost function
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    schoolbook = get_schoolbook()
    best_rank = n * m * p
    best_solution = schoolbook

    start_time = time.time()
    time_limit = 110  # Leave 10s buffer for 120s limit
    
    # Try to find rank < best_rank
    # Given the rarity of sub-multiplicative decompositions for this size, 
    # we try a few random restarts for R=29, 28, 27
    
    for R in range(best_rank - 1, 15, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # Simple ALS iteration to converge
            # We minimize || T_target - sum(U_r V_r W_r) ||
            # T_target is 0/1.
            
            # Precompute target slices? 
            # T_target[u, v, :] is non-zero only if u, v compatible.
            # Let's just iterate.
            
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                for u in range(dim_u):
                    # T_target[u, v, w]
                    # We need to solve for U[:, u]
                    # Sum_r U[u, r] V[:, r] W[:, r]^T = slice of T
                    # This is a linear system for U[:, u]
                    # Let M = V @ W.T  (dim_v x dim_w)
                    # We want vec(T_slice_u) = (W^T \otimes V) vec(U_col_u) ? No.
                    # T[u, :, :] = Sum_r U[u,r] outer(V[:,r], W[:,r])
                    # Let's flatten T[u, :, :] to T_u_vec (dim_v * dim_w)
                    # T_u_vec = Sum_r U[u,r] (W[:,r] \otimes V[:,r])? 
                    # outer(V_r, W_r) flattened is W_r \otimes V_r ?
                    # outer(a, b)[i, j] = a[i] b[j].
                    # vec(outer(a,b)) = a \otimes b.
                    # So T_u_vec = Sum_r U[u,r] (V_r \otimes W_r)
                    # Matrix A_u has columns (V_r \otimes W_r)
                    # A_u shape (dim_v * dim_w, R)
                    # T_u_vec shape (dim_v * dim_w, 1)
                    # Solve A_u X = T_u_vec for X (which is U[:, u])
                    
                    # Construct A_u
                    # V_r \otimes W_r for r=0..R-1
                    # V is (dim_v, R), W is (dim_w, R)
                    # We need all pairwise combinations? No, just r-th column of V and W.
                    # Col r of A_u is Kronecker product of V[:, r] and W[:, r]
                    
                    # Efficient way:
                    # A_u = (W.T \otimes V.T) ? No.
                    # Let's build A_u.
                    # dim_v * dim_w is 15 * 10 = 150. R is ~29.
                    # A_u is 150 x 29.
                    A_u = np.zeros((dim_v * dim_w, R))
                    for r in range(R):
                        A_u[:, r] = np.kron(V[:, r], W[:, r])
                    
                    # Get target slice T_target[u, :, :]
                    # u corresponds to (i, j)
                    i = u // m
                    j = u % m
                    T_slice = np.zeros((dim_v, dim_w))
                    for v in range(dim_v):
                        # v corresponds to (j', k)
                        jp = v // p
                        kp = v % p
                        if jp != j:
                            continue
                        for w in range(dim_w):
                            # w corresponds to (k', i')
                            kn = w // n
                            in_ = w % n
                            if kn == kp and in_ == i:
                                T_slice[v, w] = 1.0
                                
                    T_vec = T_slice.flatten()
                    
                    # Solve least squares
                    try:
                        U_new, residuals, rank_a, s = np.linalg.lstsq(A_u, T_vec, rcond=None)
                        U[:, u] = U_new
                    except:
                        pass

                # Update V
                for v in range(dim_v):
                    # v corresponds to (j, k)
                    j = v // p
                    k = v % p
                    # T[:, v, :] = Sum_r V[v, r] outer(U[:, r], W[:, r])
                    # vec = U_r \otimes W_r
                    A_v = np.zeros((dim_u * dim_w, R))
                    for r in range(R):
                        A_v[:, r] = np.kron(U[:, r], W[:, r])
                    
                    T_slice = np.zeros((dim_u, dim_w))
                    for u in range(dim_u):
                        i = u // m
                        ju = u % m
                        if ju != j:
                            continue
                        for w in range(dim_w):
                            kw = w // n
                            iw = w % n
                            if kw == k and iw == i:
                                T_slice[u, w] = 1.0
                    T_vec = T_slice.flatten()
                    try:
                        V_new, residuals, rank_a, s = np.linalg.lstsq(A_v, T_vec, rcond=None)
                        V[:, v] = V_new
                    except:
                        pass

                # Update W
                for w in range(dim_w):
                    # w corresponds to (k, i)
                    k = w // n
                    i = w % n
                    # T[:, :, w] = Sum_r W[w, r] outer(U[:, r], V[:, r])
                    # vec = U_r \otimes V_r
                    A_w = np.zeros((dim_u * dim_v, R))
                    for r in range(R):
                        A_w[:, r] = np.kron(U[:, r], V[:, r])
                    
                    T_slice = np.zeros((dim_u, dim_v))
                    for u in range(dim_u):
                        iu = u // m
                        ju = u % m
                        if iu != i:
                            continue
                        for v in range(dim_v):
                            jv = v // p
                            kv = v % p
                            if jv == ju and kv == k:
                                T_slice[u, v] = 1.0
                    T_vec = T_slice.flatten()
                    try:
                        W_new, residuals, rank_a, s = np.linalg.lstsq(A_w, T_vec, rcond=None)
                        W[:, w] = W_new
                    except:
                        pass
            
            # Check convergence and validity
            # Calculate error
            error = 0.0
            for u in range(dim_u):
                i = u // m
                j = u % m
                # Reconstruct slice
                recon = np.zeros((dim_v, dim_w))
                for r in range(R):
                    recon += U[u, r] * np.outer(V[:, r], W[:, r])
                
                # Compare with target
                for v in range(dim_v):
                    jp = v // p
                    kp = v % p
                    if jp != j:
                        continue
                    for w in range(dim_w):
                        kn = w // n
                        in_ = w % n
                        if kn == kp and in_ == i:
                            target_val = 1.0
                        else:
                            target_val = 0.0
                        error += (recon[v, w] - target_val) ** 2
            
            if error < 1e-6:
                # Check if entries are close to integers or half-integers
                valid = True
                for mat in [U, V, W]:
                    for val in mat:
                        # Check if val is close to k/2
                        closest_half = round(val * 2) / 2
                        if abs(val - closest_half) > 1e-3:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Snap to half-integers
                    U_snapped = np.round(U * 2) / 2
                    V_snapped = np.round(V * 2) / 2
                    W_snapped = np.round(W * 2) / 2
                    best_rank = R
                    best_solution = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                    found = True
                    break
        
        if found:
            break

    return best_solution
# EVOLVE-BLOCK-END
