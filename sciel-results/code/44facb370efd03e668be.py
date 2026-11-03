import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs T."""
        recon = np.zeros((n, m, p))
        for r in range(rank):
            # Vectorized update
            # U[:, r] is (n*m,), V[:, r] is (m*p,), W[:, r] is (p*n,)
            # T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n) # Note: W is indexed by k*n+i, so reshape to (p, n)
            
            # outer product of u and v gives (n, m, p) ? No.
            # u[i,j] * v[j,k] -> sum over j? No, this is a single term.
            # term[i, j, k] = u[i,j] * v[j,k] * w[k,i]
            
            # We can compute this as:
            # temp[i, k] = sum_j u[i,j] * v[j,k] * w[k,i]
            # This is not a simple matmul because w depends on i and k.
            
            # Efficient way:
            # For each k, w_k = w[k, :] (vector of length n)
            # For each i, u_i = u[i, :] (vector of length m)
            # For each j, v_j = v[:, j] (vector of length m) -- wait, v is (m, p)
            
            # Let's just loop over r and use broadcasting if possible, or simple loops for small n,m,p.
            # n=4, m=5, p=6 is small.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                        
        return np.allclose(recon, T, atol=1e-8)

    def snap_to_half(U, V, W):
        """Snap values to nearest 0.5."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap

    def als_decompose(rank, seed, max_iters=1000):
        """
        Alternating Least Squares for tensor decomposition.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Flatten T for easier handling? 
        # T is (n, m, p).
        # We want to minimize || T - sum_r U_r v_r w_r^T ||_F
        # This is equivalent to finding rank-r approximation.
        
        # ALS Iterations
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Solve for U
            # For each column r of U:
            # U[:, r] is a vector of length n*m.
            # The contribution to T is U[:, r] outer V[:, r] outer W[:, r]
            # This is tricky to vectorize across r because of the outer products.
            # However, for a fixed r, the problem is linear in U[:, r].
            # Let's iterate over r? No, that's not ALS.
            
            # Standard ALS for tensor:
            # Fix V, W. Solve for U.
            # The equation is: T = sum_r u_r v_r w_r^T
            # This can be written as: T = sum_r (u_r v_r) w_r^T
            # Let A_r = u_r v_r (vector of length n*m*p? No, u_r is n*m, v_r is m*p, their outer product is n*m*m*p? No.)
            
            # Let's reshape T to (n, m*p) or similar?
            # T[i, j, k].
            # If we fix V and W, the term for a specific (i, j, k) is:
            # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
            # Let B_r = V[:, r] * W[:, r] (element-wise product, length m*p * p*n? No, lengths don't match for direct element-wise if we flatten).
            
            # Let's define the "contracted" tensor for U.
            # For each (i, j), the vector of values over k is T[i, j, :].
            # The contribution from rank r is U[i*m+j, r] * (sum_k V[j*p+k, r] W[k*n+i, r] T[i, j, k])? No.
            
            # Let's use the matricization approach.
            # T is a tensor.
            # U is (n*m, r).
            # Let's solve for U by considering each row of U (index a = i*m+j).
            # The equation for T[a, b, c] where b=j*p+k, c=k*n+i is:
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            
            # This is a linear system for U[:, r] if we fix V and W?
            # Actually, if we fix V and W, the problem for U is:
            # Minimize || T - U * (V \circ W)^T || ? No.
            
            # Let's just do a simple gradient step or use scipy.optimize for the whole set?
            # Given the small size, we can use `scipy.optimize.least_squares` on the flattened parameters.
            
            # Parameters: U.flatten(), V.flatten(), W.flatten()
            # Total params: r*(n*m + m*p + p*n)
            # For r=50, 50*(20+30+24) = 3700 params.
            # Equations: 120.
            # This is very underdetermined.
            
            # Let's stick to a simpler heuristic: 
            # 1. Random init.
            # 2. Iterate:
            #    a. For each r, optimize U[:, r] given V, W.
            #    b. For each r, optimize V[:, r] given U, W.
            #    c. For each r, optimize W[:, r] given U, V.
            
            # How to optimize U[:, r]?
            # We want to minimize || T - sum_{l} U_l V_l W_l ||
            # If we fix all other terms, the term involving U_r is:
            # T - sum_{l != r} U_l V_l W_l - U_r V_r W_r
            # Let R = T - sum_{l != r} U_l V_l W_l
            # We want to find U_r to minimize || R - U_r V_r W_r ||
            # This is a linear least squares problem for U_r.
            # R is a tensor (n, m, p).
            # V_r is a vector (m*p). W_r is a vector (p*n).
            # The term U_r V_r W_r can be seen as:
            # For each (i, j, k): U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
            
            # Let's construct the matrix A_r such that A_r @ U_r.flatten() approximates R.flatten().
            # R.flatten() has size n*m*p.
            # U_r.flatten() has size n*m.
            # The mapping is:
            # Index in R.flatten(): idx = i*m*p + j*p + k (C-order)
            # Index in U_r.flatten(): u_idx = i*m + j
            # Coefficient for U_r[u_idx] in R[idx] is V_r[j*p+k] * W_r[k*n+i]
            
            # So, A_r is a matrix of size (n*m*p) x (n*m).
            # A_r[idx, u_idx] = V_r[j*p+k] * W_r[k*n+i] if u_idx = i*m+j and idx = i*m*p + j*p + k, else 0.
            
            # We can construct A_r efficiently.
            
            pass # Placeholder for implementation logic

    # Given the complexity and time limit, a simpler robust fallback is better.
    # The current program's greedy peeling is slow but might work for small ranks if lucky.
    # However, for 4x5x6, the rank is likely around 50-60.
    # Let's try a **randomized search for integer solutions** with small rank.
    
    # Actually, the most reliable "improvement" over the seed (which is just schoolbook) is to find *any* rank < 120.
    # A simple block decomposition can achieve this.
    
    # Let's implement a specific Strassen-like block decomposition for 4x5x6.
    # Partition:
    # A: 4x5 -> 2x2 blocks of sizes (2x2, 2x3)
    # B: 5x6 -> 2x2 blocks of sizes (2x3, 3x3) ... mismatched.
    
    # Let's use a simpler partition:
    # A: 4x5. B: 5x6.
    # Split m=5 into 2 and 3.
    # A = [A1 (4x2), A2 (4x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # Rank = Rank(4x2x6) + Rank(4x3x6)
    # Rank(4x2x6) <= 4*2*6 = 48. Can we do better?
    # Rank(4x3x6) <= 4*3*6 = 72.
    # Total <= 120. No gain.
    
    # We need overlapping blocks or Strassen tricks.
    
    # Given the difficulty of constructing a provably minimal rank decomposition on the fly within 150s,
    # and the fact that the seed program (score 1.0) just returns schoolbook,
    # the best strategy is to use the **greedy peeling** but **optimized** with numpy vectorization to be fast enough to try many seeds, and potentially find a rank < 120.
    
    # The current program's `peel_greedy` is O(rank * n*m*p * trials) which is slow.
    # Let's vectorize the scoring in `peel_greedy`.
    
    def peel_greedy_vectorized(seed, max_iters=30):
        rng = np.random.RandomState(seed)
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            if np.allclose(residual, 0, atol=1e-10):
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            n_trials = 100
            for t in range(n_trials):
                u = rng.randn(n * m)
                v = rng.randn(m * p)
                
                u_mat = u.reshape(n, m)
                v_mat = v.reshape(m, p)
                
                # Compute w_optimal for each (i, k)
                # num[i, k] = sum_j residual[i, j, k] * u_mat[i, j] * v_mat[j, k]
                # den[i, k] = sum_j (u_mat[i, j] * v_mat[j, k])^2
                
                # Vectorize num and den calculation
                # residual is (n, m, p)
                # u_mat is (n, m) -> (n, m, 1)
                # v_mat is (m, p) -> (1, m, p)
                
                u_exp = u_mat[:, :, np.newaxis] # (n, m, 1)
                v_exp = v_mat[np.newaxis, :, :] # (1, m, p)
                
                prod = u_exp * v_exp # (n, m, p)
                
                num = np.sum(residual * prod, axis=1) # (n, p)
                den = np.sum(prod * prod, axis=1) # (n, p)
                
                mask = den > 1e-12
                w_flat = np.zeros(p * n)
                # w_flat[k*n+i] = num[i,k] / den[i,k]
                # num is (n, p), so num[i, k].
                # We need to fill w_flat which is (p*n).
                # Index in w_flat: k*n + i.
                # Index in num: i, k.
                
                w_opt = np.zeros((n, p))
                np.divide(num, den, out=w_opt, where=mask)
                
                w_flat = w_opt.T.flatten() # Transpose to (p, n) then flatten to (p*n)
                
                # Compute recon
                # recon[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_opt[i, k]
                recon = u_exp * v_exp * w_opt[:, np.newaxis, :] # (n, m, p)
                
                diff = residual - recon
                score = -np.sum(diff * diff)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w_flat.copy()
            
            if best_u is None:
                break
            
            # Verify improvement
            u_mat = best_u.reshape(n, m)
            v_mat = best_v.reshape(m, p)
            w_opt = best_w.reshape(p, n).T # (n, p)
            
            recon = u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_opt[:, np.newaxis, :]
            new_residual = residual - recon
            
            if np.sum(new_residual * new_residual) >= np.sum(residual * residual) - 1e-12:
                break
            
            U_cols.append(best_u)
            V_cols.append(best_v)
            W_cols.append(best_w)
            residual = new_residual
        
        if len(U_cols) == 0:
            return None
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        rank = len(U_cols)
        
        # Verify
        if verify(U, V, W, rank):
            return U, V, W, rank
        else:
            return None

    # Try peeling
    for seed in range(100):
        if time.time() > deadline - 15.0:
            break
        result = peel_greedy_vectorized(seed, max_iters=20)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U, V, W)
                print(f"Found rank {rank} with seed {seed}")

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
