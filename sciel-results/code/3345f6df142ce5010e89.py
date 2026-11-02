import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # Indices:
    # A: i in [0, n), j in [0, m). a_idx = i * m + j
    # B: j in [0, m), k in [0, p). b_idx = j * p + k
    # C: i in [0, n), k in [0, p). c_idx = k * n + i (based on W[k*n+i])
    
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    
    # Construct the target tensor T
    # T is sparse: non-zero only at specific (a, b, c)
    # We will store non-zero indices and values
    # T[a, b, c] = 1 if j_a == j_b and i_a == i_c and k_b == k_c
    
    # For ALS, we need to access T efficiently.
    # We can represent T as a dictionary of (a, b, c) -> 1
    # Or, for each a, a matrix T_a of size dim_b x dim_c
    T_mats = [np.zeros((dim_b, dim_c)) for _ in range(dim_a)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_mats[a_idx][b_idx, c_idx] = 1.0

    # Schoolbook fallback
    def schoolbook():
        rank = dim_a * dim_b * dim_c // (m) # No, rank is n*m*p
        rank = n * m * p
        U = np.zeros((dim_a, rank))
        V = np.zeros((dim_b, rank))
        W = np.zeros((dim_c, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Try ALS for lower ranks
    # We need to solve for U, V, W
    # Equation: sum_k U[a, k] V[b, k] W[c, k] = T[a, b, c]
    
    def try_rank_als(r, max_iters=1000, restarts=5):
        # Precompute structures for speed
        # For each a, T_a is (dim_b, dim_c)
        
        for restart in range(restarts):
            if time.time() > deadline:
                return None
            
            # Initialize randomly
            U = np.random.randn(dim_a, r) * 0.1
            V = np.random.randn(dim_b, r) * 0.1
            W = np.random.randn(dim_c, r) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # For each a, solve for U[a, :]
                # T_a = sum_k U[a, k] * outer(V[:, k], W[:, k])? No.
                # T_a[b, c] = sum_k U[a, k] * V[b, k] * W[c, k]
                # Let M_a be a matrix of size (dim_b * dim_c, r)
                # M_a[b * dim_c + c, k] = V[b, k] * W[c, k]
                # Then T_a_flat = M_a @ U[a, :]
                
                # Build M_a
                # This is expensive if done every iter.
                # But dim_b * dim_c * r is small.
                
                for a in range(dim_a):
                    # Build M_a
                    # V is (dim_b, r), W is (dim_c, r)
                    # outer(V, W) is (dim_b, dim_c, r)
                    # We need to flatten the first two dims for the LSQ
                    # M_a[b, c, k] = V[b, k] * W[c, k]
                    
                    # Vectorized:
                    # V[:, :, None] * W[None, :, :] -> (dim_b, dim_c, r)
                    # Reshape to (dim_b * dim_c, r)
                    M_a = (V[:, :, None] * W[None, :, :]).reshape(dim_b * dim_c, r)
                    
                    # T_a_flat
                    T_flat = T_mats[a].flatten()
                    
                    # Solve M_a @ u = T_flat
                    # u is (r,)
                    u, residuals, rank_m, s = np.linalg.lstsq(M_a, T_flat, rcond=None)
                    U[a, :] = u
                    
                # Update V
                # T_a[b, c] = sum_k U[a, k] * V[b, k] * W[c, k]
                # For fixed U, W, solve for V[b, :]
                # Sum_a U[a, k] W[c, k] ... wait.
                # For a specific b, we have equations for all a, c.
                # T[a, b, c] = sum_k V[b, k] * (U[a, k] * W[c, k])
                # Let N_b be matrix of size (dim_a * dim_c, r)
                # N_b[a * dim_c + c, k] = U[a, k] * W[c, k]
                # T_b_flat = N_b @ V[b, :]
                
                for b in range(dim_b):
                    # N_b[a, c, k] = U[a, k] * W[c, k]
                    # U is (dim_a, r), W is (dim_c, r)
                    # U[:, :, None] * W[None, :, :] -> (dim_a, dim_c, r)
                    # Reshape to (dim_a * dim_c, r)
                    N_b = (U[:, :, None] * W[None, :, :]).reshape(dim_a * dim_c, r)
                    
                    # T_b_flat: collect T[a, b, c] for all a, c
                    # T_mats[a] is (dim_b, dim_c). We want column b.
                    # T_b_flat[a * dim_c + c] = T_mats[a][b, c]
                    T_b_flat = np.zeros(dim_a * dim_c)
                    for a in range(dim_a):
                        T_b_flat[a * dim_c : (a + 1) * dim_c] = T_mats[a][b, :]
                        
                    v, residuals, rank_n, s = np.linalg.lstsq(N_b, T_b_flat, rcond=None)
                    V[b, :] = v
                    
                # Update W
                # T[a, b, c] = sum_k W[c, k] * (U[a, k] * V[b, k])
                # For specific c, equations for all a, b.
                # T[a, b, c] = sum_k W[c, k] * P_c[a * dim_b + b, k]
                # where P_c[a * dim_b + b, k] = U[a, k] * V[b, k]
                
                for c in range(dim_c):
                    # P_c[a, b, k] = U[a, k] * V[b, k]
                    # U (dim_a, r), V (dim_b, r)
                    # U[:, None, :] * V[None, :, :] -> (dim_a, dim_b, r)
                    # Reshape to (dim_a * dim_b, r)
                    P_c = (U[:, None, :] * V[None, :, :]).reshape(dim_a * dim_b, r)
                    
                    # T_c_flat: T[a, b, c] for all a, b
                    # T_mats[a] is (dim_b, dim_c). We want column c.
                    T_c_flat = np.zeros(dim_a * dim_b)
                    for a in range(dim_a):
                        T_c_flat[a * dim_b : (a + 1) * dim_b] = T_mats[a][:, c]
                        
                    w, residuals, rank_p, s = np.linalg.lstsq(P_c, T_c_flat, rcond=None)
                    W[c, :] = w
                    
                # Check convergence
                # Compute residual
                # Res = T - sum_k U[:, k] V[:, k] W[:, k]
                # We can compute error on the non-zero entries only?
                # Or just compute norm of T - Reconstructed
                # Since T is sparse, we can check if the reconstructed value at non-zero spots is 1
                # and 0 elsewhere?
                # Actually, if rank is sufficient, it should be exact.
                # Let's check max error at non-zero entries.
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a_idx = i * m + j
                            b_idx = j * p + k
                            c_idx = k * n + i
                            
                            # Recalc T[a, b, c]
                            val = 0.0
                            for r_idx in range(r):
                                val += U[a_idx, r_idx] * V[b_idx, r_idx] * W[c_idx, r_idx]
                            
                            # T[a, b, c] is 1
                            diff = abs(val - 1.0)
                            if diff > error:
                                error = diff
                                
                if error < 1e-5:
                    # Check if we can round to integers/half-integers
                    # Multiply by 2 to check for integers
                    U2 = U * 2
                    V2 = V * 2
                    W2 = W * 2
                    
                    # Check if close to integers
                    if (np.max(np.abs(U2 - np.round(U2))) < 1e-3 and
                        np.max(np.abs(V2 - np.round(V2))) < 1e-3 and
                        np.max(np.abs(W2 - np.round(W2))) < 1e-3):
                        
                        U_round = np.round(U2).astype(int) / 2.0
                        V_round = np.round(V2).astype(int) / 2.0
                        W_round = np.round(W2).astype(int) / 2.0
                        
                        # Verify exactness with rounded values
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    a_idx = i * m + j
                                    b_idx = j * p + k
                                    c_idx = k * n + i
                                    
                                    val = 0.0
                                    for r_idx in range(r):
                                        val += U_round[a_idx, r_idx] * V_round[b_idx, r_idx] * W_round[c_idx, r_idx]
                                    if abs(val - 1.0) > 1e-9:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            return U_round, V_round, W_round, r
            
        return None

    # Try ranks 44 to 47
    for r in [44, 45, 46, 47]:
        if time.time() > deadline:
            break
        res = try_rank_als(r, max_iters=200, restarts=10)
        if res is not None:
            U, V, W, rank = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
