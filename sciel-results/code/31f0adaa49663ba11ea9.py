import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    dim_u = n * m
    dim_v = m * p
    dim_w = n * p

    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def try_rank_27():
        if time.time() - start_time > 130:
            return None
            
        rank = 27
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Precompute non-zero indices of T for residual calculation if needed, 
        # but the fast gradient trick handles the reconstruction part efficiently.
        # We still need to compute the error on T's non-zeros to check convergence.
        t_indices = np.argwhere(T != 0)
        
        for restart in range(20):
            if time.time() - start_time > 130:
                break
                
            np.random.seed(1000 + restart)
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # Normalize to avoid scaling issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)

            lr = 0.01
            momentum = 0.9
            v_U = np.zeros_like(U)
            v_V = np.zeros_like(V)
            v_W = np.zeros_like(W)
            
            for step in range(5000):
                if time.time() - start_time > 130:
                    break
                
                # Compute H = (V^T V) * (W^T W)
                # V^T V is rank x rank
                # W^T W is rank x rank
                MtV = V.T @ V
                MtW = W.T @ W
                H = MtV * MtW # Hadamard product
                
                # Gradient for U:
                # G_U = -2 * (T_residual_contraction - U @ H)
                # T_residual_contraction[i, r] = sum_{j,k} T[i,j,k] V[j,r] W[k,r]
                # Since T is sparse, we compute this explicitly for non-zeros.
                # T is 1 at (i, j, k).
                # So term is V[j,r] * W[k,r].
                # We can precompute this for each (i, r)?
                # T has 32 entries.
                # For each i, the non-zeros are at specific (j,k).
                
                # Efficiently compute T_contraction_U
                # T_contraction_U[i, r] = sum_{(j,k) in supp(T_i)} V[j,r] W[k,r]
                # We can loop over the 32 non-zeros.
                T_con_U = np.zeros((dim_u, rank))
                T_con_V = np.zeros((dim_v, rank))
                T_con_W = np.zeros((dim_w, rank))
                
                for idx in t_indices:
                    i, j, k = idx
                    # T[i,j,k] = 1
                    T_con_U[i, :] += V[j, :] * W[k, :]
                    T_con_V[j, :] += U[i, :] * W[k, :]
                    T_con_W[k, :] += U[i, :] * V[j, :]
                
                # Reconstruction term for U
                # R_U = U @ H
                # Note: H is symmetric? MtV and MtW are symmetric. H is symmetric.
                R_U = U @ H
                R_V = V @ (MtU * MtW) # Need MtU
                R_W = W @ (MtU * MtV)
                
                MtU = U.T @ U
                # Recalculate H for V and W properly
                # H_V = MtU * MtW
                # H_W = MtU * MtV
                R_V = V @ (MtU * MtW)
                R_W = W @ (MtU * MtV)
                
                G_U = -2.0 * (T_con_U - R_U)
                G_V = -2.0 * (T_con_V - R_V)
                G_W = -2.0 * (T_con_W - R_W)
                
                # Update with Momentum
                v_U = momentum * v_U - lr * G_U
                v_V = momentum * v_V - lr * G_V
                v_W = momentum * v_W - lr * W
                
                U += v_U
                V += v_V
                W += v_W
                
                # Optional: Renormalize?
                # Not strictly necessary, but helps stability
                # U /= np.linalg.norm(U)
                
                # Check error periodically
                if step % 100 == 0:
                    # Compute error on T
                    # T_hat[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                    err = 0.0
                    for idx in t_indices:
                        i, j, k = idx
                        val = np.dot(U[i, :], V[j, :] * W[k, :])
                        err += (1.0 - val) ** 2
                    # Also check zeros?
                    # If we minimize error on non-zeros, zeros might grow.
                    # We should include zeros in error if we want exact match.
                    # But the gradient trick only handles the "reconstruction" part.
                    # The gradient derived from full Frobenius norm is:
                    # G_U[i,r] = -2 sum_{j,k} (T[i,j,k] - U[i,:]V[j,:]W[k,:]) V[j,r] W[k,r]
                    # = -2 (T_con_U[i,r] - R_U[i,r])
                    # This IS the full gradient. So minimizing this minimizes full Frobenius error.
                    
                    if err < best_err:
                        best_err = err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        
                        # Check exactness with rounding
                        U_rounded = np.round(best_U * 2) / 2.0
                        V_rounded = np.round(best_V * 2) / 2.0
                        W_rounded = np.round(best_W * 2) / 2.0
                        
                        # Verify
                        T_recon = np.zeros((dim_u, dim_v, dim_w))
                        for r_idx in range(rank):
                            T_recon += np.einsum('i,j,k->ijk', U_rounded[:, r_idx], V_rounded[:, r_idx], W_rounded[:, r_idx])
                        
                        if np.allclose(T_recon, T, atol=1e-9):
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                            
            # If we got very close but not exact, try to round and check again
            if best_err < 1e-6:
                 U_rounded = np.round(best_U * 2) / 2.0
                 V_rounded = np.round(best_V * 2) / 2.0
                 W_rounded = np.round(best_W * 2) / 2.0
                 
                 T_recon = np.zeros((dim_u, dim_v, dim_w))
                 for r_idx in range(rank):
                     T_recon += np.einsum('i,j,k->ijk', U_rounded[:, r_idx], V_rounded[:, r_idx], W_rounded[:, r_idx])
                 
                 if np.allclose(T_recon, T, atol=1e-9):
                     return {
                         "U": U_rounded.tolist(),
                         "V": V_rounded.tolist(),
                         "W": W_rounded.tolist()
                     }

        return None

    res_27 = try_rank_27()
    if res_27 is not None:
        return res_27

    # Fallback to Rank 28
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_off + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_off + r) * p + (b_col_off + c)
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_off + c) * n + r
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(2, 2, 0, 0, 7)
    apply_strassen(0, 0, 2, 2, 14)
    apply_strassen(2, 2, 2, 2, 21)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B

    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc

    if not np.allclose(C, Cf):
        print("Mismatch")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
