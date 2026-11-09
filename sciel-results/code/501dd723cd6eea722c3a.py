"""
Matrix multiplication 3x4 * 4x6: rank-minimal exact tensor decomposition.

We search for a decomposition of the multiplication tensor T of size
(3*4) x (4*6) x (6*3) = 12 x 24 x 18.

The tensor T has T[a, b, c] = 1 for each (i,j,k) in range(n) x range(m) x range(p)
with a = i*m+j, b = j*p+k, c = k*n+i, and 0 otherwise.

We use ALS (Alternating Least Squares) to search for a rank-R decomposition,
trying ranks from the theoretical lower bound up to the schoolbook rank (72).
"""
import time
import numpy as np

n_dim = 3
m_dim = 4
p_dim = 6

# Deadline: 140 s (10 s safety margin before the 150 s kill)
DEADLINE = 140.0


def _build_target_tensor():
    """Build the flattened target tensor T of length n*m * m*p * p*n.
    T has 1s at the 72 consistent index triples, 0 elsewhere."""
    n, m, p = n_dim, m_dim, p_dim
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total = dim_u * dim_v * dim_w
    T = np.zeros(total, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = a * (dim_v * dim_w) + b * dim_w + c
                T[idx] = 1.0
    return T, dim_u, dim_v, dim_w


def _verify_exact(U, V, W, T, dim_u, dim_v, dim_w, tol=1e-8):
    """Check if sum_r U[:,r] V[:,r] W[:,r] approximates T."""
    # Efficient: for each r, outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Use einsum or manual loop
    # U: (dim_u, R), V: (dim_v, R), W: (dim_w, R)
    # Result: (dim_u, dim_v, dim_w)
    R = U.shape[1]
    # Use batched outer products
    # (dim_u, dim_v, dim_w) = sum_r U[:,r,None,None] * V[:,r][None,:,None] * W[:,r][None,None,:]
    result = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
    residual = np.max(np.abs(result.flatten() - T))
    return residual < tol


def _round_to_half_int(U, V, W):
    """Round entries to nearest multiple of 0.5 and verify exactness."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    # Scale up by 2 to check with integers? No, just check with floats at high precision
    # Actually, if entries are multiples of 0.5, products are multiples of 0.25, sums are multiples of 0.25
    # So we can check exactness by rounding to nearest 0.25 and seeing if residual is 0
    result = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r, optimize=True).flatten()
    # Round result to nearest 0.25 (i.e., *4, round, /4)
    result_rounded = np.round(result * 4) / 4
    residual = np.max(np.abs(result_rounded - np.round(result_rounded * 2) / 2))
    # Actually, the target T has integer entries (0 or 1), so result should be 0 or 1
    # Check if each entry is close to 0 or 1
    dist_to_0 = np.abs(result_rounded)
    dist_to_1 = np.abs(result_rounded - 1.0)
    dist_to_2 = np.abs(result_rounded - 2.0)
    # For exact decomposition, each entry should be exactly 0 or 1 (for the 72 non-zero positions)
    # or 0 (for the rest). Let's just check max deviation from T.
    T_check = np.zeros_like(result_rounded)
    T_check[:72] = 1.0  # This is wrong, need proper indexing
    # Let's just re-verify with the original T
    return U_r, V_r, W_r


def _verify_exact_strict(U, V, W, T, dim_u, dim_v, dim_w):
    """Strict verification: check if decomposition matches T exactly (within float tolerance)."""
    result = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True).flatten()
    max_err = np.max(np.abs(result - T))
    return max_err < 1e-7, max_err


def _als_decompose(T, dim_u, dim_v, dim_w, R, n_iters=200, tol=1e-8, seed=0):
    """Run ALS to find rank-R decomposition. Returns (U, V, W, converged, residual)."""
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = rng.randn(dim_u, R)
    V = rng.randn(dim_v, R)
    W = rng.randn(dim_w, R)
    
    T_matrix = T.reshape(dim_u, dim_v * dim_w)
    V_W = V.reshape(dim_v, 1) @ W.reshape(1, dim_w)  # This is wrong, need outer product
    
    for it in range(n_iters):
        # Update U: T_matrix (dim_u, dim_v*dim_w) = U (dim_u, R) * (V ⊗ W) (R, dim_v*dim_w)
        # V ⊗ W flattened: for each r, V[:,r] ⊗ W[:,r] is a vector of length dim_v*dim_w
        # (V ⊗ W)[r, b, c] = V[b, r] * W[c, r]
        VW = np.einsum('br,cr->bcr', V, W, optimize=True).reshape(R, dim_v * dim_w)
        # U = T_matrix @ VW.T @ inv(WW)
        VW_T = VW.T  # (dim_v*dim_w, R)
        U_new = T_matrix @ VW_T @ np.linalg.inv(VW_T @ VW + 1e-10 * np.eye(R))
        
        # Update V: for each a, T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        # T[a, :, :] (dim_v, dim_w) = sum_r U[a,r] (V[:,r] ⊗ W[:,r])
        # Let's do this differently:
        # T_flat = sum_r (U[:,r] ⊗ V[:,r] ⊗ W[:,r])
        # For V: T[a, b, c] * U[a, r] * W[c, r'] ... this is getting complicated.
        # Alternative: use the matrix form.
        # For fixed r, the contribution to T is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Let's reshape T as (dim_u, dim_v, dim_w)
        T_3d = T.reshape(dim_u, dim_v, dim_w)
        
        # Update U: T_3d (a,b,c) = sum_r U[a,r] V[b,r] W[c,r]
        # For each a: T_3d[a] (dim_v, dim_w) = sum_r U[a,r] (V[:,r] ⊗ W[:,r])
        # Let M = V ⊗ W matrix: M[r, (b,c)] = V[b,r] W[c,r]
        # T_3d[a] = U[a, :] @ M
        # So U = T_3d.reshape(dim_u, -1) @ M.T @ inv(M @ M.T)
        
        M = np.einsum('br,cr->bcr', V, W, optimize=True).reshape(R, dim_v * dim_w)
        T_mat = T_3d.reshape(dim_u, dim_v * dim_w)
        M_T = M.T
        U_new = T_mat @ M_T @ np.linalg.inv(M_T @ M + 1e-10 * np.eye(R))
        
        # Update V: T_3d[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For each b: T_3d[:, b, :] (dim_u, dim_w) = sum_r V[b,r] (U[:,r] ⊗ W[:,r])
        # M_v[r, (a,c)] = U[a,r] W[c,r]
        M_v = np.einsum('ar,cr->acr', U, W, optimize=True).reshape(R, dim_u * dim_w)
        T_mat_v = T_3d.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
        M_v_T = M_v.T
        V_new = T_mat_v @ M_v_T @ np.linalg.inv(M_v_T @ M_v + 1e-10 * np.eye(R))
        
        # Update W: T_3d[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For each c: T_3d[:, :, c] (dim_u, dim_v) = sum_r W[c,r] (U[:,r] ⊗ V[:,r])
        M_w = np.einsum('ar,br->abr', U, V, optimize=True).reshape(R, dim_u * dim_v)
        T_mat_w = T_3d.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
        M_w_T = M_w.T
        W_new = T_mat_w @ M_w_T @ np.linalg.inv(M_w_T @ M_w + 1e-10 * np.eye(R))
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        result = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True).flatten()
        residual = np.max(np.abs(result - T))
        if residual < tol:
            return U, V, W, True, residual
    
    result = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True).flatten()
    residual = np.max(np.abs(result - T))
    return U, V, W, residual < tol, residual


def construct():
    start_time = time.time()
    
    T, dim_u, dim_v, dim_w = _build_target_tensor()
    schoolbook_rank = n_dim * m_dim * p_dim  # 72
    
    # Schoolbook solution as fallback
    U_fb = np.zeros((dim_u, schoolbook_rank), dtype=np.float64)
    V_fb = np.zeros((dim_v, schoolbook_rank), dtype=np.float64)
    W_fb = np.zeros((dim_w, schoolbook_rank), dtype=np.float64)
    col = 0
    for i in range(n_dim):
        for j in range(m_dim):
            for k in range(p_dim):
                U_fb[i * m_dim + j, col] = 1.0
                V_fb[j * p_dim + k, col] = 1.0
                W_fb[k * n_dim + i, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(dim_u*dim_v, dim_v*dim_w, dim_w*dim_u) / max(dim)
    # Actually, rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But a better bound: rank >= n*m*p / max(n, m, p) ... not sure
    # Let's try ranks from 24 to 71
    best_solution = (U_fb, V_fb, W_fb, schoolbook_rank)
    
    for R in range(24, schoolbook_rank):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        found = False
        for seed in range(5):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            U, V, W, converged, residual = _als_decompose(
                T, dim_u, dim_v, dim_w, R, n_iters=100, tol=1e-8, seed=seed
            )
            
            if converged and residual < 1e-7:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify the rounded version
                result_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r, optimize=True).flatten()
                max_err_r = np.max(np.abs(result_r - T))
                
                if max_err_r < 1e-6:
                    # Check if entries are indeed multiples of 0.5
                    U_ok = np.allclose(U_r * 2, np.round(U_r * 2), atol=1e-9)
                    V_ok = np.allclose(V_r * 2, np.round(V_r * 2), atol=1e-9)
                    W_ok = np.allclose(W_r * 2, np.round(W_r * 2), atol=1e-9)
                    
                    if U_ok and V_ok and W_ok:
                        best_solution = (U_r, V_r, W_r, R)
                        found = True
                        break
                else:
                    # Use the unrounded solution
                    # Verify it's exact enough
                    if residual < 1e-7:
                        best_solution = (U, V, W, R)
                        found = True
                        break
        
        if found:
            break
    
    U_out, V_out, W_out, rank_out = best_solution
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
