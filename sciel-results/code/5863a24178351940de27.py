import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Construct the target tensor T
    # T[i, j, k] = 1 if i, j, k correspond to a term in A*B
    # A is n x m, B is m x p, C is n x p
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T has indices (i, j, k) where i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T[i, j, k] = 1 for all valid i, j, k? 
    # No, the tensor for matrix multiplication is T_{i j k} = delta_{j, j} ?
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k ?
    # No, the multiplication tensor is T = sum_{i,j,k} e_i^{(n)} \otimes e_j^{(m)} \otimes e_k^{(p)} ?
    # Wait, the tensor T_mm is defined such that <T_mm, A \otimes B> = <C, A B>
    # A \otimes B is a vector in R^{n m p}.
    # T_mm has 1s at positions corresponding to (i, j, k) where the term A[i,j] B[j,k] contributes to C[i,k].
    # In the trace convention W[k*n+i, r], the tensor is T[i, j, k] = 1.
    # So T is a 3D array of 1s?
    # No, T is the identity tensor?
    # The condition is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
    # So the target tensor T has T[i, j, k] = 1 for all i, j, k.
    
    T = np.ones((n, m, p))
    
    def verify_decomposition(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def is_half_integer(x):
        # Check if x is close to a half-integer
        return abs(x * 2 - np.round(x * 2)) < 1e-9

    def check_half_integers(U, V, W):
        if not all(is_half_integer(x) for x in U.flatten()): return False
        if not all(is_half_integer(x) for x in V.flatten()): return False
        if not all(is_half_integer(x) for x in W.flatten()): return False
        return True

    def try_als(rank, max_iters=100, tol=1e-12):
        # Initialize U, V, W
        np.random.seed(rank)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize
        U /= np.linalg.norm(U, axis=0)
        V /= np.linalg.norm(V, axis=0)
        W /= np.linalg.norm(W, axis=0)
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Update U
            # U[:, r] = (V \otimes W^T) \cdot T_r ?
            # We want min_U || T - sum U_r \otimes V_r \otimes W_r ||^2
            # For fixed V, W, U_r = argmin || T - sum U_r \otimes V_r \otimes W_r ||^2
            # This is a linear system for each r? No, U_r is a vector.
            # The term is U_r \otimes (V_r \otimes W_r).
            # Let M_r = V_r \otimes W_r (flattened).
            # We want to approximate T_flat by sum U_r_flat \otimes M_r? No.
            # T is 3D.
            # Let's flatten T to 1D of size n*m*p.
            # T_flat = sum_r U_r_flat \otimes (V_r \otimes W_r)_flattened?
            # No, the tensor product structure is different.
            # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
            
            # This is not a standard ALS on a flattened tensor because the indices are permuted.
            # We can rearrange T into a matrix.
            # Let A_mat be (n*m) x (p*n)?
            # T[i, j, k] -> index (i*m+j) and (k*n+i)?
            # For fixed i, j, k, the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
            
            # This is a bilinear form in U and (V*W)?
            # Let's try to update U.
            # Fix V, W.
            # We want to find U such that sum_r U[:, r] \otimes (V[:, r] \otimes W[:, r]) = T.
            # This is not linear in U if V, W are fixed?
            # Actually, for each r, the contribution to T is U[:, r] \otimes (V[:, r] \otimes W[:, r]).
            # This is a sum of rank-1 tensors in the space of (nm) x (mp) x (pn)?
            
            # Let's use a simpler approach: Gradient Descent.
            # But we need exactness.
            
            # Alternative: Use the fact that we have a good initial guess (Strassen).
            # We will stick to the structural block decomposition but try to optimize the "remainder" parts using a small-scale search.
            
            return None

    # Fallback to the proven Strassen decomposition
    cols = []
    
    def strassen_2x2x2_cols(i0, j0, k0, n_dim, m_dim, p_dim):
        cols_local = []
        cols_local.append((
            {i0*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        cols_local.append((
            {(i0+1)*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1},
            {k0*n_dim+(i0+1): 1, (k0+1)*n_dim+(i0+1): -1}
        ))
        cols_local.append((
            {i0*m_dim+j0: 1},
            {j0*p_dim+(k0+1): 1, (j0+1)*p_dim+(k0+1): -1},
            {(k0+1)*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        cols_local.append((
            {(i0+1)*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+k0: 1, j0*p_dim+k0: -1},
            {k0*n_dim+i0: 1, k0*n_dim+(i0+1): 1}
        ))
        cols_local.append((
            {i0*m_dim+j0: 1, i0*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: -1, (k0+1)*n_dim+i0: 1}
        ))
        cols_local.append((
            {(i0+1)*m_dim+j0: 1, i0*m_dim+j0: -1},
            {j0*p_dim+k0: 1, j0*p_dim+(k0+1): 1},
            {(k0+1)*n_dim+(i0+1): 1}
        ))
        cols_local.append((
            {i0*m_dim+(j0+1): 1, (i0+1)*m_dim+(j0+1): -1},
            {(j0+1)*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1}
        ))
        return cols_local

    def direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        for di in range(nc):
            for dj in range(mc):
                for dk in range(pc):
                    u_idx = (ii + di) * m_dim + (jj + dj)
                    v_idx = (jj + dj) * p_dim + (kk + dk)
                    w_idx = (kk + dk) * n_dim + (ii + di)
                    cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))

    def decompose_recursive(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        if time.time() > deadline:
            return
        if nc == 1 and mc == 1 and pc == 1:
            u_idx = ii * m_dim + jj
            v_idx = jj * p_dim + kk
            w_idx = kk * n_dim + ii
            cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
            return
        if nc == 1 or mc == 1 or pc == 1:
            direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            return
        if nc >= 2 and mc >= 2 and pc >= 2:
            cols.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
            if nc > 2:
                decompose_recursive(nc - 2, mc, pc, ii + 2, jj, kk, n_dim, m_dim, p_dim, cols)
            if mc > 2:
                decompose_recursive(2, mc - 2, pc, ii, jj + 2, kk, n_dim, m_dim, p_dim, cols)
            if pc > 2:
                decompose_recursive(2, 2, pc - 2, ii, jj, kk + 2, n_dim, m_dim, p_dim, cols)
            return
        # Fallback
        if nc > 1:
            decompose_recursive(1, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc - 1, mc, pc, ii + 1, jj, kk, n_dim, m_dim, p_dim, cols)
        elif mc > 1:
            decompose_recursive(nc, 1, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc - 1, pc, ii, jj + 1, kk, n_dim, m_dim, p_dim, cols)
        elif pc > 1:
            decompose_recursive(nc, mc, 1, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc, pc - 1, ii, jj, kk + 1, n_dim, m_dim, p_dim, cols)

    decompose_recursive(n, m, p, 0, 0, 0, n, m, p, cols)
    
    U = np.zeros((n * m, len(cols)), dtype=float)
    V = np.zeros((m * p, len(cols)), dtype=float)
    W = np.zeros((p * n, len(cols)), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(cols):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
