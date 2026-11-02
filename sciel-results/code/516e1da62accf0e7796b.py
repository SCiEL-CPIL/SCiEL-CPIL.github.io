"""Find rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the 3-way tensor T[i,j,k] = 1 if k==j else 0, shape (n, m, p)."""
    T = np.zeros((n, m, p), dtype=float)
    for j in range(m):
        for i in range(n):
            T[i, j, j] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T (as tensors)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    # T_reconstructed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_rec = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_rec[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(T_rec, T, atol=1e-9)

def _try_rank(T, n, m, p, target_rank, max_iter=5000, seed=42):
    """Try to find a decomposition of given rank via iterative fitting."""
    rng = np.random.RandomState(seed)
    T_3d = T.reshape(n, m, p)
    
    best_err = float('inf')
    for attempt in range(3):
        # Random initialization
        U = rng.randn(n*m, target_rank)
        V = rng.randn(m*p, target_rank)
        W = rng.randn(p*n, target_rank)
        # Normalize columns
        for r in range(target_rank):
            nU = np.linalg.norm(U[:, r])
            nV = np.linalg.norm(V[:, r])
            nW = np.linalg.norm(W[:, r])
            if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                scale = (nU * nV * nW) ** (1.0/3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        for it in range(max_iter):
            # Update U: for each r, U[:,r] = argmin sum_{i,j} (T[i,j,*] contracted with V[:,r],W[:,r])^2
            # Actually: fix V,W, update U. T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
            # For fixed r: T_slice_r[i,j] = sum_k T[i,j,k] * V[j,k,r] * W[k,i,r]
            # U[i,j,r] should match T_slice_r[i,j] / (norm factor)
            for r in range(target_rank):
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                # T_slice[i,j] = sum_k T[i,j,k] * v_r[j,k] * w_r[k,i]
                T_slice = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        T_slice[i, j] = np.sum(T_3d[i, j, :] * v_r[j, :] * w_r[:, i])
                U[:, r] = T_slice.reshape(-1)
            
            # Update V
            for r in range(target_rank):
                u_r = U[:, r].reshape(n, m)
                w_r = W[:, r].reshape(p, n)
                T_slice = np.zeros((m, p))
                for j in range(m):
                    for k in range(p):
                        T_slice[j, k] = np.sum(T_3d[:, j, k] * u_r[:, j] * w_r[k, :])
                V[:, r] = T_slice.reshape(-1)
            
            # Update W
            for r in range(target_rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                T_slice = np.zeros((p, n))
                for k in range(p):
                    for i in range(n):
                        T_slice[k, i] = np.sum(T_3d[i, :, k] * u_r[i, :] * v_r[:, k])
                W[:, r] = T_slice.reshape(-1)
            
            # Compute error
            T_rec = np.zeros((n, m, p))
            for r in range(target_rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        T_rec[i, j, :] += u_r[i, j] * v_r[j, :] * w_r[:, i]
            err = np.linalg.norm(T_rec - T_3d)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-7:
                # Try to round to integers/half-integers
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                if _check_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist(), True
                return U.tolist(), V.tolist(), W.tolist(), True
        if best_err < 1e-7:
            return best_U.tolist(), best_V.tolist(), best_W.tolist(), True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try ranks from low to high
    for rank in range(18, 40):
        U, V, W, success = _try_rank(T, n, m, p, rank, max_iter=3000, seed=rank*7+13)
        if success:
            # Verify
            if _check_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: use the known rank-36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        v_idx_00 = b_row_offset * p + b_col_offset
        v_idx_01 = b_row_offset * p + b_col_offset + 1
        v_idx_10 = (b_row_offset + 1) * p + b_col_offset
        v_idx_11 = (b_row_offset + 1) * p + b_col_offset + 1
        w_idx_00 = c_col_offset * n + c_row_offset
        w_idx_01 = (c_col_offset + 1) * n + c_row_offset
        w_idx_10 = c_col_offset * n + c_row_offset + 1
        w_idx_11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        U[u_idx_00][col] += 1; U[u_idx_11][col] += 1; V[v_idx_00][col] += 1; V[v_idx_11][col] += 1; W[w_idx_00][col] += 1; W[w_idx_11][col] += 1; col += 1
        U[u_idx_10][col] += 1; U[u_idx_11][col] += 1; V[v_idx_00][col] += 1; W[w_idx_10][col] += 1; W[w_idx_11][col] -= 1; col += 1
        U[u_idx_00][col] += 1; V[v_idx_01][col] += 1; V[v_idx_11][col] -= 1; W[w_idx_01][col] += 1; W[w_idx_11][col] += 1; col += 1
        U[u_idx_11][col] += 1; V[v_idx_10][col] += 1; V[v_idx_00][col] -= 1; W[w_idx_00][col] += 1; W[w_idx_10][col] += 1; col += 1
        U[u_idx_00][col] += 1; U[u_idx_01][col] += 1; V[v_idx_11][col] += 1; W[w_idx_00][col] -= 1; W[w_idx_01][col] += 1; col += 1
        U[u_idx_10][col] += 1; U[u_idx_00][col] -= 1; V[v_idx_00][col] += 1; V[v_idx_01][col] += 1; W[w_idx_11][col] += 1; col += 1
        U[u_idx_01][col] += 1; U[u_idx_11][col] -= 1; V[v_idx_10][col] += 1; V[v_idx_11][col] += 1; W[w_idx_00][col] += 1; col += 1

    def add_schoolbook_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        v_idx_0 = b_row_offset * p + b_col_offset
        v_idx_1 = (b_row_offset + 1) * p + b_col_offset
        w_idx_0 = c_col_offset * n + c_row_offset
        w_idx_1 = c_col_offset * n + c_row_offset + 1
        U[u_idx_00][col] = 1; V[v_idx_0][col] = 1; W[w_idx_0][col] = 1; col += 1
        U[u_idx_01][col] = 1; V[v_idx_1][col] = 1; W[w_idx_0][col] = 1; col += 1
        U[u_idx_10][col] = 1; V[v_idx_0][col] = 1; W[w_idx_1][col] = 1; col += 1
        U[u_idx_11][col] = 1; V[v_idx_1][col] = 1; W[w_idx_1][col] = 1; col += 1

    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
