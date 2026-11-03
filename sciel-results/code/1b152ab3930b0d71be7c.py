"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
from itertools import product
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k' and i==i', else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _matricize_and_decompose(T, n, m, p):
    """Matricize T as (m*p) x (n*m * p*n) and find rank-r decomposition.
    Returns (V, U, W) where V is (m*p, r), U is (n*m, r), W is (p*n, r).
    """
    # T has shape (n*m, m*p, p*n)
    # Matricize: M[j_idx, col_idx] where j_idx indexes (m*p) dim
    # col_idx flattens (n*m, p*n) dims
    # M = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    M = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    
    # Find rank via SVD
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Determine numerical rank
    tol = 1e-10
    num_rank = np.sum(s > tol)
    
    if num_rank == 0:
        return None
    
    # Use the top num_rank components
    A = U_svd[:, :num_rank] * s[:num_rank]  # (m*p, num_rank)
    B = Vt[:num_rank, :]  # (num_rank, n*m * p*n)
    
    # Reshape B to (n*m, p*n, num_rank)
    B_reshaped = B.reshape(n * m, p * n, num_rank)
    
    # Now B_reshaped[i_idx, k_idx, r] should equal U[i_idx, r] * W[k_idx, r]
    # We need to factor B_reshaped into U and W.
    # B_reshaped[i, k, r] = U[i, r] * W[k, r]
    # For each r, B_reshaped[:, :, r] is a (n*m) x (p*n) matrix of rank 1.
    # We can extract U[:, r] and W[:, r] from it.
    
    U_mat = np.zeros((n * m, num_rank))
    W_mat = np.zeros((p * n, num_rank))
    
    for r in range(num_rank):
        M_r = B_reshaped[:, :, r]  # (n*m) x (p*n)
        # Find a non-zero entry to normalize
        # M_r[i, k] = U[i, r] * W[k, r]
        # Pick a row i0 with non-zero norm
        row_norms = np.linalg.norm(M_r, axis=1)
        nz_rows = np.where(row_norms > tol)[0]
        if len(nz_rows) == 0:
            continue
        i0 = nz_rows[0]
        # M_r[i0, :] = U[i0, r] * W[:, r]
        # Find a non-zero entry in this row
        row_vals = M_r[i0, :]
        nz_cols = np.where(np.abs(row_vals) > tol)[0]
        if len(nz_cols) == 0:
            continue
        k0 = nz_cols[0]
        # M_r[i0, k0] = U[i0, r] * W[k0, r]
        # Set U[i0, r] = 1, W[k0, r] = M_r[i0, k0]
        U_mat[i0, r] = 1.0
        W_mat[k0, r] = M_r[i0, k0]
        # Fill in the rest
        for i in range(n * m):
            if i != i0:
                # M_r[i, k0] = U[i, r] * W[k0, r]
                if abs(W_mat[k0, r]) > tol:
                    U_mat[i, r] = M_r[i, k0] / W_mat[k0, r]
        for k in range(p * n):
            if k != k0:
                # M_r[i0, k] = U[i0, r] * W[k, r]
                if abs(U_mat[i0, r]) > tol:
                    W_mat[k, r] = M_r[i0, k] / U_mat[i0, r]
    
    V_mat = A  # (m*p, num_rank)
    
    return V_mat, U_mat, W_mat

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reproduces the tensor."""
    T = _build_tensor(n, m, p)
    R = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(R):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                expected = 1.0
                max_err = max(max_err, abs(val - expected))
    # Also check that off-diagonal entries are 0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            if j != j2 or k != k2 or i != i2:
                                val = 0.0
                                for r in range(R):
                                    val += U[i * m + j, r] * V[j2 * p + k, r] * W[k2 * n + i2, r]
                                max_err = max(max_err, abs(val))
    return max_err

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Build tensor and decompose
    T = _build_tensor(n, m, p)
    result = _matricize_and_decompose(T, n, m, p)
    
    if result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    V_mat, U_mat, W_mat = result
    
    # Verify
    max_err = _verify_decomposition(U_mat, V_mat, W_mat, n, m, p)
    
    if max_err > 1e-6:
        # Decomposition not exact, try to fix or fallback
        # Try to clean up numerical noise
        U_mat = np.round(U_mat, 10)
        V_mat = np.round(V_mat, 10)
        W_mat = np.round(W_mat, 10)
        max_err = _verify_decomposition(U_mat, V_mat, W_mat, n, m, p)
    
    if max_err > 1e-4:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists with clean values
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    # Clean up near-zero and near-integer values
    for arr in [U_list, V_list, W_list]:
        for row in arr:
            for idx in range(len(row)):
                val = row[idx]
                if abs(val) < 1e-10:
                    row[idx] = 0.0
                else:
                    # Try to snap to simple fractions
                    for denom in [1, 2, 3, 4, 6]:
                        num = round(val * denom)
                        if abs(val - num / denom) < 1e-10:
                            row[idx] = num / denom
                            break
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
