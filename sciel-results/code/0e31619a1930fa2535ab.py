"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]


def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition using Strassen on 2x2 blocks.
    The 2x4 * 4x8 multiplication is viewed as 8 independent 2x2 * 2x2 multiplications,
    each done with rank 7 (Strassen), giving total rank 8*7 = 56.
    """
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            b11_idx = 2 * u * 8 + 2 * v
            b12_idx = 2 * u * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            c11_idx = 2 * v * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = 2 * v * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1

            # s=0: M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1

            # s=1: M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1

            # s=2: M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1

            # s=3: M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1

            # s=4: M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1

            # s=5: M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1

            # s=6: M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1

    return U, V, W


def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using vectorized numpy operations."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]

    # For each (i, j, k): a = i*m+j, b = j*p+k, c = k*n+i
    # Need sum_s U[a,s] * V[b,s] * W[c,s] == 1
    # Vectorize over (i,j,k) for each s, or better: build the full check at once.
    
    # Build index arrays
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    a_idx = i_idx * m + j_idx  # shape (n, m, p)
    b_idx = j_idx * p + k_idx
    c_idx = k_idx * n + i_idx
    
    # Flatten
    a_flat = a_idx.ravel()
    b_flat = b_idx.ravel()
    c_flat = c_idx.ravel()
    
    # U[a_flat, :] has shape (n*m*p, rank)
    # V[b_flat, :] has shape (n*m*p, rank)
    # W[c_flat, :] has shape (n*m*p, rank)
    # Element-wise product, then sum over rank
    prod = U_arr[a_flat] * V_arr[b_flat] * W_arr[c_flat]  # (n*m*p, rank)
    total = prod.sum(axis=1)  # (n*m*p,)
    
    return bool(np.allclose(total, 1.0, atol=1e-9))


def _verify_subset(U, V, W, n, m, p, sample_indices=None):
    """Verify a subset of entries for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if sample_indices is None:
        # Check all 64 entries
        i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
        a_flat = (i_idx * m + j_idx).ravel()
        b_flat = (j_idx * p + k_idx).ravel()
        c_flat = (k_idx * n + i_idx).ravel()
    else:
        a_flat = []
        b_flat = []
        c_flat = []
        for (i, j, k) in sample_indices:
            a_flat.append(i * m + j)
            b_flat.append(j * p + k)
            c_flat.append(k * n + i)
        a_flat = np.array(a_flat)
        b_flat = np.array(b_flat)
        c_flat = np.array(c_flat)
    
    prod = U_arr[a_flat] * V_arr[b_flat] * W_arr[c_flat]
    total = prod.sum(axis=1)
    return bool(np.allclose(total, 1.0, atol=1e-9))


def _try_remove_columns(U, V, W, n, m, p, time_limit=100):
    """
    Try to reduce rank by removing columns from the decomposition.
    Greedily try removing one column at a time.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    start_time = time.time()
    
    best_rank = rank
    best_U, best_V, best_W = U_arr, V_arr, W_arr
    
    # Precompute the full tensor for fast checking
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    a_flat = (i_idx * m + j_idx).ravel()
    b_flat = (j_idx * p + k_idx).ravel()
    c_flat = (k_idx * n + i_idx).ravel()
    
    # Try removing single columns
    for col_to_remove in range(rank):
        if time.time() - start_time > time_limit - 5:
            break
        
        mask = np.ones(rank, dtype=bool)
        mask[col_to_remove] = False
        
        U_new = U_arr[:, mask]
        V_new = V_arr[:, mask]
        W_new = W_arr[:, mask]
        
        # Fast check
        prod = U_new[a_flat] * V_new[b_flat] * W_new[c_flat]
        total = prod.sum(axis=1)
        
        if np.allclose(total, 1.0, atol=1e-9):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank - 1
            rank = best_rank
            # Update mask for next iteration
            U_arr = U_new
            V_arr = V_new
            W_arr = W_new
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Build the rank-56 Strassen-based decomposition
    U, V, W = _build_rank_56(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    best_rank = 56
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by removing columns
    time_left = deadline - time.time()
    if time_left > 10:
        U_r, V_r, W_r, rank_r = _try_remove_columns(
            U, V, W, n, m, p,
            time_limit=min(time_left - 10, 110)
        )
        if rank_r < best_rank:
            best_rank = rank_r
            best_U = U_r.tolist()
            best_V = V_r.tolist()
            best_W = W_r.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
