"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """Build the rank-28 Strassen decomposition deterministically."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen 2x2x2 decomposition coefficients
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    # Indices for U: i * 4 + j (i in 0..1, j in 0..3)
    # Indices for V: j * 4 + k (j in 0..3, k in 0..3) -- Wait, V is B[j,k]?
    # In the code: V[k*p+j, r]. So V index is k * 4 + j.
    # Indices for W: k * 2 + i (k in 0..3, i in 0..1). W index is j * 2 + i.

    col = 0
    # Blocks:
    # Block 0: A1 (cols 0,1) * B11 (rows 0,1, cols 0,1) -> C1 (cols 0,1)
    # Block 1: A2 (cols 2,3) * B21 (rows 2,3, cols 0,1) -> C1 (cols 0,1)
    # Block 2: A1 (cols 0,1) * B12 (rows 0,1, cols 2,3) -> C2 (cols 2,3)
    # Block 3: A2 (cols 2,3) * B22 (rows 2,3, cols 2,3) -> C2 (cols 2,3)
    
    block_config = [
        (0, 0, 0, 0), # a_blk, b_row_blk, b_col_blk, c_col_blk
        (1, 2, 0, 0),
        (0, 0, 2, 2),
        (1, 2, 2, 2),
    ]
    
    for a_blk, b_row_blk, b_col_blk, c_col_blk in block_config:
        for s in range(7):
            c = col
            j0 = 2 * a_blk
            j1 = 2 * a_blk + 1
            k0 = 2 * b_row_blk
            k1 = 2 * b_row_blk + 1
            l0 = 2 * b_col_blk
            l1 = 2 * b_col_blk + 1
            
            # U contributions: A[i, j]
            U_contrib = {
                0: [(0, j0, 1), (1, j1, 1)],
                1: [(1, j0, 1), (1, j1, 1)],
                2: [(0, j0, 1)],
                3: [(1, j1, 1)],
                4: [(0, j0, 1), (0, j1, 1)],
                5: [(1, j0, 1), (0, j0, -1)],
                6: [(0, j1, 1), (1, j1, -1)],
            }
            for (i, j, coeff) in U_contrib[s]:
                U[i * 4 + j, c] += coeff

            # V contributions: B[k, l]
            V_contrib = {
                0: [(k0, l0, 1), (k1, l1, 1)],
                1: [(k0, l0, 1)],
                2: [(k0, l1, 1), (k1, l1, -1)],
                3: [(k1, l0, 1), (k0, l0, -1)],
                4: [(k1, l1, 1)],
                5: [(k0, l0, 1), (k0, l1, 1)],
                6: [(k1, l0, 1), (k1, l1, 1)],
            }
            for (k, l, coeff) in V_contrib[s]:
                V[k * 4 + l, c] += coeff

            # W contributions: C[i, l]
            # W index: l * 2 + i
            W_contrib = {
                0: {(0, l0): 1, (1, l1): 1},
                1: {(1, l0): 1, (1, l1): -1},
                2: {(0, l1): 1, (1, l1): 1},
                3: {(0, l0): 1, (1, l0): 1},
                4: {(0, l0): -1, (0, l1): 1},
                5: {(1, l1): 1},
                6: {(0, l0): 1},
            }
            for (i_loc, l_loc), coeff in W_contrib[s].items():
                W[l_loc * 2 + i_loc, c] += coeff

            col += 1

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _attempt_rank_reduction(U, V, W, n, m, p):
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W

    size = n * m * p
    tensors = np.zeros((rank, size))
    
    for r in range(rank):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * (m * p) + k * p + j
                    tensors[r, idx] = U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]

    try:
        U_svd, s, Vt = np.linalg.svd(tensors, full_matrices=False)
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            null_space = Vt[num_rank:]
            
            best_col_to_remove = -1
            best_coeff = 0.0
            best_null_vec = None
            
            for vec in null_space:
                max_idx = np.argmax(np.abs(vec))
                max_val = np.abs(vec[max_idx])
                if max_val > best_coeff:
                    best_coeff = max_val
                    best_col_to_remove = max_idx
                    best_null_vec = vec
            
            if best_col_to_remove != -1:
                new_U = np.delete(U, best_col_to_remove, axis=1)
                new_V = np.delete(V, best_col_to_remove, axis=1)
                new_W = np.delete(W, best_col_to_remove, axis=1)
                
                for r_new in range(new_U.shape[1]):
                    r_old = r_new if r_new < best_col_to_remove else r_new + 1
                    factor = 1.0 - best_null_vec[r_old] / best_null_vec[best_col_to_remove]
                    new_U[:, r_new] *= factor
                    new_V[:, r_new] *= factor
                    new_W[:, r_new] *= factor
                
                return new_U, new_V, new_W
                
    except np.linalg.LinAlgError:
        pass
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    U, V, W = _build_strassen_rank28()
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    assert _verify_decomposition(U, V, W, n, m, p), "Base rank-28 decomposition failed verification"
    
    U, V, W = _attempt_rank_reduction(U, V, W, n, m, p)
    
    if U.shape[1] < 28:
        if not _verify_decomposition(U, V, W, n, m, p):
            U, V, W = _build_strassen_rank28()
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
