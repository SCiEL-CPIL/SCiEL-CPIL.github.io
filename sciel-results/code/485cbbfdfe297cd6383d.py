"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen, each shape (4,7).
    U[i*2+j, t] for A[i,j], V[j*2+k, t] for B[j,k], W[k*2+i, t] for C[i,k]."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W


def _build_rank77():
    """Build rank-77 decomposition: 49 for 4x4x4 + 28 for 4x4x2."""
    n, m, p = 4, 4, 6
    U_s, V_s, W_s = _strassen_2x2()
    
    total_rank = 77
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Part 1: 4x4x4 recursive Strassen (columns 0-48)
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I, i0 = i // 2, i % 2
                for j in range(4):
                    J, j0 = j // 2, j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val
            
            for j in range(4):
                J, j0 = j // 2, j % 2
                for k in range(4):
                    K, k0 = k // 2, k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val
            
            for k in range(4):
                K, k0 = k // 2, k % 2
                for i in range(4):
                    I, i0 = i // 2, i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val
    
    # Part 2: 4x4x2 block Strassen (columns 49-76)
    # Split the 2 output columns into two 1-column blocks, each handled by 2x2x2 Strassen
    # Block (0,0)-(1,1) for col 0, Block (2,0)-(3,1) for col 1
    block_products = [
        (0, 0, 0, 0, 0, 0),  # rows 0-1, cols 0-1 of A, rows 0-1 of B, col 0 of output
        (0, 2, 0, 1, 0, 1),  # rows 0-1, cols 2-3 of A, rows 0-1 of B, col 1 of output
        (2, 0, 2, 0, 2, 0),  # rows 2-3, cols 0-1 of A, rows 2-3 of B, col 0 of output
        (2, 2, 2, 1, 2, 1),  # rows 2-3, cols 2-3 of A, rows 2-3 of B, col 1 of output
    ]
    
    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = 49 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val
            
            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val
            
            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val
    
    return U, V, W, total_rank


def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is correct."""
    Ua = np.array(U, dtype=float).reshape(n, m, rank)
    Va = np.array(V, dtype=float).reshape(m, p, rank)
    Wa = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.max(np.abs(T - 1.0))


def _try_rank_reduction(U, V, W, n, m, p, rank, deadline):
    """Try to reduce rank by removing zero columns or combining equivalent terms."""
    # First, check for zero columns
    zero_cols = []
    for c in range(rank):
        if np.all(U[:, c] == 0) and np.all(V[:, c] == 0) and np.all(W[:, c] == 0):
            zero_cols.append(c)
    
    if zero_cols:
        keep = [c for c in range(rank) if c not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)
    
    # Try to find linearly dependent columns in U
    # This is a heuristic: if two columns of U are identical, we might be able to merge
    # For now, just return the cleaned-up version
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit
    
    # Build the rank-77 decomposition
    U, V, W, total_rank = _build_rank77()
    
    # Verify correctness
    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=float)
        V_f = np.zeros((m * p, rank), dtype=float)
        W_f = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Try rank reduction if we have time
    if time.time() < deadline:
        U, V, W, total_rank = _try_rank_reduction(U, V, W, n, m, p, total_rank, deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
