"""Rank-reduction search for 4x4 * 4x8 -> 4x8 matrix multiplication tensor.

Strategy: Start from a valid Strassen-based rank-98 decomposition, then greedily
remove linearly dependent columns to find the minimal rank. Use numpy for
efficient linear algebra checks.
"""

# EVOLVE-BLOCK-START

import time
import numpy as np

# 2x2x2 Strassen decomposition (rank 7)
_S2_U = [
    [1, 0, 0, 1],   # M1: (a+d)
    [0, 0, 1, 1],   # M2: (c+d)
    [1, 0, 0, 0],   # M3: a
    [0, 0, 0, 1],   # M4: d
    [1, 1, 0, 0],   # M5: (a+b)
    [1, 0, -1, 0],  # M6: (a-c)
    [0, 1, 0, -1],  # M7: (b-d)
]
_S2_V = [
    [1, 0, 0, 1],   # M1: (e+h)
    [1, 0, 0, 0],   # M2: e
    [0, 1, 0, -1],  # M3: (f-h)
    [0, 0, 1, -1],  # M4: (g-e)
    [0, 0, 0, 1],   # M5: h
    [1, 1, 0, 0],   # M6: (e+f)
    [0, 0, 1, 1],   # M7: (g+h)
]
_S2_W = [
    [1, 0, 0, -1, 1, 1, 0],   # C00 = M1 - M4 + M5 + M6
    [0, 0, 1, 0, 1, 0, 0],    # C01 = M3 + M5
    [0, 1, 0, 1, 0, 0, 0],    # C10 = M2 + M4
    [1, 1, -1, 0, 0, 0, 1],   # C11 = M1 - M3 + M2 + M7
]


def _build_strassen_4x4x4():
    """Build rank-49 decomposition for 4x4 * 4x4 -> 4x4 using Strassen."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))  # 16 x 49
    V = np.zeros((m * p, rank))  # 16 x 49
    W = np.zeros((p * n, rank))  # 16 x 49

    A_block_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(1, 0, 1), (1, 1, 1)],      # M2
        [(0, 0, 1)],                 # M3
        [(1, 1, 1)],                 # M4
        [(0, 0, 1), (0, 1, 1)],      # M5
        [(0, 0, 1), (1, 0, -1)],     # M6
        [(0, 1, 1), (1, 1, -1)],     # M7
    ]
    B_block_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(0, 0, 1)],                 # M2
        [(0, 1, 1), (1, 1, -1)],     # M3
        [(1, 0, 1), (0, 0, -1)],     # M4
        [(1, 1, 1)],                 # M5
        [(0, 0, 1), (0, 1, 1)],      # M6
        [(1, 0, 1), (1, 1, 1)],      # M7
    ]
    C_block_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(1, 0, 1), (1, 1, 1)],      # M2
        [(0, 1, 1), (1, 1, -1)],     # M3
        [(0, 0, -1), (1, 0, 1)],     # M4
        [(0, 0, 1), (0, 1, 1)],      # M5
        [(0, 0, 1)],                 # M6
        [(1, 1, 1)],                 # M7
    ]

    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            for (bi, bj, ca) in A_block_coeffs[t]:
                for i0 in range(2):
                    for j0 in range(2):
                        i = 2 * bi + i0
                        j = 2 * bj + j0
                        U[i * m + j, col] += ca * _S2_U[s][i0 * 2 + j0]

            for (bj, bk, cb) in B_block_coeffs[t]:
                for j0 in range(2):
                    for k0 in range(2):
                        j = 2 * bj + j0
                        k = 2 * bk + k0
                        V[j * p + k, col] += cb * _S2_V[s][j0 * 2 + k0]

            for (bi, bk, cc) in C_block_coeffs[t]:
                for i0 in range(2):
                    for k0 in range(2):
                        i = 2 * bi + i0
                        k = 2 * bk + k0
                        W[k * n + i, col] += cc * _S2_W[i0 * 2 + k0][s]

    return U, V, W


def _build_strassen_4x4x8():
    """Build rank-98 decomposition: Strassen 4x4x4 duplicated for 8-wide output."""
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_strassen_4x4x4()
    
    # U stays the same (A-side is 4x4)
    U = np.zeros((n * m, 98))
    U[:, :49] = U4
    
    # V: duplicate for both halves of the 8-wide output
    V = np.zeros((m * p, 98))
    V[:16, :49] = V4
    V[16:, :49] = V4  # left half k=0..3
    V[:16, 49:] = V4  # right half k=4..7
    V[16:, 49:] = V4
    
    # W: duplicate for both halves
    W = np.zeros((p * n, 98))
    W[:16, :49] = W4
    W[16:, :49] = W4
    W[:16, 49:] = W4
    W[16:, 49:] = W4
    
    return U, V, W


def _verify_numpy(U, V, W, n, m, p):
    """Verify decomposition using numpy tensor contraction."""
    rank = U.shape[1]
    # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    return np.allclose(T, 1.0, atol=1e-9)


def _try_remove_column(U, V, W, col, n, m, p):
    """Try removing column `col`. Returns (success, new_U, new_V, new_W)."""
    rank = U.shape[1]
    if rank <= 1:
        return False, None, None, None
    
    # Check if column is linearly dependent on others by checking
    # if the tensor contribution can be expressed as a combination of other columns.
    # Simple check: see if removing this column still gives valid decomposition.
    # This is expensive, so we use a heuristic: check if the column is "redundant"
    # by seeing if U[:,col], V[:,col], W[:,col] are in the span of other columns.
    
    # A column is redundant if there exist scalars alpha_c for c != col such that:
    # U[:,col] = sum_c alpha_c * U[:,c]  (component-wise, but this is too strict)
    
    # Better approach: just try removing and verify. But that's O(rank * n*m*p) per removal.
    # For now, use a simpler heuristic: if the column has very small norm relative to others, try it.
    
    u_norm = np.linalg.norm(U[:, col])
    v_norm = np.linalg.norm(V[:, col])
    w_norm = np.linalg.norm(W[:, col])
    
    if u_norm < 1e-10 or v_norm < 1e-10 or w_norm < 1e-10:
        # Zero column, safe to remove
        mask = [c for c in range(rank) if c != col]
        return True, U[:, mask], V[:, mask], W[:, mask]
    
    return False, None, None, None


def _greedy_rank_reduction(U, V, W, n, m, p, deadline):
    """Greedily remove linearly dependent columns."""
    rank = U.shape[1]
    removed = 0
    
    while time.time() < deadline:
        progress = False
        for col in range(rank - 1, -1, -1):
            if time.time() > deadline:
                break
            success, nU, nV, nW = _try_remove_column(U, V, W, col, n, m, p)
            if success:
                U, V, W = nU, nV, nW
                rank -= 1
                removed += 1
                progress = True
                break  # Restart from beginning after a removal
        if not progress:
            break
    
    return U, V, W, removed


def construct():
    t0 = time.time()
    deadline = t0 + 135  # safety margin before 150s kill
    
    n, m, p = 4, 4, 8
    
    # Build initial Strassen decomposition (rank 98)
    U, V, W = _build_strassen_4x4x8()
    
    # Verify initial decomposition
    if not _verify_numpy(U, V, W, n, m, p):
        # Fallback to schoolbook rank 128
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
    
    # Try greedy rank reduction
    U, V, W, removed = _greedy_rank_reduction(U, V, W, n, m, p, deadline)
    
    # Final verification
    if not _verify_numpy(U, V, W, n, m, p):
        # If reduction broke something, return the original rank-98
        U, V, W = _build_strassen_4x4x8()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
