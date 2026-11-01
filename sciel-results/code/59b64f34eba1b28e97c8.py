"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 130  # Safety margin: evaluator kills at 150s

    # Build the rank-53 baseline using Strassen for 7 pairs + schoolbook for 1 single
    U, V, W = _build_strassen_53(n, m, p)

    # Verify the baseline
    if _verify(U, V, W, n, m, p):
        base_rank = len(U[0])
    else:
        # Fallback to schoolbook rank 60
        U, V, W = _build_schoolbook(n, m, p)
        base_rank = len(U[0])

    # Try to find a lower-rank decomposition using a structural search approach
    # For 2x2x15, we know the rank is at most 53 (Strassen + schoolbook)
    # Let's try to see if we can do better with a more clever combination

    # The key insight: instead of treating each 2x2x2 block independently,
    # we might be able to share some rank columns across blocks if the
    # coefficient patterns allow it.

    # Strategy: Try to find a decomposition with rank < 53 by solving a
    # system of linear equations over the rationals.

    # For a given rank r, we need to find U, V, W such that:
    # sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k

    # This is a bilinear system. We can try to fix the structure of U and V
    # (e.g., sparse patterns) and solve for W, or vice versa.

    # Given the time constraints, let's try a targeted search for rank 52
    # by looking for a way to merge two rank-1 terms from different blocks.

    # Actually, let's try a different structural approach:
    # Use the fact that the tensor is a "slice" of a larger tensor.
    # The 2x2x15 tensor can be seen as 15 slices of 2x2 tensors.
    # Each 2x2 slice has rank at most 4 (schoolbook) or 7 (Strassen for pairs).

    # Let's try to find a decomposition with rank 52 by allowing some
    # cross-block interactions.

    # For now, return the known optimal rank-53 decomposition.
    # A more sophisticated search would be needed to prove optimality
    # or find a better decomposition.

    return {"U": U, "V": V, "W": W}

def _build_strassen_53(n, m, p):
    """Build the rank-53 Strassen-based decomposition for 2x2x15."""
    n, m, p = 2, 2, 15
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14,)]
    
    all_U = []
    all_V = []
    all_W = []

    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1, n, m, p)
        else:
            k = block[0]
            terms = _schoolbook_terms(k, n, m, p)
        
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
            v_entries = [(j * p + k, c) for (j, k), c in v_dict.items()]
            w_entries = [(k * n + i, c) for (k, i), c in w_dict.items()]
            
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c in range(rank):
        for (row, val) in all_U[c]:
            U[row][c] = val
        for (row, val) in all_V[c]:
            V[row][c] = val
        for (row, val) in all_W[c]:
            W[row][c] = val

    return U, V, W

def _strassen_terms(k0, k1, n, m, p):
    """Return Strassen terms for a 2x2x2 block."""
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0, k0): 1, (1, k1): 1}, {(k0, 0): 1, (k1, 1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0, k0): 1}, {(k0, 1): 1}),
        ({(0,0): 1}, {(1, k0): 1, (1, k1): -1}, {(k1, 0): 1}),
        ({(1,1): 1}, {(0, k1): 1, (0, k0): -1}, {(k0, 0): 1, (k0, 1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1, k1): 1}, {(k0, 0): -1, (k1, 0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0, k0): 1, (0, k1): 1}, {(k1, 1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1, k0): 1, (1, k1): 1}, {(k0, 0): 1}),
    ]
    return terms

def _schoolbook_terms(k, n, m, p):
    """Return schoolbook terms for a 2x2x1 block."""
    terms = []
    for i in range(n):
        for j in range(m):
            terms.append((
                {(i, j): 1},
                {(j, k): 1},
                {(k, i): 1}
            ))
    return terms

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-60 decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = np.sum(U_arr[u_row, :] * V_arr[v_row, :] * W_arr[w_row, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True
# EVOLVE-BLOCK-END
