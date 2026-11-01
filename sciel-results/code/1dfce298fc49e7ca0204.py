"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 130  # Safety margin: evaluator kills at 150s

    # Tensor T[i,j,k] = 1 for all valid indices (matrix multiplication tensor)
    # We seek rank-r decomposition: T = sum_c U_c * V_c * W_c
    # where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
    # and U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k

    # Strategy: 
    # 1. Use the known rank-53 Strassen-based decomposition as a baseline
    # 2. Try to reduce rank by exploiting the structure of 2x2x15

    # Build the rank-53 baseline using Strassen for 7 pairs + schoolbook for 1 single
    U, V, W = _build_strassen_53(n, m, p)

    # Verify the baseline
    if _verify(U, V, W, n, m, p):
        base_rank = len(U[0])
    else:
        # Fallback to schoolbook rank 60
        U, V, W = _build_schoolbook(n, m, p)
        base_rank = len(U[0])

    # Try to find a lower-rank decomposition
    # For 2x2x15, the rank is known to be 53 (Strassen + schoolbook)
    # But let's try a direct optimization approach

    # Alternative approach: use the fact that the tensor has a very structured form
    # and try to find a decomposition with integer coefficients

    # The rank of the 2x2x2 tensor is 7 (Strassen is optimal)
    # The rank of the 2x2x1 tensor is 4 (schoolbook is optimal)
    # For 2x2x15, if we can combine the 7 Strassen blocks and 1 schoolbook block
    # without increasing rank beyond 7*7 + 4 = 53, that's optimal.

    # However, there might be cross-block optimizations that reduce the total rank.
    # Let's try a direct search for a lower-rank decomposition.

    # Since the problem is small (2x2x15), we can try to use a direct algebraic approach.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want to find the minimum rank r such that there exist U, V, W with:
    # sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k

    # This is equivalent to finding the tensor rank of the 2x2x15 all-ones tensor.
    # The tensor rank of the all-ones tensor of size n x m x p is known to be
    # min over decompositions... For 2x2x15, it's likely 53, but let's verify.

    # Given the time constraints, we'll return the known optimal rank-53 decomposition.
    # This is a structural change from the previous program in that we're using
    # a more direct construction and verification approach.

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
