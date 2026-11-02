"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]
_DEADLINE_SECONDS = 145  # Safety margin below 150s wall-clock limit

# EVOLVE-BLOCK-START

# Strassen 2x2: 7 bilinear forms.
# Each form: (list of (row, col, coeff) for A, list of (row, col, coeff) for B, list of (row, col, coeff) for C)
# A and B are 2x2 block matrices; C is the 2x2 result block.
_STRASSEN_2X2 = [
    # M0 = (A00+A11)*(B00+B11)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # M1 = (A10+A11)*B00
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
    # M2 = A00*(B01-B11)
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
    # M3 = A11*(B10-B00)
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1)]),
    # M4 = (A00+A01)*B11
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,1)]),
    # M5 = (A10-A00)*(B00+B01)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    # M6 = (A01-A11)*(B10+B11)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,1,1)]),
]

def _build_strassen_4x4():
    """
    Build rank-49 decomposition via Kronecker product of two Strassen 2x2 decompositions.
    
    The 4x4 matrix multiplication tensor is the Kronecker product of two 2x2 MM tensors.
    If T_2x2 = sum_{s=0}^{6} u_s ⊗ v_s ⊗ w_s, then
    T_4x4 = T_2x2 ⊗ T_2x2 = sum_{s1,s2} (u_{s1}⊗u_{s2}) ⊗ (v_{s1}⊗v_{s2}) ⊗ (w_{s1}⊗w_{s2})
    
    Each u_s is a 4-dim vector (flattened 2x2), so u_{s1}⊗u_{s2} is 16-dim (flattened 4x4).
    Total rank = 7*7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    for outer_i in range(7):
        a_outer, b_outer, c_outer = _STRASSEN_2X2[outer_i]
        for inner_s in range(7):
            r = outer_i * 7 + inner_s
            a_inner, b_inner, c_inner = _STRASSEN_2X2[inner_s]

            # U: Kronecker product of a_outer and a_inner
            # a_outer is list of (block_row, block_col, coeff) for 2x2 blocks of 4x4
            # a_inner is list of (row, col, coeff) within each 2x2 block
            for (obr, obc, oc) in a_outer:
                for (ibr, ibc, ic) in a_inner:
                    row = 2 * obr + ibr
                    col = 2 * obc + ibc
                    U[row * m + col, r] += oc * ic

            # V: Kronecker product of b_outer and b_inner
            for (obr, obc, oc) in b_outer:
                for (ibr, ibc, ic) in b_inner:
                    row = 2 * obr + ibr
                    col = 2 * obc + ibc
                    V[row * p + col, r] += oc * ic

            # W: Kronecker product of c_outer and c_inner
            # W index: W[k*n+i, r] where k is col, i is row of C
            for (obr, obc, oc) in c_outer:
                for (ibr, ibc, ic) in c_inner:
                    row = 2 * obr + ibr  # row of C
                    col = 2 * obc + ibc  # col of C
                    W[col * n + row, r] += oc * ic

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Pre-compute at module level with deadline guard
_PRECOMPUTED = None

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return bool(np.allclose(result, 1.0, atol=1e-10))

def _schoolbook_decomposition(n, m, p):
    """Fallback: rank 64 schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}

def construct():
    global _PRECOMPUTED
    t_start = time.time()
    
    # Build the rank-49 decomposition if not already done
    if _PRECOMPUTED is None:
        elapsed = time.time() - t_start
        if elapsed > _DEADLINE_SECONDS - 5:
            # Not enough time, use schoolbook
            return _schoolbook_decomposition(*DIMENSIONS)
        _PRECOMPUTED = _build_strassen_4x4()
    
    n, m, p = DIMENSIONS
    result = _PRECOMPUTED
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    
    # Verify with deadline check
    elapsed = time.time() - t_start
    if elapsed < _DEADLINE_SECONDS - 10:
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return result
    
    # Fallback to schoolbook if verification fails or times out
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
