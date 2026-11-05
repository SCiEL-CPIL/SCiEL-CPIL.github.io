"""
Matrix multiplication 6x6x6: rank-minimal exact tensor decomposition.

Strategy:
  - 6x6 = 3x3 grid of 2x2 blocks (9 blocks).
  - Use Strassen's 7 products for each 2x2 block multiplication.
  - Total rank: 3*3*3*7 = 189.
  
  The decomposition satisfies:
    sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,j,k
    
  where n=m=p=6, and W follows the trace convention W[k*n+i].
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START


def _strassen_2x2_terms():
    """
    Return Strassen's 7 products for 2x2 matrix multiplication.
    Each term: (a_coeffs, b_coeffs, c_coeffs) where coeffs are dicts mapping (i,j) -> coeff.
    """
    terms = []

    # M1 = (A11 + A22) * (B11 + B22)
    a1 = {(0, 0): 1, (1, 1): 1}
    b1 = {(0, 0): 1, (1, 1): 1}
    c1 = {(0, 0): 1, (1, 1): 1}
    terms.append((a1, b1, c1))

    # M2 = (A21 + A22) * B11
    a2 = {(1, 0): 1, (1, 1): 1}
    b2 = {(0, 0): 1}
    c2 = {(1, 0): 1, (1, 1): -1}
    terms.append((a2, b2, c2))

    # M3 = A11 * (B12 - B22)
    a3 = {(0, 0): 1}
    b3 = {(0, 1): 1, (1, 1): -1}
    c3 = {(0, 1): 1, (1, 1): 1}
    terms.append((a3, b3, c3))

    # M4 = A22 * (B21 - B11)
    a4 = {(1, 1): 1}
    b4 = {(1, 0): 1, (0, 0): -1}
    c4 = {(0, 0): 1, (1, 0): 1}
    terms.append((a4, b4, c4))

    # M5 = (A11 + A12) * B22
    a5 = {(0, 0): 1, (0, 1): 1}
    b5 = {(1, 1): 1}
    c5 = {(0, 0): -1, (0, 1): 1}
    terms.append((a5, b5, c5))

    # M6 = (A21 - A11) * (B11 + B12)
    a6 = {(1, 0): 1, (0, 0): -1}
    b6 = {(0, 0): 1, (0, 1): 1}
    c6 = {(1, 1): 1}
    terms.append((a6, b6, c6))

    # M7 = (A12 - A22) * (B21 + B22)
    a7 = {(0, 1): 1, (1, 1): -1}
    b7 = {(1, 0): 1, (1, 1): 1}
    c7 = {(0, 0): 1}
    terms.append((a7, b7, c7))

    return terms


def _build_strassen_6x6():
    """
    Build the 6x6 matrix multiplication tensor decomposition using
    3x3 grid of 2x2 blocks with Strassen's 7 products per block.
    
    Total rank = 3 * 3 * 3 * 7 = 189.
    """
    n = 6
    rank = 189
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))

    strassen_terms = _strassen_2x2_terms()

    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                    # A block is at position (i_block, k_block), size 2x2
                    # B block is at position (k_block, j_block), size 2x2
                    # C block is at position (i_block, j_block), size 2x2

                    for (ai, aj), a_val in a_coeffs.items():
                        # A index: row = i_block*2 + ai, col = k_block*2 + aj
                        u_idx = (i_block * 2 + ai) * n + (k_block * 2 + aj)
                        U[u_idx, col] += a_val

                    for (bi, bj), b_val in b_coeffs.items():
                        # B index: row = k_block*2 + bi, col = j_block*2 + bj
                        v_idx = (k_block * 2 + bi) * n + (j_block * 2 + bj)
                        V[v_idx, col] += b_val

                    for (ci, cj), c_val in c_coeffs.items():
                        # C index: row = i_block*2 + ci, col = j_block*2 + cj
                        # W convention: W[k*n+i] where k is col, i is row
                        w_idx = (j_block * 2 + cj) * n + (i_block * 2 + ci)
                        W[w_idx, col] += c_val

                    col += 1

    return {"U": U, "V": V, "W": W}


def _build_schoolbook_6x6():
    """
    Build the 6x6 matrix multiplication tensor decomposition using schoolbook.
    Total rank = 6 * 6 * 6 = 216.
    """
    n = 6
    rank = n * n * n
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))

    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + k, col] = 1
                V[k * n + j, col] = 1
                W[j * n + i, col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(decomp, n=6, tol=1e-10):
    """
    Verify that the decomposition correctly decomposes the matrix multiplication tensor.
    
    Check: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,j,k
    """
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]

    m = p = n
    rank = U.shape[1]

    # Check all (i,j,k) triples for thoroughness
    for i in range(n):
        for j in range(n):
            for k in range(n):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]

                if abs(val - 1.0) > tol:
                    return False

    return True


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix
    by a 6x6 matrix.
    
    Uses Strassen block decomposition (rank 189) with verification.
    Falls back to schoolbook (rank 216) if verification fails.
    """
    t0 = time.time()
    DEADLINE = 130.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6

    # Build Strassen 2x2 block approach (rank 189)
    if (time.time() - t0) < DEADLINE:
        decomp = _build_strassen_6x6()
        if _verify_decomposition(decomp):
            return decomp

    # Final fallback: schoolbook (rank 216)
    decomp = _build_schoolbook_6x6()
    return decomp

# EVOLVE-BLOCK-END
