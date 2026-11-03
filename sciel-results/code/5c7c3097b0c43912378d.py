"""Matrix multiplication 4x4x6 via block-Kronecker Strassen decomposition."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np


def _strassen_2x2_factors():
    """
    Return (U, V, W) for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U shape (4, 7): rows indexed by i*2+j
    V shape (4, 7): rows indexed by j*2+k
    W shape (4, 7): rows indexed by k*2+i
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen products M1..M7:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # Each entry: (U_terms, V_terms, W_terms)
    # U_terms: list of (i, j, coeff)
    # V_terms: list of (j, k, coeff)
    # W_terms: list of (i, k, coeff) -> stored at W[k*2+i]

    prods = [
        # M1 = (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
        ([(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)]),
        # M2 = (A10+A11)B00 -> contributes to C10(+1), C11(-1)
        ([(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)],
         [(1,0,1.0),(1,1,-1.0)]),
        # M3 = A00(B01-B11) -> contributes to C01(+1), C11(+1)
        ([(0,0,1.0)],
         [(0,1,1.0),(1,1,-1.0)],
         [(0,1,1.0),(1,1,1.0)]),
        # M4 = A11(B10-B00) -> contributes to C00(+1), C10(+1)
        ([(1,1,1.0)],
         [(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(1,0,1.0)]),
        # M5 = (A00+A01)B11 -> contributes to C00(-1), C01(+1)
        ([(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)],
         [(0,0,-1.0),(0,1,1.0)]),
        # M6 = (A10-A00)(B00+B01) -> contributes to C11(+1)
        ([(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)]),
        # M7 = (A01-A11)(B10+B11) -> contributes to C00(+1)
        ([(0,1,1.0),(1,1,-1.0)],
         [(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)]),
    ]

    for s, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][s] += c
        for (j, k, c) in Vb:
            V[j * p + k][s] += c
        for (i, k, c) in Wb:
            W[k * n + i][s] += c

    return U, V, W


def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Block decomposition with 2x2 blocks:
    # A (4x4) -> 2x2 grid of 2x2 blocks
    # B (4x6) -> 2x3 grid of 2x2 blocks
    # C (4x6) -> 2x3 grid of 2x2 blocks
    #
    # Block indices: A[I,J], B[J,K], C[I,K] with I in {0,1}, J in {0,1}, K in {0,1,2}
    # Global indices:
    #   A[i,j]: i = 2*I+ii, j = 2*J+jj
    #   B[j,k]: j = 2*J+jj, k = 2*K+kk
    #   C[i,k]: i = 2*I+ii, k = 2*K+kk
    #   W row: k*n+i = (2*K+kk)*4 + (2*I+ii)
    #
    # Strategy: Use Strassen (rank 7) for the 2x2 sub-problem (K=0,1),
    # and schoolbook (rank 7 per product) for K=2.
    #
    # For K=0,1 (Strassen part): 7 block-products
    # For K=2 (schoolbook part): 2 block-products (J=0,1)
    # Total: 9 block-products, each using rank-7 2x2x2 Strassen
    # Total rank: 9 * 7 = 63

    inner_rank = 7
    total_prods = 9  # 7 Strassen + 2 schoolbook
    rank = total_prods * inner_rank

    U = [[0.0] * rank for _ in range(n * m)]   # 16 rows
    V = [[0.0] * rank for _ in range(m * p)]   # 24 rows
    W = [[0.0] * rank for _ in range(p * n)]   # 24 rows

    # Get the rank-7 Strassen factors for 2x2x2
    U_s, V_s, W_s = _strassen_2x2_factors()

    def aidx(I, ii, J, jj):
        """Global A index: i*m+j where i=2*I+ii, j=2*J+jj"""
        return (2 * I + ii) * m + (2 * J + jj)

    def bidx(J, jj, K, kk):
        """Global B index: j*p+k where j=2*J+jj, k=2*K+kk"""
        return (2 * J + jj) * p + (2 * K + kk)

    def widx(K, kk, I, ii):
        """Global W index: k*n+i where k=2*K+kk, i=2*I+ii"""
        return (2 * K + kk) * n + (2 * I + ii)

    def apply_block_prod(prod_idx, I, J, K, U_s, V_s, W_s):
        """Apply one 2x2x2 Strassen product at block position (I,J,K)."""
        base = prod_idx * inner_rank
        for s in range(inner_rank):
            for ii in range(2):
                for jj in range(2):
                    val = U_s[ii * 2 + jj][s]
                    if val != 0.0:
                        U[aidx(I, ii, J, jj)][base + s] += val
            for jj in range(2):
                for kk in range(2):
                    val = V_s[jj * 2 + kk][s]
                    if val != 0.0:
                        V[bidx(J, jj, K, kk)][base + s] += val
            for ii in range(2):
                for kk in range(2):
                    val = W_s[kk * 2 + ii][s]
                    if val != 0.0:
                        W[widx(K, kk, I, ii)][base + s] += val

    # --- Strassen part: 7 products for K=0,1 ---
    # The 7 Strassen products operate on the 2x2 block sub-matrix (I in {0,1}, K in {0,1}).
    # Each Strassen product M_s is: (linear combo of A-blocks) x (linear combo of B-blocks) -> (linear combo of C-blocks)
    # But we need to map each Strassen product to specific (I,J,K) block positions.
    #
    # Actually, the block-Kronecker Strassen works as follows:
    # The 7 Strassen products each involve specific A-blocks and B-blocks, producing specific C-blocks.
    # Each product is a 2x2x2 multiplication (the inner block multiplication).
    #
    # Let me define the 7 Strassen block-products explicitly:
    # M1 = (A00+A11) x (B00+B10) -> C00, C10 (wait, this isn't right for non-square)
    
    # For the 2x2 x 2x3 case, standard Strassen doesn't directly apply.
    # Let me use a different approach: just do all 12 block-products with rank-7 each.
    # 12 * 7 = 84. This is simpler and guaranteed correct.

    # Reset
    rank = 12 * inner_rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    prod_idx = 0
    for I in range(2):
        for J in range(2):
            for K in range(3):
                apply_block_prod(prod_idx, I, J, K, U_s, V_s, W_s)
                prod_idx += 1

    # Verify
    err = _verify(U, V, W, n, m, p, rank)
    if err > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
