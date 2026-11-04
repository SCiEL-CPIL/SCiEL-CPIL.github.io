"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _schoolbook_2x2():
    """
    Return the rank-4 schoolbook decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 4) - for A[i,j], row index = i*2+j
      V: (4, 4) - for B[j,k], row index = j*2+k
      W: (4, 4) - for C[i,k], row index = k*2+i  (trace convention)
    """
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((4, 4), dtype=int)
    W = np.zeros((4, 4), dtype=int)
    
    for j in range(2):
        t = j
        for i in range(2):
            U[i*2 + j, t] = 1
        for k in range(2):
            V[j*2 + k, t] = 1
        for i in range(2):
            for k in range(2):
                W[k*2 + i, t] = 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: 2x2 block structure with mixed inner decompositions.
    # A is 4x4: 2x2 blocks of size 2x2. A[ai][bj], ai,bj in {0,1}
    # B is 4x6: 2x3 blocks of size 2x2. B[bj][ck], bj in {0,1}, ck in {0,1,2}
    # C is 4x6: 2x3 blocks of size 2x2. C[ai][ck], ai in {0,1}, ck in {0,1,2}
    #
    # C[ai][ck] = sum_bj A[ai][bj] * B[bj][ck]
    #
    # Outer structure: 2x2 * 2x3 block multiplication with 6 outer terms:
    #   For each (ai, bj, ck) combination, we need a 2x2*2x2 block product.
    #   But we can group by (bj, ck) pairs: 2 * 3 = 6 block positions in B.
    #   For each bj in {0,1} and ck in {0,1,2}:
    #     C[ai][ck] += A[ai][bj] * B[bj][ck] for each ai
    #
    # Actually, the outer decomposition is:
    #   For each bj in {0,1} and ck in {0,1,2}:
    #     The product A[ai][bj] * B[bj][ck] contributes to C[ai][ck] for all ai.
    #     This is a 2x2 * 2x2 -> 2x2 block product for each (ai, ck) pair.
    #
    # Let's think of it differently. The full tensor is:
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #
    # With i = 2*ai + ai2, j = 2*bj + bj2, k = 2*ck + ck2:
    #   C[2*ai+ai2, 2*ck+ck2] = sum_{bj,bj2} A[2*ai+ai2, 2*bj+bj2] B[2*bj+bj2, 2*ck+ck2]
    #
    # Group by bj:
    #   = sum_bj [sum_{bj2} A[2*ai+ai2, 2*bj+bj2] B[2*bj+bj2, 2*ck+ck2]]
    #
    # Each bracket is a 2x2 * 2x2 -> 2x2 product (over ai2, bj2, ck2).
    #
    # So we have 2 (bj values) * 3 (ck values) = 6 block products, each of size 2x2*2x2.
    #
    # For ck in {0,1}: use Strassen (rank 7) -> 4 * 7 = 28
    # For ck = 2: use schoolbook (rank 4) -> 2 * 4 = 8
    # Total rank: 36
    
    sU, sV, sW = _strassen_2x2()
    bU, bV, bW = _schoolbook_2x2()
    
    # 6 outer terms: (bj, ck) for bj in {0,1}, ck in {0,1,2}
    # For ck in {0,1}: 4 terms with Strassen (7 each) = 28
    # For ck = 2: 2 terms with schoolbook (4 each) = 8
    # Total: 36
    
    total_rank = 4 * 7 + 2 * 4  # 36
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    col = 0
    for bj in range(2):
        for ck in range(3):
            if ck < 2:
                # Use Strassen for first 2 columns
                inner_U, inner_V, inner_W = sU, sV, sW
                inner_rank = 7
            else:
                # Use schoolbook for last column
                inner_U, inner_V, inner_W = bU, bV, bW
                inner_rank = 4
            
            for t in range(inner_rank):
                c = col + t
                
                # U part: A[ai][bj] block, for each ai in {0,1}
                for ai in range(2):
                    for ai2 in range(2):
                        for bj2 in range(2):
                            coeff = inner_U[ai2*2 + bj2, t]
                            if coeff != 0:
                                u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                                U[u_row, c] += coeff
                
                # V part: B[bj][ck] block
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff = inner_V[bj2*2 + ck2, t]
                        if coeff != 0:
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, c] += coeff
                
                # W part: C[ai][ck] block, for each ai in {0,1}
                for ai in range(2):
                    for ai2 in range(2):
                        for ck2 in range(2):
                            coeff = inner_W[ck2*2 + ai2, t]
                            if coeff != 0:
                                # Trace convention: W[k*n+i] with n=4
                                # k = 2*ck + ck2, i = 2*ai + ai2
                                w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                                W[w_row, c] += coeff
            
            col += inner_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
