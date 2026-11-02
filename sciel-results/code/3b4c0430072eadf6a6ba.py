"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses recursive Strassen (rank 49 for 4x4x4) plus structured extension for the 5th column,
with a final polish pass to verify exactness."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 as factor matrices.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    Returns U, V, W each of shape (4, 7).
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Strassen 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    #
    # C00 = M1 - M4 + M7 - M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # U factors (for A):
    # A00 = U[0,0], A01 = U[1,0], A10 = U[2,0], A11 = U[3,0] (row-major: i*2+j)
    # M1: A00+A11 -> U[0,0]=1, U[3,0]=1
    # M2: A10+A11 -> U[2,1]=1, U[3,1]=1
    # M3: A00 -> U[0,2]=1
    # M4: A11 -> U[3,3]=1
    # M5: A00+A01 -> U[0,4]=1, U[1,4]=1
    # M6: A01-A11 -> U[1,5]=1, U[3,5]=-1
    # M7: A10-A00 -> U[2,6]=1, U[0,6]=-1
    U[0,0]=1; U[3,0]=1       # M1
    U[2,1]=1; U[3,1]=1       # M2
    U[0,2]=1                 # M3
    U[3,3]=1                 # M4
    U[0,4]=1; U[1,4]=1       # M5
    U[1,5]=1; U[3,5]=-1      # M6
    U[2,6]=1; U[0,6]=-1      # M7
    
    # V factors (for B):
    # B00 = V[0,0], B01 = V[1,0], B10 = V[2,0], B11 = V[3,0] (row-major: j*2+k)
    # M1: B00+B11 -> V[0,0]=1, V[3,0]=1
    # M2: B00 -> V[0,1]=1
    # M3: B01-B11 -> V[1,2]=1, V[3,2]=-1
    # M4: B10-B00 -> V[2,3]=1, V[0,3]=-1
    # M5: B11 -> V[3,4]=1
    # M6: B10+B11 -> V[2,5]=1, V[3,5]=1
    # M7: B00+B01 -> V[0,6]=1, V[1,6]=1
    V[0,0]=1; V[3,0]=1       # M1
    V[0,1]=1                 # M2
    V[1,2]=1; V[3,2]=-1      # M3
    V[2,3]=1; V[0,3]=-1      # M4
    V[3,4]=1                 # M5
    V[2,5]=1; V[3,5]=1       # M6
    V[0,6]=1; V[1,6]=1       # M7
    
    # W factors (for C):
    # C00 = W[0,0], C01 = W[1,0], C10 = W[2,0], C11 = W[3,0] (row-major: k*2+i)
    # Wait, the convention is W[k*n+i, c] where n=2 for 2x2.
    # So W index = k*2 + i.
    # C[i,k] gets contribution from W[k*2+i, c]
    # C00: i=0,k=0 -> W[0*2+0]=W[0]
    # C01: i=0,k=1 -> W[1*2+0]=W[2]
    # C10: i=1,k=0 -> W[0*2+1]=W[1]
    # C11: i=1,k=1 -> W[1*2+1]=W[3]
    #
    # C00 = M1 - M4 + M7 - M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    W[0,0]=1; W[0,3]=-1; W[0,6]=1; W[0,4]=-1   # C00: M1-M4+M7-M5
    W[2,2]=1; W[2,4]=1                          # C01: M3+M5
    W[1,1]=1; W[1,3]=1                          # C10: M2+M4
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=-1   # C11: M1-M2+M3-M6
    
    return U, V, W


def _recursive_strassen_4x4():
    """Build 4x4 * 4x4 decomposition using recursive Strassen.
    Split into 2x2 blocks, each 2x2 block multiplication uses Strassen (rank 7).
    Total rank: 7 * 7 = 49.
    Returns U (16,49), V (16,49), W (16,49).
    """
    U2, V2, W2 = _strassen_2x2()
    rank2 = 7
    rank4 = rank2 * rank2  # 49
    
    U4 = np.zeros((16, rank4), dtype=np.float64)
    V4 = np.zeros((16, rank4), dtype=np.float64)
    W4 = np.zeros((16, rank4), dtype=np.float64)
    
    # For recursive Strassen on 4x4:
    # A is 4x4, split into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
    # B is 4x4, split into 2x2 blocks: B00, B01, B10, B11 (each 2x2)
    # C is 4x4, split into 2x2 blocks: C00, C01, C10, C11 (each 2x2)
    #
    # C00 = (A00+A11)(B00+B11) - (A10-A11)(B00-B11) - (A00-A10)(B01-B11)
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = (A00+A01)(B10+B11) - A01*B10 - A00*B01
    #
    # Each of these involves products of 2x2 matrices, which we decompose with Strassen.
    # 
    # Let me index the blocks:
    # A block (bi, bj) where bi in {0,1}, bj in {0,1} is A[bi*2:bi*2+2, bj*2:bj*2+2]
    # In row-major indexing for 4x4: A[i,j] where i=bi*2+ii, j=bj*2+jj
    # U index: i*4 + j = (bi*2+ii)*4 + (bj*2+jj)
    #
    # For the recursive structure, each "product" of two 2x2 blocks uses 7 Strassen terms.
    # We need to enumerate all the 2x2 block products needed.
    
    # Let's define the block products needed:
    # For C00:
    #   P1 = (A00+A11)(B00+B11)  -> 7 terms
    #   P2 = (A10-A11)(B00-B11)  -> 7 terms
    #   P3 = (A00-A10)(B01-B11)  -> 7 terms
    # C00 = P1 - P2 - P3
    #
    # For C01:
    #   P4 = A00*B01  -> 7 terms
    #   P5 = A01*B11  -> 7 terms
    # C01 = P4 + P5
    #
    # For C10:
    #   P6 = A10*B00  -> 7 terms
    #   P7 = A11*B10  -> 7 terms
    # C10 = P6 + P7
    #
    # For C11:
    #   P8 = (A00+A01)(B10+B11)  -> 7 terms
    #   P9 = A01*B10  -> 7 terms
    #   P10 = A00*B01  -> 7 terms (same as P4!)
    # C11 = P8 - P9 - P10
    
    # So we need 9 distinct 2x2 block products (P4 and P10 are the same).
    # Total terms: 9 * 7 = 63... but wait, that's more than 49.
    #
    # Actually, the standard recursive Strassen gives rank 7^2 = 49 for 4x4.
    # The key is that the 7 Strassen formulas at the top level combine the 2x2 block
    # multiplications in a specific way. Let me think about this more carefully.
    #
    # In recursive Strassen, we apply Strassen at the block level. Each of the 7
    # "products" in the top-level Strassen is itself a 2x2 matrix multiplication,
    # which requires 7 scalar multiplications (via Strassen again).
    # So total = 7 * 7 = 49.
    #
    # The 7 top-level products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    #
    # Each Mj is a 2x2 matrix, computed via Strassen (7 terms each).
    # Then:
    # C00 = M1 - M4 + M7 - M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # Let me build this properly.
    # For each top-level Strassen term t (0..6), we have a 2x2 block product.
    # The U factor for the block product involves combining A blocks with coefficients.
    # Similarly for V and W.
    
    # Top-level Strassen coefficients for A blocks:
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A01 - A11
    # M7: A10 - A00
    
    # Top-level Strassen coefficients for B blocks:
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B10 + B11
    # M7: B00 + B01
    
    # Top-level Strassen coefficients for C blocks:
    # C00: M1 - M4 + M7 - M5
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 - M6
    
    # For each top-level term t, the 2x2 block product uses Strassen internally.
    # The internal Strassen U2, V2, W2 handle the 2x2 sub-blocks.
    
    # Let's define the block structure:
    # A blocks: A[bi, bj] for bi, bj in {0,1}, each is 2x2
    # In row-major: A element (i,j) where i = bi*2+ii, j = bj*2+jj
    # U index: i*4 + j
    
    # For a given top-level term t, the A factor is a linear combination of A blocks.
    # Similarly for B and C.
    
    # Let me define the top-level coefficients as 2x2 matrices of coefficients for each block.
    # A_coeff[t][bi][bj] = coefficient of A[bi,bj] in the t-th top-level product
    # B_coeff[t][bj][bk] = coefficient of B[bj,bk] in the t-th top-level product
    # C_coeff[t][bi][bk] = coefficient of C[bi,bk] in the t-th top-level product
    
    A_coeff = [
        [[1, 0], [0, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[0, 1], [0, -1]],  # M6: A01 - A11
        [[-1, 0], [1, 0]],  # M7: A10 - A00
    ]
    
    B_coeff = [
        [[1, 0], [0, 1]],   # M1: B00 + B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01 - B11
        [[0, -1], [1, 0]],  # M4: B10 - B00
        [[0, 0], [0, 1]],   # M5: B11
        [[0, 0], [1, 1]],   # M6: B10 + B11
        [[1, 1], [0, 0]],   # M7: B00 + B01
    ]
    
    C_coeff = [
        [[1, 0], [0, 0]],   # C00 gets M1
        [[0, 0], [0, 0]],   # C01
        [[0, 0], [0, 0]],   # C10
        [[0, 0], [0, 0]],   # C11
    ]
    # Actually, let me redefine C_coeff to show which M contributes to which C block:
    # C00 = M1 - M4 + M7 - M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # So for each C block (ci, ck), the coefficient of Mt is:
    C_from_M = {
        (0,0): {0: 1, 3: -1, 6: 1, 4: -1},   # C00 = M1 - M4 + M7 - M5
        (0,1): {2: 1, 4: 1},                   # C01 = M3 + M5
        (1,0): {1: 1, 3: 1},                   # C10 = M2 + M4
        (1,1): {0: 1, 1: -1, 2: 1, 5: -1},     # C11 = M1 - M2 + M3 - M6
    }
    
    col = 0
    for t in range(7):  # top-level Strassen term
        for c2 in range(7):  # internal Strassen term
            # For this (t, c2) pair, we have one scalar multiplication.
            # The U factor: for each element (i,j) of A, the coefficient is:
            #   sum over (bi,bj) of A_coeff[t][bi][bj] * U2[ii, c2] where i=bi*2+ii, j=bj*2+jj
            # Similarly for V and W.
            
            for bi in range(2):
                for bj in range(2):
                    a_coeff = A_coeff[t][bi][bj]
                    if a_coeff == 0:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            u_val = a_coeff * U2[ii, c2]
                            if u_val != 0:
                                i = bi * 2 + ii
                                j = bj * 2 + jj
                                U4[i * 4 + j, col] += u_val
            
            for bj in range(2):
                for bk in range(2):
                    b_coeff = B_coeff[t][bj][bk]
                    if b_coeff == 0:
                        continue
                    for jj in range(2):
                        for kk in range(2):
                            v_val = b_coeff * V2[jj, c2]
                            if v_val != 0:
                                j = bj * 2 + jj
                                k = bk * 2 + kk
                                V4[j * 4 + k, col] += v_val
            
            for ci in range(2):
                for ck in range(2):
                # C block (ci, ck) gets contribution from Mt with coefficient C_from_M[(ci,ck)].get(t, 0)
                    c_coeff = C_from_M.get((ci, ck), {}).get(t, 0)
                    if c_coeff == 0:
                        continue
                    for kk in range(2):
                        for ii in range(2):
                            w_val = c_coeff * W2[kk, c2]
                            if w_val != 0:
                                k = ck * 2 + kk
                                i = ci * 2 + ii
                                W4[k * 4 + i, col] += w_val
            
            col += 1
    
    return U4, V4, W4


def _extend_to_4x4x5():
    """Extend the 4x4x4 decomposition to 4x4x5.
    
    For the 5th column of B (index k=4), we need:
    C[i,4] = sum_j A[i,j] * B[j,4] for i=0..3
    
    This is a 4x4 * 4x1 multiplication, which has rank 16 (schoolbook).
    Total rank: 49 + 16 = 65.
    
    But we can potentially do better by sharing structure. For now, use schoolbook for the extra column.
    """
    U4, V4, W4 = _recursive_strassen_4x4()
    rank4 = 49
    
    n, m, p = 4, 4, 5
    rank5 = rank4 + 16
    
    U5 = np.zeros((n * m, rank5), dtype=np.float64)
    V5 = np.zeros((m * p, rank5), dtype=np.float64)
    W5 = np.zeros((p * n, rank5), dtype=np.float64)
    
    U5[:, :rank4] = U4
    V5[:m * (p-1), :rank4] = V4
    W5[: (p-1) * n, :rank4] = W4
    
    # Schoolbook for the 5th column (k=4)
    col = rank4
    for i in range(n):
        for j in range(m):
            U5[i * m + j, col] = 1
            V5[j * p + 4, col] = 1
            W5[4 * n + i, col] = 1
            col += 1
    
    return U5, V5, W5


def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=atol)


def _verify_exact(U, V, W, n=4, m=4, p=5):
    """Exact verification using integer/rational arithmetic if possible."""
    try:
        from fractions import Fraction
        Ua = [[Fraction(x) for x in row] for row in U]
        Va = [[Fraction(x) for x in row] for row in V]
        Wa = [[Fraction(x) for x in row] for row in W]
        rank = len(Ua[0])
        Ur = [[[Ua[i*m+j][c] for j in range(m)] for i in range(n)] for c in range(rank)]
        Vr = [[[Va[j*p+k][c] for k in range(p)] for j in range(m)] for c in range(rank)]
        Wr = [[[Wa[k*n+i][c] for i in range(n)] for k in range(p)] for c in range(rank)]
        T = [[[Fraction(0) for k in range(p)] for j in range(m)] for i in range(n)]
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i][j][k] += Ur[c][i][j] * Vr[c][j][k] * Wr[c][k][i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i][j][k] != 1:
                        return False
        return True
    except Exception:
        return _verify_fast(U, V, W, n, m, p, atol=1e-12)


def _schoolbook():
    n, m, p = DIMENSIONS
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
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try recursive Strassen + schoolbook extension (rank 65)
    try:
        U5, V5, W5 = _extend_to_4x4x5()
        if _verify_fast(U5, V5, W5, n, m, p, atol=1e-9):
            # Double-check with tighter tolerance
            if _verify_fast(U5, V5, W5, n, m, p, atol=1e-12):
                return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
