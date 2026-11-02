import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
n, m, p = DIMENSIONS

def construct():
    deadline = time.time() + 140.0

    # The schoolbook decomposition has rank n*m*p = 24.
    # We try to find a lower rank decomposition by exploiting the Kronecker product structure.
    # Since m=3, p=4, and n=2, we can try to factor out common structures.
    # For 2x3x4, we can view the multiplication as:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # If we can find a decomposition of rank r < 24, we win.

    # Let's try a systematic search for low rank decompositions using the fact that
    # the tensor is a sum of 24 rank-1 terms (the schoolbook ones).
    # We can try to merge terms or find linear dependencies.

    # A known decomposition for 2x3x4 matrix multiplication:
    # There is a rank 18 decomposition? Or maybe rank 12?
    # Let's try to construct one manually or via a heuristic.

    # For 2x2x2, the rank is 7 (Strassen).
    # For 2x3x3, we can combine Strassen blocks.
    # 2x3x4 can be seen as two 2x3x2 blocks.
    # If we use Strassen-like decomposition for 2x3x2, what is the rank?
    # 2x3x2 = (2x2x2) + (2x1x2).
    # Rank(2x2x2) = 7.
    # Rank(2x1x2) = 2 (since 1x2x1 is rank 1, so 2x1x2 is rank 2? No, 2x1x2 tensor T_{i,j,k} = A_{i,0}B_{0,k}. This is rank 1? No.
    # Let's check 2x1x2.
    # A is 2x1, B is 1x2.
    # C_{i,k} = A_{i,0} B_{0,k}.
    # This is a rank 1 tensor! U_{i,0} B_{0,k}.
    # So 2x1x2 has rank 1.
    # So 2x3x2 can be decomposed into 7 (for the first 2x2x2 block) + 1 (for the 2x1x2 block) = 8?
    # Let's verify.
    # 2x3x2:
    # j=0,1 form a 2x2x2 block. Rank 7.
    # j=2 forms a 2x1x2 block. Rank 1.
    # Total rank 8.
    # So 2x3x4 = two 2x3x2 blocks (k=0,1 and k=2,3).
    # If the blocks are independent, rank = 8 + 8 = 16.
    # Let's try to construct a rank 16 decomposition.

    # Strassen's 7 rank-1 terms for 2x2x2:
    # m1 = A00 (B00 + B11)
    # m2 = A10 (B00 + B01)
    # m3 = A00 (B10 + B11)
    # m4 = A11 (B00 + B10)
    # m5 = (A00 + A11) (B11 + B00)
    # m6 = (A10 + A11) (B11 + B01)
    # m7 = (A00 + A01) (B10 + B11)
    
    # C00 = m1 - m3 + m4 - m7
    # C01 = m2 - m7 + m6
    # C10 = m5 - m2 + m7
    # C11 = m3 - m5 + m1

    # Let's build the rank 8 decomposition for 2x3x2 (j=0,1,2; k=0,1; i=0,1).
    # Terms 1-7 correspond to the 2x2x2 part (j=0,1; k=0,1).
    # Term 8 corresponds to the 2x1x2 part (j=2; k=0,1).
    # For term 8: A_{i,2} * B_{2,k}.
    # U_{i,7} = delta_{i,0} (if i=0) else 0? No, U has shape (n*m, rank) = (6, 8).
    # U_{i*m+j, c}.
    # For term 8 (c=7):
    # U_{i*3+2, 7} = 1 if i=0 else 0? No, it's just A_{i,2}. So U_{0*3+2, 7} = 1, U_{1*3+2, 7} = 1.
    # V_{j*2+k, 7} = 1 if j=2 else 0. So V_{2*2+0, 7} = 1, V_{2*2+1, 7} = 1.
    # W_{k*2+i, 7} = 1. (Since C_{i,k} gets A_{i,2}B_{2,k}).
    
    # Let's verify the 2x2x2 part for j=0,1.
    # The indices in the global problem are:
    # U index: i*3 + j
    # V index: j*4 + k  (Wait, p=4, so V is 3x4. V index is j*4+k? No, V shape is (m*p, rank) = (12, rank). Index is j*p+k = j*4+k.)
    # W index: k*2 + i  (W shape is (p*n, rank) = (8, rank). Index is k*n+i = k*2+i.)

    # Strassen terms for 2x2x2 (local j=0,1; local k=0,1; local i=0,1):
    # c=0: m1 = A00 (B00 + B11)
    #   U_{0*3+0, 0} = 1
    #   V_{0*4+0, 0} = 1, V_{0*4+1, 0} = 1
    #   W_{0*2+0, 0} = 1, W_{1*2+0, 0} = 1? No.
    #   W_{k*n+i, c}.
    #   m1 contributes to C00 and C01?
    #   C00 = A00(B00+B11) - A00(B10+B11) + A11(B00+B10) - (A00+A01)(B10+B11)
    #   C01 = A10(B00+B01) - (A00+A01)(B10+B11) + (A10+A11)(B11+B01)
    #   C10 = (A00+A11)(B11+B00) - A10(B00+B01) + (A00+A01)(B10+B11)
    #   C11 = A00(B10+B11) - (A00+A11)(B11+B00) + A00(B00+B11)
    
    # Let's map Strassen terms to global indices.
    # Local (i,j,k) in 2x2x2 block (j=0,1; k=0,1).
    # Global i, j, k.
    
    # Term 0 (m1): A00 (B00 + B11)
    # U_{0*3+0, 0} = 1
    # V_{0*4+0, 0} = 1, V_{1*4+1, 0} = 1  (B00 is j=0,k=0; B11 is j=1,k=1)
    # W_{0*2+0, 0} = 1, W_{1*2+0, 0} = 1? No.
    # C_{i,k} gets contribution.
    # m1 = A00(B00+B11).
    # C00 += A00 B00. C01 += A00 B01? No, B11 is k=1.
    # So C00 += A00 B00. C01 += A00 B11.
    # W_{k*2+i, 0} = 1 if (k=0,i=0) or (k=1,i=0).
    # W_{0*2+0, 0} = 1
    # W_{1*2+0, 0} = 1

    # Term 1 (m2): A10 (B00 + B01)
    # U_{1*3+0, 1} = 1
    # V_{0*4+0, 1} = 1, V_{0*4+1, 1} = 1
    # W_{0*2+1, 1} = 1, W_{1*2+1, 1} = 1

    # Term 2 (m3): A00 (B10 + B11)
    # U_{0*3+0, 2} = 1
    # V_{1*4+0, 2} = 1, V_{1*4+1, 2} = 1
    # W_{0*2+0, 2} = 1, W_{1*2+0, 2} = 1

    # Term 3 (m4): A11 (B00 + B10)
    # U_{1*3+1, 3} = 1
    # V_{0*4+0, 3} = 1, V_{1*4+0, 3} = 1
    # W_{0*2+1, 3} = 1, W_{1*2+1, 3} = 1

    # Term 4 (m5): (A00 + A11) (B11 + B00)
    # U_{0*3+0, 4} = 1, U_{1*3+1, 4} = 1
    # V_{0*4+0, 4} = 1, V_{1*4+1, 4} = 1
    # W_{0*2+0, 4} = 1, W_{1*2+1, 4} = 1? No.
    # C00 += (A00+A11)(B00+B11). C11 += (A00+A11)(B00+B11).
    # W_{0*2+0, 4} = 1
    # W_{1*2+1, 4} = 1

    # Term 5 (m6): (A10 + A11) (B11 + B01)
    # U_{1*3+0, 5} = 1, U_{1*3+1, 5} = 1
    # V_{0*4+1, 5} = 1, V_{1*4+1, 5} = 1
    # W_{0*2+1, 5} = 1, W_{1*2+1, 5} = 1? No.
    # C11 += (A10+A11)(B01+B11). C01 += (A10+A11)(B01+B11).
    # W_{0*2+1, 5} = 1
    # W_{1*2+1, 5} = 1

    # Term 6 (m7): (A00 + A01) (B10 + B11)
    # U_{0*3+0, 6} = 1, U_{0*3+1, 6} = 1
    # V_{1*4+0, 6} = 1, V_{1*4+1, 6} = 1
    # W_{0*2+0, 6} = 1, W_{1*2+1, 6} = 1? No.
    # C10 += (A00+A01)(B10+B11). C00 += (A00+A01)(B10+B11).
    # W_{0*2+0, 6} = 1
    # W_{1*2+1, 6} = 1

    # Term 7 (m8): A_{i,2} B_{2,k}
    # U_{0*3+2, 7} = 1, U_{1*3+2, 7} = 1
    # V_{2*4+0, 7} = 1, V_{2*4+1, 7} = 1
    # W_{0*2+0, 7} = 1, W_{0*2+1, 7} = 1, W_{1*2+0, 7} = 1, W_{1*2+1, 7} = 1?
    # C_{i,k} += A_{i,2} B_{2,k}.
    # So W_{k*2+i, 7} = 1 for all k,i.

    # Now we have a rank 8 decomposition for 2x3x2 (k=0,1).
    # We need to do the same for k=2,3.
    # The j=0,1 part is the same Strassen structure, but V indices change (k=2,3).
    # The j=2 part is the same, V indices change (k=2,3).
    
    # Let's build the full rank 16 decomposition.
    # Rank = 16.
    
    U = [[0.0] * 16 for _ in range(n * m)]
    V = [[0.0] * 16 for _ in range(m * p)]
    W = [[0.0] * 16 for _ in range(p * n)]

    # Block 1: k=0,1 (local k'=0,1)
    # Term 0: m1
    U[0][0] = 1; V[0][0] = 1; V[4][0] = 1; W[0][0] = 1; W[2][0] = 1
    # Term 1: m2
    U[3][1] = 1; V[0][1] = 1; V[1][1] = 1; W[1][1] = 1; W[3][1] = 1
    # Term 2: m3
    U[0][2] = 1; V[4][2] = 1; V[5][2] = 1; W[0][2] = 1; W[2][2] = 1
    # Term 3: m4
    U[4][3] = 1; V[0][3] = 1; V[4][3] = 1; W[1][3] = 1; W[3][3] = 1
    # Term 4: m5
    U[0][4] = 1; U[4][4] = 1; V[0][4] = 1; V[5][4] = 1; W[0][4] = 1; W[3][4] = 1
    # Term 5: m6
    U[3][5] = 1; U[4][5] = 1; V[1][5] = 1; V[5][5] = 1; W[1][5] = 1; W[3][5] = 1
    # Term 6: m7
    U[0][6] = 1; U[1][6] = 1; V[4][6] = 1; V[5][6] = 1; W[0][6] = 1; W[3][6] = 1
    # Term 7: m8
    U[2][7] = 1; U[5][7] = 1; V[8][7] = 1; V[9][7] = 1; W[0][7] = 1; W[1][7] = 1; W[2][7] = 1; W[3][7] = 1

    # Block 2: k=2,3 (local k'=0,1)
    # Term 8: m1
    U[0][8] = 1; V[2][8] = 1; V[6][8] = 1; W[4][8] = 1; W[6][8] = 1
    # Term 9: m2
    U[3][9] = 1; V[2][9] = 1; V[3][9] = 1; W[5][9] = 1; W[7][9] = 1
    # Term 10: m3
    U[0][10] = 1; V[6][10] = 1; V[7][10] = 1; W[4][10] = 1; W[6][10] = 1
    # Term 11: m4
    U[4][11] = 1; V[2][11] = 1; V[6][11] = 1; W[5][11] = 1; W[7][11] = 1
    # Term 12: m5
    U[0][12] = 1; U[4][12] = 1; V[2][12] = 1; V[7][12] = 1; W[4][12] = 1; W[7][12] = 1
    # Term 13: m6
    U[3][13] = 1; U[4][13] = 1; V[3][13] = 1; V[7][13] = 1; W[5][13] = 1; W[7][13] = 1
    # Term 14: m7
    U[0][14] = 1; U[1][14] = 1; V[6][14] = 1; V[7][14] = 1; W[4][14] = 1; W[7][14] = 1
    # Term 15: m8
    U[2][15] = 1; U[5][15] = 1; V[10][15] = 1; V[11][15] = 1; W[4][15] = 1; W[5][15] = 1; W[6][15] = 1; W[7][15] = 1

    # Verification
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Let's test with random matrices.
    import numpy as np
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    C_calc = np.zeros((n, p))
    for c in range(16):
        Ac = np.zeros((n, m))
        Bc = np.zeros((m, p))
        for i in range(n):
            for j in range(m):
                Ac[i, j] = U_np[i*m+j, c]
        for j in range(m):
            for k in range(p):
                Bc[j, k] = V_np[j*p+k, c]
        
        M = Ac @ Bc
        for i in range(n):
            for k in range(p):
                C_calc[i, k] += W_np[k*n+i, c] * M[i, k]
                
    if not np.allclose(C_true, C_calc):
        # Fallback to schoolbook if verification fails
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
