"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    
    # Rank-7 decomposition of 2x2 matrix multiplication.
    # For (A11*A21 + A12*A22, A21*A11 + A22*A12; A11*A21 + A12*A22, A21*A11 + A22*A12)
    # The 7-term decomposition uses:
    # f1 = A11*(B11+B12), f2 = A12*(B21+B22), f3 = A21*(B11+B22), f4 = A22*(B21+B12)
    # f5 = A11*B22, f6 = A22*B11, f7 = A12*B21 - A21*B12
    # This gives C11 = f1 + f2 - f5 - f7, C12 = f3 + f4 - f6 - f7
    # C21 = f3 + f4 - f5 - f7, C22 = f1 + f2 - f6 - f7
    # Wait, let me use the standard Strassen-like rank 7:
    
    # Standard 2x2 rank-7 tensor decomposition coefficients:
    # Each term c is a triple (a_factor, b_factor, c_row, c_col) where
    # C_row,col += a_factor * a[row1,col1]*b[row2,col2]
    
    # Using the known rank-7 decomposition:
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # 
    # 7 bilinear forms:
    # m1 = a11 * (b11 + b12)
    # m2 = a12 * (b21 + b22)  
    # m3 = a21 * (b11 + b22)
    # m4 = a22 * (b21 + b12)
    # m5 = a11 * b22
    # m6 = a22 * b11
    # m7 = a12 * b21 - a21 * b12
    #
    # C11 = m1 + m2 - m5 - m7
    # C12 = m3 + m4 - m6 - m7
    # C21 = m3 + m4 - m5 - m7  
    # C22 = m1 + m2 - m6 - m7
    
    # For 3x4 * 4x4, decompose 3x4 into two 2x4 blocks (rows 0-1 and row 2),
    # and 4x4 into two 2x4 blocks.
    # Actually, let's think of it as:
    # A is 3x4, B is 4x4.
    # Split A into A_top (2x4) and A_bot (1x4)
    # Split B into B_left (4x2) and B_right (4x2)
    # 
    # Result C is 3x4.
    # C_top_left (2x2) = A_top * B_left
    # C_top_right (2x2) = A_top * B_right
    # C_bot_left (1x2) = A_bot * B_left
    # C_bot_right (1x2) = A_bot * B_right
    
    # For each 2x2 block product, use rank-7.
    # A_top * B_left: 2x4 * 4x2 -> 2x2, use rank 7
    # A_top * B_right: 2x4 * 4x2 -> 2x2, use rank 7
    # A_bot * B_left: 1x4 * 4x2 -> 1x2, rank 2 (schoolbook)
    # A_bot * B_right: 1x4 * 4x2 -> 1x2, rank 2 (schoolbook)
    # Total: 7 + 7 + 2 + 2 = 18? No, that's not right either.
    
    # Let me reconsider. The tensor for 3x4 * 4x4 has:
    # n=3, m=4, p=4
    # U shape: (12, rank), V shape: (16, rank), W shape: (12, rank)
    # W[k*n+i, r] means W index is k*3 + i
    
    # Schoolbook rank = 3*4*4 = 48
    
    # Strategy: Use the 2x2 rank-7 structure.
    # Write the 3x4 matrix A as:
    # A = [[a11, a12, a13, a14],
    #      [a21, a22, a23, a24],
    #      [a31, a32, a33, a34]]
    # 
    # B = [[b11, b12, b13, b14],
    #      [b21, b22, b23, b24],
    #      [b31, b32, b33, b34],
    #      [b41, b42, b43, b44]]
    
    # C = A*B is 3x4.
    
    # Split into 2x2 blocks of the output:
    # C[0:2, 0:2] = A[0:2, :] * B[:, 0:2]  (2x4 * 4x2)
    # C[0:2, 2:4] = A[0:2, :] * B[:, 2:4]  (2x4 * 4x2)
    # C[2:3, 0:2] = A[2:3, :] * B[:, 0:2]  (1x4 * 4x2)
    # C[2:3, 2:4] = A[2:3, :] * B[:, 2:4]  (1x4 * 4x2)
    
    # For the 2x4 * 4x2 products, I can further split the middle dimension.
    # 2x4 * 4x2: split the 4 into 2+2.
    # A_top = [[a11, a12], [a21, a22]] and [[a13, a14], [a23, a24]]
    # B_left = [[b11, b12], [b21, b22]] and [[b31, b32], [b41, b42]]
    
    # So C[0:2, 0:2] = A_top1 * B_left1 + A_top2 * B_left2
    # Each 2x2 * 2x2 = 2x2 has rank 7.
    # So C[0:2, 0:2] needs 7+7=14 terms.
    # Similarly C[0:2, 2:4] needs 14 terms.
    # C[2:3, 0:2] = A_bot1 * B_left1 + A_bot2 * B_left2
    # 1x2 * 2x2 = 1x2 has rank 2 each, so 2+2=4 terms.
    # C[2:3, 2:4] needs 4 terms.
    # Total: 14+14+4+4 = 36.
    
    # Let me implement this.
    
    # Rank-7 2x2*2x2 decomposition:
    # For A=[[a11,a12],[a21,a22]], B=[[b11,b12],[b21,b22]], C=A*B
    # 7 terms with coefficients:
    # term0: A factor: a11, B factor: b11+b12, contributes to C[0,0] with +1, C[0,1] with +1
    # term1: A factor: a12, B factor: b21+b22, contributes to C[0,0] with +1, C[0,1] with +1
    # term2: A factor: a21, B factor: b11+b22, contributes to C[1,0] with +1, C[1,1] with +1
    # term3: A factor: a22, B factor: b21+b12, contributes to C[1,0] with +1, C[1,1] with +1
    # term4: A factor: a11, B factor: b22, contributes to C[0,0] with -1, C[1,1] with -1
    # term5: A factor: a22, B factor: b11, contributes to C[0,0] with -1, C[1,1] with -1
    # term6: A factor: a12-b21... wait this is getting confusing.
    
    # Let me use a cleaner formulation. The 2x2 matrix multiplication tensor
    # T_{ijk} = delta_{ik}delta_{jk}... no.
    # T is a 2x2x2x2 tensor: T[i,j,k,l] = 1 if i==k and j==l (for C[i,j] = sum A[i,k]*B[k,j])
    # Actually for 2x2: C[i,j] = sum_k A[i,k]*B[k,j]
    # So the tensor has entries T[i,k,j] where i,k,j in {0,1}, T[i,k,j] = 1 always (it's the identity in k dimension).
    # Wait no, T[i,k,j] = delta_{ik}... no.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So the multiplication tensor is M[i,k,j] = 1 for all i,k,j in {0,1}. It's just the 8-term sum.
    # The rank is 7 because we can find 7 triples (u,v,w) such that
    # sum_r u_r[i,k] * v_r[k,j] * w_r[i,j] = delta_{ik}... 
    
    # I'm overcomplicating this. Let me just verify the schoolbook works and try a simple reduction.
    
    # For now, return schoolbook as fallback.
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
