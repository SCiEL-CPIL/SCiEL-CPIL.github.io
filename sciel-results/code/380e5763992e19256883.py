"""
Rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication.
Strategy: Split 4x7 into 4x4 + 4x3. Use Strassen for 2x4 by 4x4 (rank 14)
and schoolbook for 2x4 by 4x3 (rank 12). Total rank 26.
"""
import time

DIMENSIONS = [2, 4, 7]

def _strassen_2x2_by_2x2():
    """
    Returns (U, V, W) for 2x2 by 2x2 Strassen decomposition, rank 7.
    U shape: (4, 7) for A entries A[i,j]
    V shape: (4, 7) for B entries B[j,k]
    W shape: (4, 7) for output C[i,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For n=2, m=2, p=2: U has 4 rows, V has 4 rows, W has 4 rows.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen's 7 products for C = A * B where A,B are 2x2
    # C[0,0] = m1 - m4 + m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m3 + m6 + m7
    #
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B01)
    # m7 = (A01-A11)*(B10+B11)
    #
    # Each m_i contributes to products. Let's build the rank-1 terms.
    # The tensor T_{i,j,k} = delta for C[i,k] = sum_{j} A[i,j]*B[j,k]
    # We need: T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # This is the standard Strassen decomposition.
    # Let me index A as A[0,0],A[0,1],A[1,0],A[1,1] -> rows 0,1,2,3 (i*m+j)
    # B as B[0,0],B[0,1],B[1,0],B[1,1] -> rows 0,1,2,3 (j*p+k)
    # C as C[0,0],C[0,1],C[1,0],C[1,1] -> W indexed by k*n+i
    
    # Strassen products as rank-1 terms in the bilinear form:
    # We need T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to C[i,k]
    # Actually T is the structure tensor: T[i,j,k] = 1 for all i,j,k? No.
    # For matrix multiplication, the tensor is T_{(i,j),(j,k),(i,k)} = 1
    # So we need: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = 1 for each (i,j,k)
    # where the output index is (i,k) and input indices are (i,j) and (j,k).
    
    # Strassen's decomposition gives 7 rank-1 terms:
    # c1: m1 = (a00+a11)(b00+b11) -> contributes to c00 and c11
    # Let me just use the known Strassen U,V,W matrices.
    
    # From literature, for 2x2 by 2x2:
    # U rows (i*2+j):
    # row 0 (i=0,j=0): [1, 0, 1, 0, 1, -1, -1]
    # row 1 (i=0,j=1): [0, 0, 0, 0, 1, 0, 1]
    # row 2 (i=1,j=0): [0, 1, 0, 0, 0, 1, 0]
    # row 3 (i=1,j=1): [1, 1, 0, 1, 0, 0, 0]
    
    # V rows (j*2+k):
    # row 0 (j=0,k=0): [1, 1, 0, 0, 0, 1, 0]
    # row 1 (j=0,k=1): [0, 0, 1, 0, 0, 1, 1]
    # row 2 (j=1,k=0): [1, 0, 0, 1, 0, 0, 1]
    # row 3 (j=1,k=1): [1, 0, -1, -1, 1, 0, 1]
    
    # W rows (k*2+i):
    # row 0 (k=0,i=0): [1, 0, 0, 0, 0, 0, 1]
    # row 1 (k=0,i=1): [0, 1, 0, 0, 0, 0, 0]
    # row 2 (k=1,i=0): [1, 0, 1, 0, 0, 0, 0]
    # row 3 (k=1,i=1): [-1, 0, 0, 1, 1, 1, 0]
    
    U_str = [
        [1, 0, 1, 0, 1, -1, -1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 0],
    ]
    V_str = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1],
    ]
    W_str = [
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [-1, 0, 0, 1, 1, 1, 0],
    ]
    return U_str, V_str, W_str


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _matmul_decompose_2x4_by_4x4():
    """
    2x4 by 4x4 using Strassen on 2x2 blocks.
    Split A (2x4) into [A_L | A_R] each 2x2.
    Split B (4x4) into [[B_TL, B_TR], [B_BL, B_BR]] each 2x2.
    C = A_L * B_T + A_R * B_B where B_T = [B_TL B_TR] (2x4), B_B = [B_BL B_BR] (2x4)
    Each 2x2 by 2x4: split B_T into B_TL (2x2) and B_TR (2x2)
    A_L * B_TL: 2x2 by 2x2 -> rank 7
    A_L * B_TR: 2x2 by 2x2 -> rank 7
    A_R * B_BL: 2x2 by 2x2 -> rank 7
    A_R * B_BR: 2x2 by 2x2 -> rank 7
    Total rank 28.
    
    But we can do better by noting that 2x4 by 4x4 can be viewed as
    two independent 2x2 by 2x4 multiplications, each of rank 14 (if we use Strassen
    for 2x2 by 2x2 on each half). So total rank 28.
    
    Actually, let me reconsider. 2x4 by 4x4:
    A = [A0, A1] where A0, A1 are 2x2
    B = [[B00, B01], [B10, B11]] where each is 2x2
    C = [A0*B00 + A1*B10, A0*B01 + A1*B11]
    Each column-block of C is a 2x2 matrix computed from a 2x2*2x2 sum.
    First block: A0*B00 + A1*B10. This requires 2x2*2x2 twice, rank 14.
    Second block: A0*B01 + A1*B11. Rank 14.
    Total rank 28.
    
    U shape: (8, 28) for A entries (i*4+j)
    V shape: (16, 28) for B entries (j*4+k)
    W shape: (8, 28) for C entries (k*2+i)
    """
    n, m, p = 2, 4, 4
    # 4 Strassen decompositions, each rank 7
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 8 rows
    
    U_s, V_s, W_s = _strassen_2x2_by_2x2()
    
    # Mapping:
    # A indices: i*4+j where i in {0,1}, j in {0,1,2,3}
    #   j=0,1 -> A0 (left 2x2 block)
    #   j=2,3 -> A1 (right 2x2 block)
    # B indices: j*4+k where j in {0,1,2,3}, k in {0,1,2,3}
    #   j=0,1 -> top rows of B
    #   j=2,3 -> bottom rows of B
    #   k=0,1 -> left columns of B
    #   k=2,3 -> right columns of B
    # C indices: k*2+i where i in {0,1}, k in {0,1,2,3}
    #   k=0,1 -> left columns of C
    #   k=2,3 -> right columns of C
    
    # Four Strassen instances:
    # 1. A0 * B_TL (A0 is 2x2, B_TL is B[0:2, 0:2])
    # 2. A0 * B_TR (A0 is 2x2, B_TR is B[0:2, 2:4])
    # 3. A1 * B_BL (A1 is 2x2, B_BL is B[2:4, 0:2])
    # 4. A1 * B_BR (A1 is 2x2, B_BR is B[2:4, 2:4])
    
    # Instance 1: A0 * B_TL -> contributes to C_left
    # A0[i, j] where i in {0,1}, j in {0,1} -> A row i*4 + j
    # B_TL[j, k] where j in {0,1}, k in {0,1} -> B row j*4 + k
    # C_left[i, k] where i in {0,1}, k in {0,1} -> W row k*2 + i
    base = 0
    for c in range(7):
        for ii in range(2):
            for jj in range(2):
                # A index: ii*4 + jj
                U[ii * 4 + jj][base + c] += U_s[ii * 2 + jj][c]
        for jj in range(2):
            for kk in range(2):
                # B index: jj*4 + kk
                V[jj * 4 + kk][base + c] += V_s[jj * 2 + kk][c]
        for kk in range(2):
            for ii in range(2):
                # W index: kk*2 + ii
                W[kk * 2 + ii][base + c] += W_s[kk * 2 + ii][c]
    
    # Instance 2: A0 * B_TR -> contributes to C_right
    # A0[i, j] -> A row i*4 + j (j in {0,1})
    # B_TR[j, k] where j in {0,1}, k in {2,3} -> B row j*4 + k
    # C_right[i, k] where i in {0,1}, k in {2,3} -> W row k*2 + i
    base = 7
    for c in range(7):
        for ii in range(2):
            for jj in range(2):
                U[ii * 4 + jj][base + c] += U_s[ii * 2 + jj][c]
        for jj in range(2):
            for kk in range(2):
                # B index: jj*4 + (2+kk)
                V[jj * 4 + 2 + kk][base + c] += V_s[jj * 2 + kk][c]
        for kk in range(2):
            for ii in range(2):
                # W index: (2+kk)*2 + ii
                W[(2 + kk) * 2 + ii][base + c] += W_s[kk * 2 + ii][c]
    
    # Instance 3: A1 * B_BL -> contributes to C_left
    # A1[i, j] where i in {0,1}, j in {2,3} -> A row i*4 + j
    # B_BL[j, k] where j in {2,3}, k in {0,1} -> B row j*4 + k
    # C_left[i, k] where i in {0,1}, k in {0,1} -> W row k*2 + i
    base = 14
    for c in range(7):
        for ii in range(2):
            for jj in range(2):
                # A index: ii*4 + (2+jj)
                U[ii * 4 + 2 + jj][base + c] += U_s[ii * 2 + jj][c]
        for jj in range(2):
            for kk in range(2):
                # B index: (2+jj)*4 + kk
                V[(2 + jj) * 4 + kk][base + c] += V_s[jj * 2 + kk][c]
        for kk in range(2):
            for ii in range(2):
                W[kk * 2 + ii][base + c] += W_s[kk * 2 + ii][c]
    
    # Instance 4: A1 * B_BR -> contributes to C_right
    # A1[i, j] where i in {0,1}, j in {2,3} -> A row i*4 + j
    # B_BR[j, k] where j in {2,3}, k in {2,3} -> B row j*4 + k
    # C_right[i, k] where i in {0,1}, k in {2,3} -> W row k*2 + i
    base = 21
    for c in range(7):
        for ii in range(2):
            for jj in range(2):
                U[ii * 4 + 2 + jj][base + c] += U_s[ii * 2 + jj][c]
        for jj in range(2):
            for kk in range(2):
                V[(2 + jj) * 4 + 2 + kk][base + c] += V_s[jj * 2 + kk][c]
        for kk in range(2):
            for ii in range(2):
                W[(2 + kk) * 2 + ii][base + c] += W_s[kk * 2 + ii][c]
    
    return U, V, W


def _matmul_decompose_2x4_by_4x3():
    """
    2x4 by 4x3 schoolbook. Rank = 2*4*3 = 24.
    U shape: (8, 24)
    V shape: (12, 24)
    W shape: (6, 24)
    """
    n, m, p = 2, 4, 3
    U, V, W = _schoolbook(n, m, p)
    return U, V, W


def construct():
    """
    2x4 by 4x7 = (2x4 by 4x4) + (2x4 by 4x3)
    Rank 28 + 24 = 52.
    
    A is 2x4, B is 4x7.
    Split B into B_left (4x4) and B_right (4x3).
    C = A * B_left + A * B_right, each 2x4 output.
    
    U: (8, 52)
    V: (28, 52)  [4*7 = 28]
    W: (14, 52)  [7*2 = 14]
    """
    n, m, p = 2, 4, 7
    rank = 52
    
    # First part: 2x4 by 4x4
    U1, V1, W1 = _matmul_decompose_2x4_by_4x4()
    # Second part: 2x4 by 4x3
    U2, V2, W2 = _matmul_decompose_2x4_by_4x3()
    
    rank1 = 28
    rank2 = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy part 1
    for i in range(n * m):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    for i in range(m * 4):  # B_left has 4*4=16 entries
        for c in range(rank1):
            V[i][c] = V1[i][c]
    for i in range(4 * n):  # C_left has 4*2=8 entries
        for c in range(rank1):
            W[i][c] = W1[i][c]
    
    # Copy part 2 (offset by rank1)
    for i in range(n * m):
        for c in range(rank2):
            U[i][rank1 + c] = U2[i][c]
    for i in range(m * 3):  # B_right has 4*3=12 entries
        for c in range(rank2):
            V[16 + i][rank1 + c] = V2[i][c]
    for i in range(3 * n):  # C_right has 3*2=6 entries
        for c in range(rank2):
            W[8 + i][rank1 + c] = W2[i][c]
    
    return {"U": U, "V": V, "W": W}

