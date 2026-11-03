"""Optimized matrix multiplication tensor decomposition for 2x3 * 3x4."""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns rank-minimal (or near-minimal) exact tensor decomposition for 2x3 * 3x4.
    Uses Strassen on the 2x2 core block to reduce rank from schoolbook 24 to 22.
    
    Decomposition:
    C = A @ B, A is 2x3, B is 3x4, C is 2x4.
    Split B into B1 (cols 0,1) and B2 (cols 2,3).
    C = [A1 @ B1 | A1 @ B2]  where A1 = A[:, 0] (2x1)
       = [A_R @ B_R1 | A_R @ B_R2]  where A_R = A[:, 1:3] (2x2), B_R1 = B[1:3, 0:2] (2x2), B_R2 = B[1:3, 2:4] (2x2)
    
    Part 1: A1 @ B1 (2x1 * 1x2) -> 2x2, rank 2 (schoolbook is 2, optimal)
    Part 2a: A_R @ B_R1 (2x2 * 2x2) -> 2x2, rank 7 (Strassen)
    Part 2b: A_R @ B_R2 (2x2 * 2x2) -> 2x2, rank 7 (Strassen)
    
    Total rank: 2 + 7 + 7 = 16? Wait, let me re-check.
    Actually, A1 is 2x1, B1 is 1x2. The tensor for 2x1 * 1x2 has rank 2 (just sum over the single index).
    But wait, the variables are:
    A: 2x3 = 6 vars: A00, A01, A02, A10, A11, A12
    B: 3x4 = 12 vars: B00, B01, B02, B03, B10, B11, B12, B13, B20, B21, B22, B23
    C: 2x4 = 8 vars: C00, C01, C02, C03, C10, C11, C12, C13
    
    C_i,k = sum_j A_i,j * B_j,k
    
    Let's use the Strassen decomposition on the 2x2 block multiplication of A_R (2x2) and B_R (2x2).
    A_R = [[A01, A02], [A11, A12]]
    B_R1 = [[B10, B11], [B20, B21]]
    B_R2 = [[B12, B13], [B22, B23]]
    
    For 2x2 * 2x2 -> 2x2, Strassen uses 7 multiplications.
    But we have two separate 2x2 multiplications sharing the same left matrix A_R.
    This is a tensor product structure or a direct sum.
    Since the right matrices are disjoint, the rank is sum of ranks: 7 + 7 = 14 for the A_R part.
    Plus the A1 part: A1 = [A00, A10]^T.
    C00 += A00*B00, C01 += A00*B01
    C10 += A10*B00, C11 += A10*B01
    
    This part involves A00, A10, B00, B01.
    The tensor for this part has rank 4 (schoolbook: A00*B00, A00*B01, A10*B00, A10*B01).
    Can we do better? 2x1 * 1x2 is essentially outer product. Rank is 2? No, rank of tensor T_{i,k} = sum_j A_{i,j} B_{j,k} with j=0 only.
    T_{i,k} = A_{i,0} B_{0,k}. This is a rank 1 tensor if we consider A_{i,0} and B_{0,k} as vectors.
    Wait, the tensor is defined over indices (i, j, k). Here j is fixed to 0.
    The support is only at j=0. The tensor is T(i, 0, k) = A(i,0) B(0,k).
    This is a rank 1 tensor in the subspace where j=0? No, the tensor rank is the minimum number of rank-1 tensors to sum to it.
    T(i, j, k) = 0 for j != 0. For j=0, T(i, 0, k) = A(i,0) B(0,k).
    This can be written as 1 * (A(i,0) B(0,k) if j=0 else 0).
    Actually, it's the product of vector u_i = A(i,0), v_j = delta_{j,0}, w_k = B(0,k).
    So it is a single rank-1 tensor?
    U_i = A(i,0), V_j = 1 if j=0 else 0, W_k = B(0,k).
    Sum_c U_ic V_jc W_kc.
    If c=0: U_i0 = A(i,0), V_00 = 1, W_k0 = B(0,k).
    This gives A(i,0) * 1 * B(0,k) for j=0, and 0 for j!=0 because V_j0 = 0 for j!=0.
    So yes, rank 1 is sufficient for the A1*B1 part!
    
    So total rank = 1 (for A1*B1) + 7 (for A_R*B_R1) + 7 (for A_R*B_R2) = 15.
    
    Let's verify the indices for the rank 1 part:
    U[i*m+j, c] with m=3. i in 0..1, j in 0..2.
    For A1*B1 part, only j=0 is involved.
    U[i*3+0, 0] = 1 for i=0,1.
    V[j*p+k, 0] with p=4. j in 0..2, k in 0..3.
    Only j=0 is involved.
    V[0*4+k, 0] = 1 for k=0,1. (B1 is B[:,0:2])
    W[k*n+i, 0] with n=2. k in 0..3, i in 0..1.
    W[k*2+i, 0] = 1 for k=0,1. (C1 is C[:,0:2])
    
    Now for the Strassen parts.
    A_R = [[A01, A02], [A11, A12]]. Indices in U:
    A01: i=0, j=1 -> idx 0*3+1 = 1
    A02: i=0, j=2 -> idx 0*3+2 = 2
    A11: i=1, j=1 -> idx 1*3+1 = 4
    A12: i=1, j=2 -> idx 1*3+2 = 5
    
    B_R1 = [[B10, B11], [B20, B21]]. Indices in V:
    B10: j=1, k=0 -> idx 1*4+0 = 4
    B11: j=1, k=1 -> idx 1*4+1 = 5
    B20: j=2, k=0 -> idx 2*4+0 = 8
    B21: j=2, k=1 -> idx 2*4+1 = 9
    
    B_R2 = [[B12, B13], [B22, B23]]. Indices in V:
    B12: j=1, k=2 -> idx 1*4+2 = 6
    B13: j=1, k=3 -> idx 1*4+3 = 7
    B22: j=2, k=2 -> idx 2*4+2 = 10
    B23: j=2, k=3 -> idx 2*4+3 = 11
    
    C_R1 (output of A_R*B_R1) = [[C00', C01'], [C10', C11']]. Indices in W:
    C00': i=0, k=0 -> idx 0*2+0 = 0
    C01': i=0, k=1 -> idx 1*2+0 = 2
    C10': i=1, k=0 -> idx 0*2+1 = 1
    C11': i=1, k=1 -> idx 1*2+1 = 3
    
    C_R2 (output of A_R*B_R2) = [[C02', C03'], [C12', C13']]. Indices in W:
    C02': i=0, k=2 -> idx 2*2+0 = 4
    C03': i=0, k=3 -> idx 3*2+0 = 6
    C12': i=1, k=2 -> idx 2*2+1 = 5
    C13': i=1, k=3 -> idx 3*2+1 = 7
    
    Strassen 2x2 * 2x2:
    M1 = (a+d)(w+z)
    M2 = (c+d)w
    M3 = a(x-z)
    M4 = d(y-x)
    M5 = (a+b)z
    M6 = (c-a)(w+x)
    M7 = (b-d)(y+z)
    
    Result:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For B_R1:
    a = A01, b = A02, c = A11, d = A12
    w = B10, x = B11, y = B20, z = B21
    
    M1 = (A01+A12)(B10+B21)
    M2 = (A11+A12)B10
    M3 = A01(B11-B21)
    M4 = A12(B20-B11)
    M5 = (A01+A02)B21
    M6 = (A11-A01)(B10+B11)
    M7 = (A02-A12)(B20+B21)
    
    C00' = M1 + M4 - M5 + M7
    C01' = M3 + M5
    C10' = M2 + M4
    C11' = M1 - M2 + M3 + M6
    
    For B_R2:
    a = A01, b = A02, c = A11, d = A12
    w = B12, x = B13, y = B22, z = B23
    
    M1' = (A01+A12)(B12+B23)
    M2' = (A11+A12)B12
    M3' = A01(B13-B23)
    M4' = A12(B22-B13)
    M5' = (A01+A02)B23
    M6' = (A11-A01)(B12+B13)
    M7' = (A02-A12)(B22+B23)
    
    C02' = M1' + M4' - M5' + M7'
    C03' = M3' + M5'
    C12' = M2' + M4'
    C13' = M1' - M2' + M3' + M6'
    
    Total rank 15.
    Let's construct the matrices.
    """
    n, m, p = DIMENSIONS
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A1 * B1 (rank 1)
    # U[i*3+0, 0] = 1 for i=0,1
    U[0][0] = 1
    U[3][0] = 1
    # V[0*4+k, 0] = 1 for k=0,1
    V[0][0] = 1
    V[1][0] = 1
    # W[k*2+i, 0] = 1 for k=0,1
    W[0][0] = 1  # k=0, i=0
    W[2][0] = 1  # k=1, i=0
    W[1][0] = 1  # k=0, i=1
    W[3][0] = 1  # k=1, i=1
    
    # Part 2a: A_R * B_R1 (rank 7)
    # Indices in U: A01=1, A02=2, A11=4, A12=5
    # Indices in V: B10=4, B11=5, B20=8, B21=9
    # Indices in W: C00=0, C01=2, C10=1, C11=3
    
    # M1 = (A01+A12)(B10+B21)
    U[1][1] = 1
    U[5][1] = 1
    V[4][1] = 1
    V[9][1] = 1
    W[0][1] = 1   # C00 += M1
    W[3][1] = 1   # C11 += M1
    
    # M2 = (A11+A12)B10
    U[4][2] = 1
    U[5][2] = 1
    V[4][2] = 1
    W[1][2] = 1   # C10 += M2
    W[3][2] = -1  # C11 -= M2
    
    # M3 = A01(B11-B21)
    U[1][3] = 1
    V[5][3] = 1
    V[9][3] = -1
    W[2][3] = 1   # C01 += M3
    W[3][3] = 1   # C11 += M3
    
    # M4 = A12(B20-B11)
    U[5][4] = 1
    V[8][4] = 1
    V[5][4] = -1
    W[0][4] = 1   # C00 += M4
    W[1][4] = 1   # C10 += M4
    
    # M5 = (A01+A02)B21
    U[1][5] = 1
    U[2][5] = 1
    V[9][5] = 1
    W[0][5] = -1  # C00 -= M5
    W[2][5] = 1   # C01 += M5
    
    # M6 = (A11-A01)(B10+B11)
    U[4][6] = 1
    U[1][6] = -1
    V[4][6] = 1
    V[5][6] = 1
    W[3][6] = 1   # C11 += M6
    
    # M7 = (A02-A12)(B20+B21)
    U[2][7] = 1
    U[5][7] = -1
    V[8][7] = 1
    V[9][7] = 1
    W[0][7] = 1   # C00 += M7
    
    # Part 2b: A_R * B_R2 (rank 7)
    # Indices in U: A01=1, A02=2, A11=4, A12=5
    # Indices in V: B12=6, B13=7, B22=10, B23=11
    # Indices in W: C02=4, C03=6, C12=5, C13=7
    
    # M1' = (A01+A12)(B12+B23)
    U[1][8] = 1
    U[5][8] = 1
    V[6][8] = 1
    V[11][8] = 1
    W[4][8] = 1   # C02 += M1'
    W[7][8] = 1   # C13 += M1'
    
    # M2' = (A11+A12)B12
    U[4][9] = 1
    U[5][9] = 1
    V[6][9] = 1
    W[5][9] = 1   # C12 += M2'
    W[7][9] = -1  # C13 -= M2'
    
    # M3' = A01(B13-B23)
    U[1][10] = 1
    V[7][10] = 1
    V[11][10] = -1
    W[6][10] = 1  # C03 += M3'
    W[7][10] = 1  # C13 += M3'
    
    # M4' = A12(B22-B13)
    U[5][11] = 1
    V[10][11] = 1
    V[7][11] = -1
    W[4][11] = 1  # C02 += M4'
    W[5][11] = 1  # C12 += M4'
    
    # M5' = (A01+A02)B23
    U[1][12] = 1
    U[2][12] = 1
    V[11][12] = 1
    W[4][12] = -1 # C02 -= M5'
    W[6][12] = 1  # C03 += M5'
    
    # M6' = (A11-A01)(B12+B13)
    U[4][13] = 1
    U[1][13] = -1
    V[6][13] = 1
    V[7][13] = 1
    W[7][13] = 1  # C13 += M6'
    
    # M7' = (A02-A12)(B22+B23)
    U[2][14] = 1
    U[5][14] = -1
    V[10][14] = 1
    V[11][14] = 1
    W[4][14] = 1  # C02 += M7'
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
