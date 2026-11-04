"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _build_2x2_strassen():
    """Return (U, V, W) for 2x2 x 2x2 matrix multiplication, rank 7 (Strassen).
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Strassen's 7 products for 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a10+a11)*b00
    # p3 = a00*(b01-b11)
    # p4 = a11*(b10-b00)
    # p5 = (a00+a01)*b11
    # p6 = (a10-a00)*(b00+b10)
    # p7 = (a01-a11)*(b01+b11)
    # C00 = p1+p4-p5+p7
    # C01 = p3+p5
    # C10 = p2+p4
    # C11 = p1-p2+p3+p6
    
    # Row indices for U: i*2+j
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # Row indices for V: j*2+k
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # Row indices for W: k*2+i
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # p1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    # p2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = 1
    # p3
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    # p4
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    # p5
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    # p6
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[2,5] = 1
    W[3,5] = 1
    # p7
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _build_2x2x1():
    """Return (U, V, W) for 2x2 x 2x1, rank 4.
    U shape (4,4), V shape (2,4), W shape (2,4).
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    # C0 = a00*b0 + a01*b1, C1 = a10*b0 + a11*b1
    for j in range(2):
        for k in range(2):
            r = j*2+k
            U[j*2, r] = 1  # a0j
            U[j*2+2, r] = 1  # a1j
            V[k, r] = 1
            W[k*2+j, r] = 1  # k*2+i where i=0 for C0, i=1 for C1 -> k*2+0, k*2+1
    # Fix W indices: W[k*2+i, r]
    # For C0 (i=0): W[2k, r]
    # For C1 (i=1): W[2k+1, r]
    W = np.zeros((2, 4))
    for j in range(2):
        for k in range(2):
            r = j*2+k
            U[j*2, r] = 1  # a0j (row j*2+0)
            U[j*2+2, r] = 1  # a1j (row j*2+2)
            V[k, r] = 1
            W[k*2, r] = 1  # C0
            W[k*2+1, r] = 1  # C1
    return U, V, W

def _build_1x2x2():
    """Return (U, V, W) for 1x2 x 2x2, rank 4.
    U shape (2,4), V shape (4,4), W shape (2,4).
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    # A is 1x2: [a0, a1], B is 2x2, C = A*B is 1x2
    # C0 = a0*b00 + a1*b10, C1 = a0*b01 + a1*b11
    for j in range(2):
        for k in range(2):
            r = j*2+k
            U[j, r] = 1  # a_j
            V[j*2+k, r] = 1  # b_jk
            W[k, r] = 1  # C_k (since i=0, W[k*1+0, r] = W[k, r])
    return U, V, W

def _build_2x1x1():
    """Return (U, V, W) for 2x1 x 1x1, rank 2.
    U shape (2,2), V shape (1,2), W shape (1,2).
    """
    U = np.zeros((2, 2))
    V = np.zeros((1, 2))
    W = np.zeros((1, 2))
    for i in range(2):
        U[i, i] = 1
        V[0, i] = 1
        W[0, i] = 1
    return U, V, W

def _build_1x1x2():
    """Return (U, V, W) for 1x1 x 1x2, rank 2.
    U shape (1,2), V shape (2,2), W shape (2,2).
    """
    U = np.zeros((1, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    for k in range(2):
        U[0, k] = 1
        V[k, k] = 1
        W[k, k] = 1
    return U, V, W

def _build_2x2_dot():
    """Return (U, V, W) for 2x2 dot product (scalar output), rank 1? No, rank 4.
    Actually, a^T b where a, b are 2-vectors. This is 1x2 x 2x1, rank 2.
    Wait, the previous analysis said rank 1 for dot product? No, sum a_i b_i is not rank 1.
    Let's use 1x2 x 2x1 which is rank 2.
    """
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    for i in range(2):
        U[i, i] = 1
        V[i, i] = 1
        W[0, i] = 1
    return U, V, W

def _build_3x3x1():
    """Return (U, V, W) for 3x3 x 3x1, rank 9.
    U shape (9,9), V shape (3,9), W shape (3,9).
    """
    U = np.zeros((9, 9))
    V = np.zeros((3, 9))
    W = np.zeros((3, 9))
    for i in range(3):
        for j in range(3):
            r = i*3+j
            U[i*3+j, r] = 1
            V[j, r] = 1
            W[j, r] = 1  # W[k*n+i] where k=j, n=3, i=i -> W[j*3+i]
    # Fix W: W[k*3+i, r]
    W = np.zeros((3, 9))
    for i in range(3):
        for j in range(3):
            r = i*3+j
            U[i*3+j, r] = 1
            V[j, r] = 1
            W[j*3+i, r] = 1
    return U, V, W

def _build_3x3x3_rank22():
    """
    Build a rank 22 decomposition for 3x3 x 3x3.
    Partition A into [[A11, a0], [a1, a22]] where A11 is 2x2, a0 is 1x2, a1 is 2x1, a22 is scalar.
    Similarly for B.
    C = A*B
    C11 = A11*B11 + a0*b1 + a1*b0 + a22*b22
    C02 = a0*B02 + a22*b22_row? 
    Let's use the standard partition:
    A = [[A11, a0], [a1, a22]]
    B = [[B11, b0], [b1, b22]]
    C11 = A11*B11 + a0*b1 + a1*b0 + a22*b22
    C02 = a0*B02 + a22*b22_row? No.
    A is 3x3: rows 0,1,2. Cols 0,1,2.
    A11 = A[0:2, 0:2] (2x2)
    a0 = A[0, 2] (1x1)
    a1 = A[1, 2] (1x1)
    a2 = A[2, 0:2] (1x2)
    a3 = A[2, 2] (1x1)
    
    B11 = B[0:2, 0:2] (2x2)
    b0 = B[0:2, 2] (2x1)
    b1 = B[2, 0:2] (1x2)
    b2 = B[2, 2] (1x1)
    
    C = A*B
    C[0:2, 0:2] = A11*B11 + a0*b0 + a1*b1? No.
    C[0:2, 0:2] = A11*B11 + A[0:2,2]*B[2,0:2]
    = A11*B11 + [a0, a1]^T * b1
    C[0:2, 2] = A11*b0 + a0*b2
    C[2, 0:2] = a2*B11 + a3*b1
    C[2, 2] = a2*b0 + a3*b2
    
    Terms:
    1. A11*B11: Rank 7 (Strassen)
    2. [a0, a1]^T * b1: This is 2x1 * 1x2 -> 2x2. Rank 2.
       (a0*b1[0], a0*b1[1]; a1*b1[0], a1*b1[1])
       Actually it's outer product of vector [a0, a1] and [b1[0], b1[1]]. Rank 1? No, it's 2x2.
       Wait, [a0, a1]^T is 2x1, b1 is 1x2. Product is 2x2.
       This is 2x1 x 1x2 multiplication. Rank is 2*1*2 = 4? No, rank of tensor M_{2,1,2} is 2.
       Let's check: C_ij = u_i v_j. Sum over r.
       U[i,r] = u_i, V[j,r] = v_j, W[k,r] = delta_{ik}delta_{jk}? No.
       For 2x1 x 1x2:
       A is 2x1: u. B is 1x2: v. C is 2x2: u v^T.
       C_ij = u_i v_j.
       Decomposition: For each i,j, C_ij = u_i * v_j.
       This is not rank 1. It's sum of rank 1 tensors?
       u v^T = sum_k (u e_k) (e_k v^T)? No.
       Actually, the tensor M_{2,1,2} has rank 2.
       Proof: C_00 = u0 v0, C_01 = u0 v1, C_10 = u1 v0, C_11 = u1 v1.
       Rank 2:
       r1: U=[1,0], V=[1,0], W=[1,0] -> C00=1, others 0. (u0 v0)
       r2: U=[0,1], V=[0,1], W=[0,1] -> C11=1, others 0. (u1 v1)
       Missing C01, C10.
       Actually rank is 4?
       Let's check lower bound. max(2*1, 1*2, 2*2) = max(2,2,4) = 4.
       So rank is 4.
       
    3. a1*b1? No, the term is a0*b1[0] etc.
    Let's re-evaluate the terms.
    
    C11 (2x2) = A11*B11 + A[0:2,2] * B[2,0:2]
    Term 2 is 2x1 * 1x2. Rank 4.
    
    C02 (2x1) = A11*b0 + a0*b2
    Term 3: A11*b0 is 2x2 * 2x1. Rank 4.
    Term 4: a0*b2 is 1x1 * 1x1. Rank 1. (Scalar times vector? No, a0 is scalar, b2 is scalar. Result is 2x1? No.)
    Wait. A[0:2, 2] is 2x1. B[2, 0:2] is 1x2.
    C[0:2, 2] = A[0:2, 0:2] * B[0:2, 2] + A[0:2, 2] * B[2, 2]
    = A11 * b0 + A_col * b2
    A11*b0: 2x2 x 2x1. Rank 4.
    A_col*b2: 2x1 x 1x1. Rank 2.
    
    C[2, 0:2] = A[2, 0:2] * B[0:2, 0:2] + A[2, 2] * B[2, 0:2]
    = a2 * B11 + a3 * b1
    a2*B11: 1x2 x 2x2. Rank 4.
    a3*b1: 1x1 x 1x2. Rank 2.
    
    C[2, 2] = A[2, 0:2] * B[0:2, 2] + A[2, 2] * B[2, 2]
    = a2 * b0 + a3 * b2
    a2*b0: 1x2 x 2x1. Rank 2.
    a3*b2: 1x1 x 1x1. Rank 1.
    
    Total Rank:
    1. A11*B11: 7
    2. A_col * b1: 4
    3. A11 * b0: 4
    4. A_col * b2: 2
    5. a2 * B11: 4
    6. a3 * b1: 2
    7. a2 * b0: 2
    8. a3 * b2: 1
    Sum: 7+4+4+2+4+2+2+1 = 26.
    
    This is not 22. The previous claim of 22 was wrong or I missed a trick.
    Known best for 3x3 is 23 (Strassen-like).
    I will use a known rank 23 decomposition or just use schoolbook 27 if I can't find 23 easily.
    Given time constraints, I will use a block approach with Strassen for 2x2 and schoolbook for the rest, aiming for a good score.
    Rank 23 for 3x3x3 is known. I will implement a specific rank 23 decomposition if possible, or fall back to a combination of Strassen and schoolbook.
    
    Actually, let's just use the schoolbook for 3x3x3 (rank 27) and see the total rank.
    Total rank for 3x3x8 using schoolbook for 3x3x3 blocks:
    2 * 27 = 54.
    Schoolbook for 3x3x8 is 72.
    72/54 = 1.33. This is a solid improvement.
    
    Can I do better?
    If I use rank 23 for 3x3x3:
    2 * 23 = 46.
    72/46 = 1.56.
    
    I will implement a rank 23 decomposition for 3x3x3.
    Reference: "On the rank of the 3x3x3 matrix multiplication tensor"
    One such decomposition uses 23 terms.
    
    However, implementing a specific 23-term decomposition from scratch is error-prone.
    I will stick to the block decomposition with Strassen (rank 7) for 2x2 and schoolbook for the rest, but optimized.
    
    Let's try to combine Strassen for the 2x2 core and efficient handling of the cross terms.
    
    Alternative: Just use the schoolbook for the whole 3x3x8 but with a better constant? No.
    
    Let's go with the block decomposition:
    3x3x8 = 2 * (3x3x4)
    3x3x4 = 3x3x3 + 3x3x1
    Rank(3x3x3) = 23 (if I can find it) or 27 (schoolbook).
    Rank(3x3x1) = 9.
    
    If I use 27: 2*(27+9) = 72. No gain.
    If I use 23: 2*(23+9) = 64. Gain 72/64 = 1.125.
    
    I will implement a rank 23 decomposition for 3x3x3.
    One known decomposition is based on Strassen's 7 terms for 2x2, plus 16 more terms.
    
    Actually, I will use a simpler approach:
    Use Strassen for the 2x2 part (rank 7).
    Use schoolbook for the remaining parts.
    This gave rank 26 above.
    
    Is there a rank 23? Yes.
    I will hardcode a known rank 23 decomposition for 3x3x3.
    
    Since I don't have the explicit 23 terms memorized, I will use the rank 26 decomposition (Strassen + Schoolbook cross terms) which is 26.
    2 * (26 + 9) = 70.
    72/70 = 1.028.
    
    This is a small improvement.
    
    Can I reduce the cross terms?
    The cross terms were:
    2. A_col * b1: 4
    3. A11 * b0: 4
    4. A_col * b2: 2
    5. a2 * B11: 4
    6. a3 * b1: 2
    7. a2 * b0: 2
    8. a3 * b2: 1
    Sum: 19.
    Total: 7 + 19 = 26.
    
    If I can reduce any of these...
    A_col * b1 (2x1 x 1x2) rank 4.
    A11 * b0 (2x2 x 2x1) rank 4.
    a2 * B11 (1x2 x 2x2) rank 4.
    These are minimal.
    
    So 26 is likely the best I can get with this partition.
    
    I will implement the rank 26 decomposition for 3x3x3.
    Total rank for 3x3x8: 2 * (26 + 9) = 70.
    
    Let's implement this.
    """
    # This function will return U, V, W for 3x3x3 with rank 26.
    U = np.zeros((9, 26))
    V = np.zeros((9, 26))
    W = np.zeros((9, 26))
    
    # 1. A11*B11 (Strassen, rank 7)
    U11, V11, W11 = _build_2x2_strassen()
    # A11 is A[0:2, 0:2], indices 0,1,2,3 in U (i*3+j)
    # A11 rows: 0,1 (i=0,1), cols 0,1 (j=0,1). U indices: 0,1,3,4.
    # B11 is B[0:2, 0:2], indices 0,1,3,4 in V (j*3+k).
    # C11 is C[0:2, 0:2], indices 0,1,3,4 in W (k*3+i).
    
    # Map Strassen U (4x7) to global U (9x7)
    # Strassen U row i*2+j -> global U row i*3+j
    # i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 3; i=1,j=1 -> 4
    map_u = [0, 1, 3, 4]
    for r in range(7):
        for local_i, global_i in enumerate(map_u):
            U[global_i, r] = U11[local_i, r]
    
    # Map Strassen V (4x7) to global V (9x7)
    # Strassen V row j*2+k -> global V row j*3+k
    # j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 3; j=1,k=1 -> 4
    map_v = [0, 1, 3, 4]
    for r in range(7):
        for local_j, global_j in enumerate(map_v):
            V[global_j, r] = V11[local_j, r]
    
    # Map Strassen W (4x7) to global W (9x7)
    # Strassen W row k*2+i -> global W row k*3+i
    # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 3; k=1,i=1 -> 4
    map_w = [0, 1, 3, 4]
    for r in range(7):
        for local_k, global_k in enumerate(map_w):
            W[global_k, r] = W11[local_k, r]
    
    # 2. A_col * b1 (2x1 x 1x2, rank 4)
    # A_col = A[0:2, 2], indices 2, 5 in U.
    # b1 = B[2, 0:2], indices 6, 7 in V.
    # Result in C[0:2, 0:2], indices 0,1,3,4 in W.
    # C_ij = A_i2 * B_2j
    # Terms:
    # r=7: A_02 * B_20 -> C_00
    # r=8: A_02 * B_21 -> C_01
    # r=9: A_12 * B_20 -> C_10
    # r=10: A_12 * B_21 -> C_11
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[1, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[3, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[4, 10] = 1
    
    # 3. A11 * b0 (2x2 x 2x1, rank 4)
    # A11 indices 0,1,3,4 in U.
    # b0 = B[0:2, 2], indices 2, 5 in V.
    # Result in C[0:2, 2], indices 2, 5 in W.
    # C_i2 = sum_j A_ij B_j2
    # Terms:
    # r=11: A_00*B_02 -> C_02
    # r=12: A_01*B_12 -> C_02
    # r=13: A_10*B_02 -> C_12
    # r=14: A_11*B_12 -> C_12
    U[0, 11] = 1; V[2, 11] = 1; W[2, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[2, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    
    # 4. A_col * b2 (2x1 x 1x1, rank 2)
    # A_col indices 2, 5 in U.
    # b2 = B[2, 2], index 8 in V.
    # Result in C[0:2, 2], indices 2, 5 in W.
    # C_02 += A_02*B_22
    # C_12 += A_12*B_22
    U[2, 15] = 1; V[8, 15] = 1; W[2, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    # 5. a2 * B11 (1x2 x 2x2, rank 4)
    # a2 = A[2, 0:2], indices 6, 7 in U.
    # B11 indices 0,1,3,4 in V.
    # Result in C[2, 0:2], indices 6, 7 in W.
    # C_2j = sum_i A_2i B_ij
    # Terms:
    # r=17: A_20*B_00 -> C_20
    # r=18: A_20*B_10 -> C_20
    # r=19: A_21*B_01 -> C_21
    # r=20: A_21*B_11 -> C_21
    U[6, 17] = 1; V[0, 17] = 1; W[6, 17] = 1
    U[6, 18] = 1; V[3, 18] = 1; W[6, 18] = 1
    U[7, 19] = 1; V[1, 19] = 1; W[7, 19] = 1
    U[7, 20] = 1; V[4, 20] = 1; W[7, 20] = 1
    
    # 6. a3 * b1 (1x1 x 1x2, rank 2)
    # a3 = A[2, 2], index 8 in U.
    # b1 indices 6, 7 in V.
    # Result in C[2, 0:2], indices 6, 7 in W.
    # C_20 += A_22*B_20
    # C_21 += A_22*B_21
    U[8, 21] = 1; V[6, 21] = 1; W[6, 21] = 1
    U[8, 22] = 1; V[7, 22] = 1; W[7, 22] = 1
    
    # 7. a2 * b0 (1x2 x 2x1, rank 2)
    # a2 indices 6, 7 in U.
    # b0 indices 2, 5 in V.
    # Result in C[2, 2], index 8 in W.
    # C_22 = A_20*B_02 + A_21*B_12
    U[6, 23] = 1; V[2, 23] = 1; W[8, 23] = 1
    U[7, 24] = 1; V[5, 24] = 1; W[8, 24] = 1
    
    # 8. a3 * b2 (1x1 x 1x1, rank 1)
    # a3 index 8 in U.
    # b2 index 8 in V.
    # Result in C[2, 2], index 8 in W.
    U[8, 25] = 1; V[8, 25] = 1; W[8, 25] = 1
    
    return U, V, W

def _build_3x3x4():
    """Build 3x3x4 decomposition.
    3x3x4 = 3x3x3 (rank 26) + 3x3x1 (rank 9).
    Total rank 35.
    """
    U333, V333, W333 = _build_3x3x3_rank22()
    U331, V331, W331 = _build_3x3x1()
    
    rank = 26 + 9
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Copy 3x3x3 part
    U[:, :26] = U333
    V[:, :26] = V333
    W[:, :26] = W333
    
    # Copy 3x3x1 part
    # 3x3x1: A is 3x3, B is 3x1, C is 3x1.
    # B indices in global 3x3x4: B is 3x4. The 3x1 part is B[:, 3] (last column).
    # So V rows for B[:, 3] are j*4+3.
    # In the local 3x3x1, V rows are j*1+0 = j.
    # We need to map local V row j to global V row j*4+3.
    # U rows are i*3+j. Same as global 3x3 part.
    # W rows are k*3+i. Same as global 3x3 part.
    
    for r in range(9):
        for i in range(3):
            for j in range(3):
                local_r = i*3+j
                if U331[i*3+j, local_r] != 0:
                    U[i*3+j, 26+r] = U331[i*3+j, local_r]
                if V331[j, local_r] != 0:
                    V[j*4+3, 26+r] = V331[j, local_r]
                if W331[j*3+i, local_r] != 0:
                    W[j*3+i, 26+r] = W331[j*3+i, local_r]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    
    # Build two 3x3x4 decompositions
    U1, V1, W1 = _build_3x3x4()
    U2, V2, W2 = _build_3x3x4()
    
    # Global shapes:
    # U: (n*m, rank) = (9, 70)
    # V: (m*p, rank) = (24, 70)
    # W: (p*n, rank) = (24, 70)
    
    rank_total = 70
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    # First 3x3x4 block: B columns 0:4
    # U rows 0:9
    # V rows 0:12 (j*4+k, k=0..3)
    # W rows 0:12 (k*3+i, k=0..3)
    U[:, :35] = U1
    V[0:12, :35] = V1
    W[0:12, :35] = W1
    
    # Second 3x3x4 block: B columns 4:8
    # U rows 0:9
    # V rows 12:24 (j*4+k, k=4..7 -> j*4+4..j*4+7)
    # W rows 12:24 (k*3+i, k=4..7 -> k*3+i)
    U[:, 35:] = U2
    V[12:24, 35:] = V2
    W[12:24, 35:] = W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
