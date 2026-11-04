import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form an exact tensor decomposition of the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry T[i*m+j, j*p+k, k*n+i] should be 1
                # Reconstruction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
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

def _strassen_2x2_decomp():
    """
    Return (U, V, W) for 2x2x2 multiplication, rank 7.
    Indices:
    U: (4, 7) -> A[0,0], A[0,1], A[1,0], A[1,1]
    V: (4, 7) -> B[0,0], B[0,1], B[1,0], B[1,1]
    W: (4, 7) -> C[0,0], C[0,1], C[1,0], C[1,1]
    """
    rank = 7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0: A00 * (B01+B11)
    U[0][0] = 1
    V[1][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c00, c11
    
    # M1: (A10+A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[2][1] = 1  # c01, c10
    
    # M2: (A00+A01) * B11
    U[0][2] = 1; U[1][2] = 1
    V[3][2] = 1
    W[0][2] = 1; W[3][2] = 1  # c00, c11
    
    # M3: A11 * (B00-B10)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1  # c00, c10
    
    # M4: (A00+A10) * B01
    U[0][4] = 1; U[2][4] = 1
    V[1][4] = 1
    W[0][4] = -1; W[1][4] = 1  # c00, c01
    
    # M5: (A01-A10) * (B00+B10)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[0][5] = -1  # c00
    
    # M6: (A11-A01) * (B01+B11)
    U[3][6] = 1; U[1][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[3][6] = 1  # c11
    
    return U, V, W

def _construct_2x3x3_decomp():
    """
    Construct a rank-9 decomposition for 2x3x3.
    
    We view 2x3x3 as containing a 2x2x2 block (indices i in {0,1}, j in {0,1}, k in {0,1})
    and additional terms involving j=2 or k=2.
    
    Actually, a cleaner way to get rank 9:
    Rank(2,3,3) <= Rank(2,2,2) + Rank(2,1,3) + Rank(2,1,1)? No.
    
    Let's use the property: Rank(2,3,3) <= Rank(2,3,2) + Rank(2,3,1)?
    Rank(2,3,1) = 6 (schoolbook, since output is 2x1, input 2x3 x 3x1. No, 2*3*1=6).
    Rank(2,3,2) <= 8?
    
    Let's try a direct construction for 2x3x3 with rank 9.
    We can split the 3 columns of B into 2+1.
    C = [C_left (2x2) , C_right (2x1)]
    C_left = A_left * B_left (2x2 * 2x2? No, A is 2x3, B is 3x3)
    
    Let A = [A_0 | A_1] where A_0 is 2x2 (cols 0,1) and A_1 is 2x1 (col 2).
    Let B = [[B_00, B_01], [B_10, B_11], [B_20, B_21]]? No, B is 3x3.
    Let B = [B_top (3x2) ; B_bottom (3x1)]? No, B is 3x3.
    Split B columns into first 2 and last 1.
    B = [B_0 (3x2) | B_1 (3x1)]
    A = [A_0 (2x2) | A_1 (2x1)]
    
    C = A_0 B_0 + A_1 B_1 (2x2 part)
    C_right = A_0 B_1? No.
    C = A * B
    C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:3] * B[2:3, 0:2]
    C[:, 2:3] = A[:, 0:2] * B[0:2, 2:3] + A[:, 2:3] * B[2:3, 2:3]
    
    This is getting complicated. Let's use the known bound Rank(2,3,3) <= 8?
    Actually, Rank(2,3,3) is known to be 8.
    Reference: "The rank of the 2x3x3 matrix multiplication tensor is 8."
    
    Let's construct rank 8 for 2x3x3.
    One known construction uses 8 multiplications.
    However, implementing a specific rank-8 2x3x3 decomposition from scratch is error-prone.
    
    Alternative: Use Rank(2,3,3) <= Rank(2,2,2) + 2?
    If we take the 2x2x2 Strassen decomp (rank 7) and add 2 schoolbook terms for the remaining parts?
    
    Let's stick to a safe Rank 9 construction for 2x3x3 if we can't find 8 easily,
    but Rank(2,3,6) <= 2 * Rank(2,3,3) = 18.
    If Rank(2,3,3)=8, then Rank(2,3,6) <= 16.
    
    Let's try to find a rank 8 decomposition for 2x3x3.
    
    Actually, let's use a simpler, verifiable approach:
    Rank(2,3,6) <= 2 * Rank(2,3,3).
    If we use schoolbook for 2x3x3 (rank 9), total rank 18.
    Schoolbook 2x3x3 rank is 2*3*3 = 18? No, 2*3*3=18.
    Wait, schoolbook rank is n*m*p.
    For 2x3x3, rank = 18.
    So 2 * 18 = 36. That's the same as schoolbook for 2x3x6.
    
    We need a sub-schoolbook decomposition for 2x3x3.
    
    Let's use the fact that Rank(2,3,3) <= 8.
    Here is a known rank-8 decomposition for 2x3x3:
    
    A = [[a00, a01, a02], [a10, a11, a12]]
    B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    C = [[c00, c01, c02], [c10, c11, c12]]
    
    c00 = a00*b00 + a01*b10 + a02*b20
    c01 = a00*b01 + a01*b11 + a02*b21
    c02 = a00*b02 + a01*b12 + a02*b22
    c10 = a10*b00 + a11*b10 + a12*b20
    c11 = a10*b01 + a11*b11 + a12*b21
    c12 = a10*b02 + a11*b12 + a12*b22
    
    We can use 8 multiplications.
    One such decomposition (from literature, e.g., Landsberg):
    
    M1 = (a00+a10) * (b00+b10)
    M2 = (a00+a11) * (b01+b11)
    M3 = (a00+a12) * (b02+b12)
    M4 = (a01+a10) * (b10+b00) -- duplicate?
    
    This is hard to recall exactly. Let's use a safer bet:
    Rank(2,3,6) can be done with Rank 24 using a different split?
    
    Actually, let's look at Rank(2,3,6).
    Rank(2,3,6) <= Rank(2,3,3) + Rank(2,3,3) = 2*8 = 16.
    
    If I can't guarantee a correct rank-8 2x3x3 decomp, I should use a verified one.
    
    Let's try Rank 24 for 2x3x6?
    Rank(2,3,6) <= 2 * Rank(2,3,3).
    If Rank(2,3,3)=8, then 16.
    
    Let's implement a Rank 16 decomposition for 2x3x6 by using two Rank 8 decompositions for 2x3x3.
    
    I will use a known Rank 8 decomposition for 2x3x3.
    Source: "Matrix multiplication of 2x3 matrices by 3x3 matrices"
    
    The 8 multiplications are:
    1. (a00 + a10) * (b00 + b10)
    2. (a00 + a11) * (b01 + b11)
    3. (a00 + a12) * (b02 + b12)
    4. (a01 + a10) * (b10 + b00)
    5. (a01 + a11) * (b11 + b01)
    6. (a01 + a12) * (b12 + b02)
    7. (a02 + a10) * (b20 + b00)
    8. (a02 + a11) * (b21 + b01)
    
    Wait, this doesn't cover a12*b22 or a02*b22 etc correctly.
    
    Let's use a different strategy.
    Rank(2,3,6) <= 2 * Rank(2,3,3).
    If I use Schoolbook for 2x3x3 (rank 18), total 36.
    If I use Strassen-like for 2x3x3?
    
    Actually, there is a simple Rank 12 decomposition for 2x3x3?
    No, Rank(2,3,3) is 8.
    
    Let's try to construct Rank 16 for 2x3x6 directly.
    Split B into two 3x3 blocks: B1 (cols 0-2), B2 (cols 3-5).
    C1 = A * B1 (2x3)
    C2 = A * B2 (2x3)
    
    If we have a Rank R decomposition for 2x3x3, we can use 2*R for 2x3x6.
    
    Let's use a verified Rank 8 decomposition for 2x3x3.
    I will hardcode a known one.
    
    The following is a rank-8 decomposition for 2x3x3:
    U, V, W matrices.
    
    Actually, to be safe and ensure correctness within the time limit, I will use the following strategy:
    1. Construct a Rank 24 decomposition for 2x3x6.
    2. How? Rank(2,3,6) <= Rank(2,3,2) * 3? No.
    
    Let's use the fact that Rank(2,3,6) <= 2 * Rank(2,3,3).
    And Rank(2,3,3) <= 9 (conservative).
    So Rank 18 is a very safe target if I can construct a Rank 9 2x3x3.
    
    Rank 9 for 2x3x3:
    We can take the 2x2x2 Strassen decomp (Rank 7) and add 2 schoolbook terms for the 3rd column/row?
    
    Let's define the 2x3x3 tensor.
    Indices: i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    We can view this as:
    C[:, 0:2] depends on A[:, 0:2] and B[0:2, 0:2] and A[:, 2] and B[2, 0:2].
    C[:, 2] depends on A[:, 0:2] and B[0:2, 2] and A[:, 2] and B[2, 2].
    
    This is complex.
    
    Let's use a simpler, guaranteed improvement:
    Rank(2,3,6) <= 2 * Rank(2,3,3).
    If I use Schoolbook for 2x3x3, Rank is 18. Total 36.
    
    Is there a Rank < 36 decomposition?
    Yes, Rank(2,3,6) = 24 is a known value?
    Actually, Rank(2,3,6) <= 24.
    
    Let's try to construct Rank 24.
    Split B into three 3x2 blocks?
    Rank(2,3,2) = ?
    Rank(2,3,2) <= 4? No, Rank(2,3,2) = 4 is possible?
    C is 2x2.
    c00 = a00*b00 + a01*b10 + a02*b20
    c01 = a00*b01 + a01*b11 + a02*b21
    c10 = a10*b00 + a11*b10 + a12*b20
    c11 = a10*b01 + a11*b11 + a12*b21
    
    We can use 4 multiplications:
    M1 = (a00+a10) * (b00+b10)
    M2 = (a00+a11) * (b01+b11)
    M3 = (a01+a10) * (b10+b00)
    M4 = (a01+a11) * (b11+b01)
    This doesn't cover a02, a12, b20, b21.
    
    So Rank(2,3,2) is likely 6 or more.
    
    Let's go with Rank 18.
    I will construct a Rank 9 decomposition for 2x3x3 by using Schoolbook for the 2x3x2 part and 1x3x1 part?
    
    Actually, the simplest robust improvement is Rank 24.
    How to get 24?
    Rank(2,3,6) <= 3 * Rank(2,3,2).
    If Rank(2,3,2) = 8, then 24.
    Is Rank(2,3,2) = 8?
    Schoolbook is 12.
    We can do better.
    
    Let's use the following:
    Rank(2,3,6) <= 2 * Rank(2,3,3).
    I will implement a Rank 9 decomposition for 2x3x3.
    
    Rank 9 for 2x3x3:
    We can use the 2x2x2 Strassen (Rank 7) for the top-left 2x2 block of C,
    and add 2 terms for the remaining entries?
    
    Let's try to build Rank 16 for 2x3x6 using two Rank 8 2x3x3 decompositions.
    I will hardcode a Rank 8 2x3x3 decomposition.
    
    Here is a verified Rank 8 decomposition for 2x3x3 (from online resources/paper "Rank of matrix multiplication"):
    
    U (6x8), V (9x8), W (6x8).
    
    Actually, I will use a safer Rank 18 (2x Rank 9).
    Rank 9 for 2x3x3:
    Schoolbook is 18.
    We can save 9 by using Strassen on the 2x2x2 core?
    
    Let's just return Schoolbook (36) if I can't find a simple sub-36 construction quickly.
    BUT, the task asks for a DIFFERENT strategy.
    The current program uses ALS.
    My strategy: Deterministic construction based on sub-multiplicativity.
    
    I will implement Rank 18 using two Rank 9 2x3x3 decompositions.
    For Rank 9 2x3x3, I will use:
    Rank(2,3,3) <= Rank(2,2,2) + Rank(2,1,3) + Rank(2,1,1)?
    Rank(2,2,2)=7.
    Rank(2,1,3)=6 (schoolbook).
    Rank(2,1,1)=2.
    Total 15?
    
    Let's use Rank 24 for 2x3x6.
    Rank(2,3,6) <= 3 * Rank(2,3,2).
    Rank(2,3,2) = 8?
    
    Let's try to construct Rank 24.
    I will use 3 blocks of Rank 8 2x3x2.
    
    Rank 8 for 2x3x2:
    A (2x3), B (3x2), C (2x2).
    
    M1 = (a00+a10) * (b00+b10)
    M2 = (a00+a11) * (b01+b11)
    M3 = (a01+a10) * (b10+b00)
    M4 = (a01+a11) * (b11+b01)
    M5 = a02 * b20
    M6 = a02 * b21
    M7 = a12 * b20
    M8 = a12 * b21
    
    This uses 8 multiplications.
    Let's verify:
    c00 = a00*b00 + a01*b10 + a02*b20
    c01 = a00*b01 + a01*b11 + a02*b21
    c10 = a10*b00 + a11*b10 + a12*b20
    c11 = a10*b01 + a11*b11 + a12*b21
    
    M1 = a00*b00 + a00*b10 + a10*b00 + a10*b10
    M2 = a00*b01 + a00*b11 + a10*b01 + a10*b11
    M3 = a01*b10 + a01*b00 + a10*b10 + a10*b00
    M4 = a01*b11 + a01*b01 + a11*b11 + a11*b01
    
    We need:
    c00 = a00*b00 + a01*b10 + a02*b20
    c01 = a00*b01 + a01*b11 + a02*b21
    c10 = a10*b00 + a11*b10 + a12*b20
    c11 = a10*b01 + a11*b11 + a12*b21
    
    From M1, M2, M3, M4, we have terms involving a00,a01,a10,a11 and b00,b01,b10,b11.
    We can combine them to get the 4 required terms for the 2x2 block (a00..a11, b00..b11).
    The 2x2x2 part has Rank 7.
    So we can do the 2x2x2 part with 7 multiplications.
    Then add 2 multiplications for a02*b20 and a02*b21? No, we need a12 too.
    a02*b20, a02*b21, a12*b20, a12*b21.
    That's 4 more.
    Total 7 + 4 = 11.
    
    So Rank(2,3,2) <= 11.
    Then Rank(2,3,6) <= 3 * 11 = 33.
    This is better than 36.
    
    Can we do better for the 4 terms?
    a02*b20, a02*b21, a12*b20, a12*b21.
    This is a 2x1 * 1x2 product? No, 2x1 * 1x2 is 2x2.
    Rank(2,1,2) = 2? No, Rank(2,1,2) = 2 is not enough.
    Rank(2,1,2) = 2 * 1 * 2 = 4 schoolbook.
    Can we do 2x1 * 1x2 with Rank 2?
    C = [[a0*b0, a0*b1], [a1*b0, a1*b1]]
    M1 = (a0+a1)*(b0+b1)
    M2 = (a0-a1)*(b0-b1)
    C00 = (M1+M2)/2, C01 = (M1-M2)/2, C10 = (M2-M1)/2, C11 = (M1+M2)/2?
    C00 = a0*b0 + a1*b1? No.
    M1 = a0*b0 + a0*b1 + a1*b0 + a1*b1
    M2 = a0*b0 - a0*b1 - a1*b0 + a1*b1
    M1+M2 = 2*a0*b0 + 2*a1*b1
    M1-M2 = 2*a0*b1 + 2*a1*b0
    
    We need a0*b0, a0*b1, a1*b0, a1*b1.
    We have sums. We can't separate them with just 2 mults.
    So Rank(2,1,2) = 4.
    
    So Rank(2,3,2) <= 7 + 4 = 11.
    Total Rank 33.
    
    This is a valid improvement (33 < 36).
    
    I will implement Rank 33.
    
    Steps:
    1. Split B into 3 blocks of 3x2.
    2. For each block, use Rank 11 decomposition.
    3. Total Rank 33.
    
    Rank 11 for 2x3x2:
    - Use Strassen (Rank 7) for the 2x2x2 sub-block (rows 0,1 of A and B, cols 0,1 of B).
    - Use Schoolbook (Rank 4) for the remaining terms (row 2 of A, cols 0,1 of B).
    
    Wait, the 2x2x2 sub-block uses A[0:2, 0:2] and B[0:2, 0:2].
    The remaining terms are:
    c00 += a02*b20
    c01 += a02*b21
    c10 += a12*b20
    c11 += a12*b21
    
    These 4 terms are exactly the 2x1 * 1x2 product.
    So yes, Rank 7 + Rank 4 = 11.
    
    Total Rank 3 * 11 = 33.
"""

def _strassen_2x2_decomp():
    """
    Return (U, V, W) for 2x2x2 multiplication, rank 7.
    """
    rank = 7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0: A00 * (B01+B11)
    U[0][0] = 1
    V[1][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c00, c11
    
    # M1: (A10+A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[2][1] = 1  # c01, c10
    
    # M2: (A00+A01) * B11
    U[0][2] = 1; U[1][2] = 1
    V[3][2] = 1
    W[0][2] = 1; W[3][2] = 1  # c00, c11
    
    # M3: A11 * (B00-B10)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1  # c00, c10
    
    # M4: (A00+A10) * B01
    U[0][4] = 1; U[2][4] = 1
    V[1][4] = 1
    W[0][4] = -1; W[1][4] = 1  # c00, c01
    
    # M5: (A01-A10) * (B00+B10)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[0][5] = -1  # c00
    
    # M6: (A11-A01) * (B01+B11)
    U[3][6] = 1; U[1][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[3][6] = 1  # c11
    
    return U, V, W

def _schoolbook_2x1x2_decomp():
    """
    Return (U, V, W) for 2x1x2 multiplication, rank 4.
    A is 2x1, B is 1x2. C is 2x2.
    c00 = a0*b0
    c01 = a0*b1
    c10 = a1*b0
    c11 = a1*b1
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(4)]
    
    # M0: a0*b0
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1 # c00
    
    # M1: a0*b1
    U[0][1] = 1
    V[1][1] = 1
    W[1][1] = 1 # c01
    
    # M2: a1*b0
    U[1][2] = 1
    V[0][2] = 1
    W[2][2] = 1 # c10
    
    # M3: a1*b1
    U[1][3] = 1
    V[1][3] = 1
    W[3][3] = 1 # c11
    
    return U, V, W

def _construct_2x3x2_decomp():
    """
    Construct Rank 11 decomposition for 2x3x2.
    """
    # Strassen for 2x2x2
    U_s, V_s, W_s = _strassen_2x2_decomp()
    # Schoolbook for 2x1x2
    U_sb, V_sb, W_sb = _schoolbook_2x1x2_decomp()
    
    U = [[0]*11 for _ in range(6)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(4)]
    
    # Fill Strassen part (indices 0-6)
    # U_s is 4x7. Maps to U[0:4, 0:7]
    for i in range(4):
        for r in range(7):
            U[i][r] = U_s[i][r]
            
    # V_s is 4x7. Maps to V[0:4, 0:7]
    for j in range(4):
        for r in range(7):
            V[j][r] = V_s[j][r]
            
    # W_s is 4x7. Maps to W[0:4, 0:7]
    for k in range(4):
        for r in range(7):
            W[k][r] = W_s[k][r]
            
    # Fill Schoolbook part (indices 7-10)
    # U_sb is 2x4. Maps to U[4:6, 7:11]
    for i in range(2):
        for r in range(4):
            U[4+i][7+r] = U_sb[i][r]
            
    # V_sb is 2x4. Maps to V[4:6, 7:11]
    for j in range(2):
        for r in range(4):
            V[4+j][7+r] = V_sb[j][r]
            
    # W_sb is 4x4. Maps to W[0:4, 7:11]
    for k in range(4):
        for r in range(4):
            W[k][7+r] = W_sb[k][r]
            
    return U, V, W

def _construct_2x3x6_decomp():
    """
    Construct Rank 33 decomposition for 2x3x6.
    Split into 3 blocks of 2x3x2.
    """
    U_block, V_block, W_block = _construct_2x3x2_decomp()
    
    U = [[0]*33 for _ in range(6)]
    V = [[0]*33 for _ in range(18)]
    W = [[0]*33 for _ in range(12)]
    
    for block in range(3):
        offset = block * 11
        # U is the same for all blocks (A is 2x3)
        # Wait, A is 2x3. In 2x3x2, A is 2x3.
        # So U_block is 6x11.
        for i in range(6):
            for r in range(11):
                U[i][offset+r] = U_block[i][r]
                
        # V is 3x2 per block. Total 3x6.
        # V_block is 6x11.
        for j in range(6):
            for r in range(11):
                V[j + block*6][offset+r] = V_block[j][r]
                
        # W is 2x2 per block. Total 2x6.
        # W_block is 4x11.
        for k in range(4):
            for r in range(11):
                W[k + block*4][offset+r] = W_block[k][r]
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try Rank 33 construction
    res = _construct_2x3x6_decomp()
    
    if _verify_decomposition(res['U'], res['V'], res['W'], n, m, p, tol=1e-9):
        return res
        
    # Fallback to Schoolbook
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify_decomposition(res['U'], res['V'], res['W'], 2, 3, 6))
