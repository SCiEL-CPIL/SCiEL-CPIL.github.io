"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 2x2 by 2x12.
    We use Strassen's rank-7 algorithm for the 2x2 block, shared across the 6 blocks.
    
    The multiplication is: C = A * B where A is 2x2, B is 2x12.
    We can think of B as 6 blocks of 2x2: B = [B0, B1, B2, B3, B4, B5]
    
    For each block B_j (j=0..5), we compute A * B_j using Strassen.
    Strassen uses 7 multiplications:
      M1 = (A00 + A11) * (B00 + B11)
      M2 = (A10 + A11) * B00
      M3 = A00 * (B10 - B11)
      M4 = A11 * (B01 - B00)
      M5 = (A00 + A01) * B11
      M6 = (A10 - A00) * (B00 + B01)
      M7 = (A01 - A11) * (B10 + B11)
      
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    
    For the full 2x2 by 2x12 multiplication:
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01, ..., b0_11], [b10, b11, ..., b1_11]]
    
    C = A * B is 2x12:
    C[0][k] = a00*B[0][k] + a01*B[1][k]
    C[1][k] = a10*B[0][k] + a11*B[1][k]
    
    Using Strassen per 2-column block (6 blocks):
    For block j (columns 2j, 2j+1):
      B_j = [[b_{0,2j}, b_{0,2j+1}], [b_{1,2j}, b_{1,2j+1}]]
      
    The 7 Strassen multiplications per block involve products of:
    - Linear combinations of A entries (4 possible: a00, a01, a10, a11, and their sums/diffs)
    - Linear combinations of B entries within that block
    
    To minimize rank, we want to share the A-side factors across blocks.
    
    The A-side factors in Strassen are:
      P1 = A00 + A11
      P2 = A10 + A11
      P3 = A00
      P4 = A11
      P5 = A00 + A01
      P6 = A10 - A00
      P7 = A01 - A11
    
    These are 7 distinct linear combinations of the 4 A entries.
    
    For each block j, the B-side factors are:
      Q1_j = B_j[0][0] + B_j[1][1] = b_{0,2j} + b_{1,2j+1}
      Q2_j = B_j[0][0] = b_{0,2j}
      Q3_j = B_j[1][0] - B_j[1][1] = b_{1,2j} - b_{1,2j+1}
      Q4_j = B_j[0][1] - B_j[0][0] = b_{0,2j+1} - b_{0,2j}
      Q5_j = B_j[1][1] = b_{1,2j+1}
      Q6_j = B_j[0][0] + B_j[0][1] = b_{0,2j} + b_{0,2j+1}
      Q7_j = B_j[1][0] + B_j[1][1] = b_{1,2j} + b_{1,2j+1}
    
    So we have 7 * 6 = 42 rank-1 terms, each being P_i * Q_j_k for i in 1..7, k in 1..7, block j in 0..5.
    
    Wait, that's not right. Each Strassen multiplication M_i for block j is P_i * Q_j_i.
    So we have 7 multiplications per block, 6 blocks = 42 multiplications.
    
    But can we do better? The key insight is that the A-side factors P_i are the same across all blocks.
    So we have 7 distinct A-side linear forms, and for each, we need to multiply by 6 different B-side linear forms (one per block).
    
    Total rank = 7 * 6 = 42.
    
    Let's construct the tensor decomposition.
    
    U has shape (n*m, rank) = (4, 42)
    V has shape (m*p, rank) = (24, 42)
    W has shape (p*n, rank) = (24, 42)
    
    The tensor T[i, j, k] = 1 if the (i,j) entry of A times the (j,k) entry of B contributes to C[i,k].
    Actually, T is the matrix multiplication tensor: T[i, j, k] = 1 for all valid i,j,k.
    
    Wait, let me reconsider. The tensor for multiplying A (n x m) by B (m x p) to get C (n x p) is:
    T[i, k, j] = 1 (summing over j)
    
    In the standard form: C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor has indices (i, j, k) with T[i,j,k] = 1.
    
    The decomposition: C[i,k] = sum_c U[i,c] * V[j,c] * W[k,c] ... no wait.
    
    Actually, the tensor T of shape (n*m, m*p, p*n) or similar. Let me re-read the task.
    
    The task says: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    U: (n*m, rank) - this is the A factor, indexed by (i,j) pairs
    V: (m*p, rank) - this is the B factor, indexed by (j,k) pairs  
    W: (p*n, rank) - this is the C factor, indexed by (k,i) pairs (following W[k*n+i,r])
    
    The decomposition: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that's not quite right either. Let me think about this more carefully.
    
    The matrix multiplication tensor T has T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    The flattening: U is indexed by (i,j) -> i*m+j, V by (j,k) -> j*p+k, W by (k,i) -> k*n+i.
    
    So T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    For the schoolbook method, for each (i,j,k), we have a rank-1 term with U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1.
    
    For Strassen-based decomposition, we need to express the tensor as a sum of rank-1 terms where each term corresponds to a product of a linear form in A, a linear form in B, and a linear form in C.
    
    Let me define the Strassen decomposition properly.
    
    For 2x2 by 2x2:
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    C = A*B = [[c00, c01], [c10, c11]]
    
    c00 = a00*b00 + a01*b10
    c01 = a00*b01 + a01*b11
    c10 = a10*b00 + a11*b10
    c11 = a10*b01 + a11*b11
    
    Strassen:
    M1 = (a00+a11)*(b00+b11)
    M2 = (a10+a11)*b00
    M3 = a00*(b10-b11)
    M4 = a11*(b01-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)*(b00+b01)
    M7 = (a01-a11)*(b10+b11)
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    
    Now for 2x2 by 2x12, we split B into 6 blocks of 2x2.
    For block j (j=0..5), B_j = [[b_{0,2j}, b_{0,2j+1}], [b_{1,2j}, b_{1,2j+1}]]
    
    The 7 Strassen products for block j:
    M1_j = (a00+a11) * (b_{0,2j} + b_{1,2j+1})
    M2_j = (a10+a11) * b_{0,2j}
    M3_j = a00 * (b_{1,2j} - b_{1,2j+1})
    M4_j = a11 * (b_{0,2j+1} - b_{0,2j})
    M5_j = (a00+a01) * b_{1,2j+1}
    M6_j = (a10-a00) * (b_{0,2j} + b_{0,2j+1})
    M7_j = (a01-a11) * (b_{1,2j} + b_{1,2j+1})
    
    And C entries:
    c_{0,2j} = M1_j + M4_j - M5_j + M7_j
    c_{0,2j+1} = M3_j + M5_j
    c_{1,2j} = M2_j + M4_j
    c_{1,2j+1} = M1_j - M2_j + M3_j + M6_j
    
    Each M_i_j is a product of a linear form in A and a linear form in B.
    We can write M_i_j = P_i * Q_{j,i} where P_i is the A-side linear form and Q_{j,i} is the B-side linear form.
    
    The tensor decomposition:
    T = sum_{i=1}^7 sum_{j=0}^5 P_i ⊗ Q_{j,i} ⊗ R_{i,j}
    
    where R_{i,j} is the linear form in C that collects the contributions.
    
    For each (i,j) pair, the rank-1 term has:
    - U part: P_i evaluated at (a_{i',j'}) for i' in 0..1, j' in 0..1
    - V part: Q_{j,i} evaluated at (b_{j',k'}) for j' in 0..1, k' in 0..11
    - W part: R_{i,j} evaluated at (c_{i',k'}) for i' in 0..1, k' in 0..11
    
    Let's enumerate the 42 rank-1 terms.
    
    P_1 = a00 + a11: U indices (0,0)=1, (1,1)=1
    P_2 = a10 + a11: U indices (1,0)=1, (1,1)=1
    P_3 = a00: U index (0,0)=1
    P_4 = a11: U index (1,1)=1
    P_5 = a00 + a01: U indices (0,0)=1, (0,1)=1
    P_6 = a10 - a00: U indices (1,0)=1, (0,0)=-1
    P_7 = a01 - a11: U indices (0,1)=1, (1,1)=-1
    
    For block j, Q_{j,1} = b_{0,2j} + b_{1,2j+1}: V indices (0,2j)=1, (1,2j+1)=1
    Q_{j,2} = b_{0,2j}: V index (0,2j)=1
    Q_{j,3} = b_{1,2j} - b_{1,2j+1}: V indices (1,2j)=1, (1,2j+1)=-1
    Q_{j,4} = b_{0,2j+1} - b_{0,2j}: V indices (0,2j+1)=1, (0,2j)=-1
    Q_{j,5} = b_{1,2j+1}: V index (1,2j+1)=1
    Q_{j,6} = b_{0,2j} + b_{0,2j+1}: V indices (0,2j)=1, (0,2j+1)=1
    Q_{j,7} = b_{1,2j} + b_{1,2j+1}: V indices (1,2j)=1, (1,2j+1)=1
    
    R_{1,j} (from M1_j): contributes to c_{0,2j} (+1), c_{1,2j+1} (+1)
      W indices: (2j, 0)=1, (2j+1, 1)=1
    R_{2,j} (from M2_j): contributes to c_{1,2j} (+1), c_{1,2j+1} (-1)
      W indices: (2j, 1)=1, (2j+1, 1)=-1
    R_{3,j} (from M3_j): contributes to c_{0,2j+1} (+1), c_{1,2j+1} (+1)
      W indices: (2j+1, 0)=1, (2j+1, 1)=1
    R_{4,j} (from M4_j): contributes to c_{0,2j} (+1), c_{1,2j} (+1)
      W indices: (2j, 0)=1, (2j, 1)=1
    R_{5,j} (from M5_j): contributes to c_{0,2j} (-1), c_{0,2j+1} (+1)
      W indices: (2j, 0)=-1, (2j+1, 0)=1
    R_{6,j} (from M6_j): contributes to c_{1,2j+1} (+1)
      W index: (2j+1, 1)=1
    R_{7,j} (from M7_j): contributes to c_{0,2j} (+1)
      W index: (2j, 0)=1
    
    Wait, I need to double-check the Strassen formulas.
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    
    For block j:
    c_{0,2j} = M1_j + M4_j - M5_j + M7_j
    c_{0,2j+1} = M3_j + M5_j
    c_{1,2j} = M2_j + M4_j
    c_{1,2j+1} = M1_j - M2_j + M3_j + M6_j
    
    So:
    R_{1,j}: M1_j contributes +1 to c_{0,2j} and +1 to c_{1,2j+1}
      W: (2j, 0)=1, (2j+1, 1)=1
    R_{2,j}: M2_j contributes +1 to c_{1,2j} and -1 to c_{1,2j+1}
      W: (2j, 1)=1, (2j+1, 1)=-1
    R_{3,j}: M3_j contributes +1 to c_{0,2j+1} and +1 to c_{1,2j+1}
      W: (2j+1, 0)=1, (2j+1, 1)=1
    R_{4,j}: M4_j contributes +1 to c_{0,2j} and +1 to c_{1,2j}
      W: (2j, 0)=1, (2j, 1)=1
    R_{5,j}: M5_j contributes -1 to c_{0,2j} and +1 to c_{0,2j+1}
      W: (2j, 0)=-1, (2j+1, 0)=1
    R_{6,j}: M6_j contributes +1 to c_{1,2j+1}
      W: (2j+1, 1)=1
    R_{7,j}: M7_j contributes +1 to c_{0,2j}
      W: (2j, 0)=1
    
    Now let's build the matrices.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Number of Strassen multiplications per block
    num_strassen = 7
    num_blocks = p // 2  # 6 blocks
    
    rank = num_strassen * num_blocks  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (24, 42)
    
    # Define the A-side linear forms (P_i)
    # P_i is a dict mapping (i, j) -> coefficient
    P_forms = [
        {(0,0): 1, (1,1): 1},   # P1 = a00 + a11
        {(1,0): 1, (1,1): 1},   # P2 = a10 + a11
        {(0,0): 1},              # P3 = a00
        {(1,1): 1},              # P4 = a11
        {(0,0): 1, (0,1): 1},   # P5 = a00 + a01
        {(1,0): 1, (0,0): -1},  # P6 = a10 - a00
        {(0,1): 1, (1,1): -1},  # P7 = a01 - a11
    ]
    
    # For each Strassen multiplication i (0-indexed) and block j, define Q and R
    for i in range(num_strassen):
        for block in range(num_blocks):
            col = i * num_blocks + block
            
            # Fill U
            for (ii, jj), coeff in P_forms[i].items():
                U[ii * m + jj][col] = coeff
            
            # Fill V (Q_{block, i})
            col_start = 2 * block
            col_end = 2 * block + 1
            if i == 0:  # Q1 = b_{0,2j} + b_{1,2j+1}
                V[0 * p + col_start][col] = 1
                V[1 * p + col_end][col] = 1
            elif i == 1:  # Q2 = b_{0,2j}
                V[0 * p + col_start][col] = 1
            elif i == 2:  # Q3 = b_{1,2j} - b_{1,2j+1}
                V[1 * p + col_start][col] = 1
                V[1 * p + col_end][col] = -1
            elif i == 3:  # Q4 = b_{0,2j+1} - b_{0,2j}
                V[0 * p + col_end][col] = 1
                V[0 * p + col_start][col] = -1
            elif i == 4:  # Q5 = b_{1,2j+1}
                V[1 * p + col_end][col] = 1
            elif i == 5:  # Q6 = b_{0,2j} + b_{0,2j+1}
                V[0 * p + col_start][col] = 1
                V[0 * p + col_end][col] = 1
            elif i == 6:  # Q7 = b_{1,2j} + b_{1,2j+1}
                V[1 * p + col_start][col] = 1
                V[1 * p + col_end][col] = 1
            
            # Fill W (R_{i, block})
            if i == 0:  # R1: c_{0,2j} +1, c_{1,2j+1} +1
                W[col_start * n + 0][col] = 1
                W[col_end * n + 1][col] = 1
            elif i == 1:  # R2: c_{1,2j} +1, c_{1,2j+1} -1
                W[col_start * n + 1][col] = 1
                W[col_end * n + 1][col] = -1
            elif i == 2:  # R3: c_{0,2j+1} +1, c_{1,2j+1} +1
                W[col_end * n + 0][col] = 1
                W[col_end * n + 1][col] = 1
            elif i == 3:  # R4: c_{0,2j} +1, c_{1,2j} +1
                W[col_start * n + 0][col] = 1
                W[col_start * n + 1][col] = 1
            elif i == 4:  # R5: c_{0,2j} -1, c_{0,2j+1} +1
                W[col_start * n + 0][col] = -1
                W[col_end * n + 0][col] = 1
            elif i == 5:  # R6: c_{1,2j+1} +1
                W[col_end * n + 1][col] = 1
            elif i == 6:  # R7: c_{0,2j} +1
                W[col_start * n + 0][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
