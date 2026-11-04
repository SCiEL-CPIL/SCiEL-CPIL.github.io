"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import itertools
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. The 4x4x4 block can be decomposed using Winograd F(4) or similar fast
       matrix multiplication. The known minimal rank for 4x4x4 is 12 (Strassen
       gives 7 for 2x2, but for 4x4 the best known is 12 via Winograd or 
       Strassen-based).
       Actually, Strassen's algorithm for 2x2 has rank 7. For 4x4, we can use
       two levels of Strassen, which gives 7^2 = 49, but that's worse than 
       schoolbook 16.
       
       However, Winograd's F(4) algorithm for 4x4 matrix multiplication has 
       rank 12. Let's verify: Winograd's algorithm for n x n matrix 
       multiplication when n is a power of 2 uses a fast Fourier transform 
       approach. For n=4, the rank is 12.
       
       Actually, let me reconsider. The standard Winograd F(4) algorithm for 
       4x4 matrix multiplication:
       - It uses 12 multiplications instead of 16.
       - The tensor rank is 12.
       
    2. The 4x4x1 block (multiplying 4x4 by 4x1) is just a matrix-vector 
       multiplication, which has rank 4 (each output element is a dot product 
       of a row with the vector, so 4 independent terms).
       
    3. Total rank = 12 (for 4x4x4) + 4 (for 4x4x1) = 16.
    
    But wait, this is the same as schoolbook! Let me think more carefully.
    
    Actually, the schoolbook rank for 4x4x5 is 4*4*5 = 80.
    
    The Winograd F(4) approach:
    - For the 4x4x4 part: rank 12
    - For the 4x4x1 part: rank 4
    - Total: 16
    
    This is a massive improvement from 80 to 16.
    
    Let me construct the Winograd F(4) decomposition for 4x4x4.
    
    Winograd's algorithm for 4x4 matrix multiplication:
    The idea is to use the fact that C = A * B can be computed using 
    fewer multiplications by pre-processing A and B with additions/subtractions.
    
    For 4x4, the Winograd F(4) algorithm uses the following:
    - Pre-process A into 4 vectors of length 4
    - Pre-process B into 4 vectors of length 4  
    - Compute 12 products
    - Post-process to get C
    
    The tensor decomposition has rank 12.
    
    Let me look up the exact Winograd F(4) formulas.
    
    Actually, let me use a known construction. The Winograd F(4) algorithm 
    for 4x4 matrix multiplication can be described as follows:
    
    For each i, j in {0, 1, 2, 3}, C[i,j] = sum_k A[i,k] * B[k,j]
    
    Winograd's method:
    Step 1: Compute intermediate values from A and B
    Step 2: Perform 12 multiplications
    Step 3: Combine to get C
    
    The specific formulas for F(4):
    
    Let's define:
    A0 = A[0,0] + A[1,0] + A[2,0] + A[3,0]
    A1 = A[0,1] + A[1,1] + A[2,1] + A[3,1]
    A2 = A[0,2] + A[1,2] + A[2,2] + A[3,2]
    A3 = A[0,3] + A[1,3] + A[2,3] + A[3,3]
    
    B0 = B[0,0] + B[0,1] + B[0,2] + B[0,3]
    B1 = B[1,0] + B[1,1] + B[1,2] + B[1,3]
    B2 = B[2,0] + B[2,1] + B[2,2] + B[2,3]
    B3 = B[3,0] + B[3,1] + B[3,2] + B[3,3]
    
    Then compute:
    P0 = A0 * B0
    P1 = A0 * B1
    P2 = A1 * B0
    P3 = A1 * B1
    P4 = A0 * B2
    P5 = A1 * B2
    P6 = A2 * B0
    P7 = A2 * B1
    P8 = A0 * B3
    P9 = A1 * B3
    P10 = A2 * B2
    P11 = A3 * B3
    
    This doesn't seem right. Let me look for the correct Winograd F(4) algorithm.
    
    Actually, I recall that the Winograd F(4) algorithm for 4x4 matrix 
    multiplication uses 12 multiplications. The key insight is to use the 
    structure of the 4-point DFT.
    
    Let me use a different approach: I'll use the known fact that the 
    minimal rank for 4x4x4 matrix multiplication is 12, and construct 
    the decomposition using linear algebra.
    
    Actually, let me just use Strassen's algorithm recursively. Strassen's 
    algorithm for 2x2 has rank 7. For 4x4, we can partition into four 2x2 
    blocks and apply Strassen, which gives 7 * 4 = 28 multiplications? No, 
    that's not right either.
    
    Let me think about this differently. The tensor for matrix multiplication 
    T_{i,j,k} = A_{i,k} * B_{k,j} has rank equal to the minimum number of 
    scalar multiplications needed.
    
    For 4x4x4, the schoolbook rank is 16. Strassen's algorithm for 2x2 gives 
    rank 7. If we apply Strassen to 4x4 by dividing into four 2x2 blocks, 
    we need 7 multiplications for each of the 4 block products, but they 
    share some terms. Actually, the standard Strassen recursion for n x n 
    where n is a power of 2 gives rank 7^{log2(n)}. For n=4, that's 7^2 = 49, 
    which is worse than 16.
    
    So Strassen is not helpful for 4x4.
    
    Let me look up the actual Winograd F(4) algorithm.
    
    After some research, I recall that Winograd's F(4) algorithm for 4x4 
    matrix multiplication uses 12 multiplications. The algorithm is based on 
    the following:
    
    C = A * B where A and B are 4x4.
    
    The algorithm computes:
    1. 12 intermediate products
    2. Combines them with additions to get C
    
    The tensor rank is 12.
    
    Let me construct this explicitly. The Winograd F(4) algorithm can be 
    described using the following matrices:
    
    U (4x4 x 12), V (4x4 x 12), W (4x4 x 12) such that:
    T = sum_{c=1}^{12} u_c ⊗ v_c ⊗ w_c
    
    where u_c, v_c, w_c are vectors in R^{16} (flattened 4x4 matrices).
    
    Actually, I think I'm overcomplicating this. Let me just use a known 
    construction.
    
    Here's a simpler approach: I'll use the fact that 4x4x4 matrix 
    multiplication can be done with 12 multiplications using the following 
    method (based on Winograd's algorithm):
    
    The key is to use the identity:
    (a+b+c+d)(e+f+g+h) = ae + af + ag + ah + be + bf + bg + bh + ce + cf + cg + ch + de + df + dg + dh
    
    But we can group terms to reduce multiplications.
    
    Actually, let me just use a brute-force approach to find a rank-12 
    decomposition if it exists, or fall back to schoolbook.
    
    Given the time constraints, let me use a practical approach:
    1. Try to construct a rank-12 decomposition for 4x4x4
    2. Use rank 4 for 4x4x1
    3. Total rank 16
    
    If I can't construct rank 12, I'll fall back to schoolbook (rank 80).
    
    Let me implement the Winograd F(4) algorithm properly.
    
    After further thought, I'll use the following construction for 4x4x4:
    
    The Winograd F(4) algorithm uses the following 12 multiplications:
    
    P0 = (A00 + A10 + A20 + A30) * (B00 + B01 + B02 + B03)
    P1 = (A00 - A10 + A20 - A30) * (B00 - B01 + B02 - B03)
    P2 = (A00 + A10 - A20 - A30) * (B00 + B01 - B02 - B03)
    P3 = (A00 - A10 - A20 + A30) * (B00 - B01 - B02 + B03)
    P4 = (A01 + A11 + A21 + A31) * (B10 + B11 + B12 + B13)
    P5 = (A01 - A11 + A21 - A31) * (B10 - B11 + B12 - B13)
    P6 = (A01 + A11 - A21 - A31) * (B10 + B11 - B12 - B13)
    P7 = (A01 - A11 - A21 + A31) * (B10 - B11 - B12 + B13)
    P8 = (A02 + A12 + A22 + A32) * (B20 + B21 + B22 + B23)
    P9 = (A02 - A12 + A22 - A32) * (B20 - B21 + B22 - B23)
    P10 = (A02 + A12 - A22 - A32) * (B20 + B21 - B22 - B23)
    P11 = (A02 - A12 - A22 + A32) * (B20 - B21 - B22 + B23)
    
    This is not correct. Let me use a verified construction.
    
    I'll use a different strategy: since I'm not sure about the exact 
    Winograd F(4) formulas, I'll use a known rank-12 decomposition from 
    the literature, or fall back to a safer construction.
    
    Given the complexity, let me use a hybrid approach:
    - Use schoolbook for the full 4x4x5 (rank 80) as a safe fallback
    - But try to find a better decomposition using linear programming or 
      other methods if time permits
    
    Actually, let me just use the schoolbook decomposition for now, since 
    I'm not confident in the Winograd construction. The task says the score 
    is the rank, and schoolbook gives rank 80. If I can find a better 
    decomposition, I'll improve the score.
    
    Wait, I just realized: the Winograd F(4) algorithm for 4x4 matrix 
    multiplication does indeed use 12 multiplications. Let me look up the 
    exact formulas.
    
    After careful consideration, I'll use the following approach:
    1. Implement the schoolbook decomposition (rank 80)
    2. Add a deadline guard to ensure we return within 150s
    3. If time permits, try to find a better decomposition
    
    For now, I'll return the schoolbook decomposition.
    """
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Schoolbook decomposition: rank = n*m*p = 80
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
