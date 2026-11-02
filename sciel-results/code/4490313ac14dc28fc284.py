"""Program to find rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a bilinear decomposition for 4x5 * 5x5 matrix multiplication.
    Uses 2x2 Strassen on the 4x4 core to reduce rank below schoolbook 100.
    
    Partition:
      A (4x5) = [A_left (4x4) | A_right (4x1)]
      B (5x5) = [B_top (4x5); B_bot (1x5)]
      
      C = A_left @ B_top + A_right @ B_bot
      
    For A_left @ B_top:
      A_left is 4x4, B_top is 4x5.
      Split A_left into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
      Split B_top into 2x2 and 2x1 blocks: B11, B12 (2x2), B13, B23 (2x1)
      
      Use Strassen-like approach for the 2x2 * 2x2 parts and schoolbook for 2x2 * 2x1.
      
    Total rank: 7 (Strassen for 4x4*4x2 part) + 4 (schoolbook for 4x4*1) 
              + 4 (schoolbook for 4x1*4) + 1 (schoolbook for 4x1*1)
              = 7 + 4 + 4 + 1 = 16? No, that's not right.
              
    Let me be more careful.
    
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    
    Part 1: sum_{j=0}^{3} A[i,j] B[j,k] for i in 0..3, k in 0..4
    This is a 4x4 times 4x5 multiplication.
    
    Split i into i1 (0,1) and i2 (2,3), and j into j1 (0,1) and j2 (2,3).
    
    C[i1,k] = sum_{j1} A[i1,j1] B[j1,k] + sum_{j2} A[i1,j2] B[j2,k]
    C[i2,k] = sum_{j1} A[i2,j1] B[j1,k] + sum_{j2} A[i2,j2] B[j2,k]
    
    For the k dimension (0..4), split k into k1 (0,1) and k2 (2,3) and k3 (4).
    
    So we have:
    C[i1,k1] = A11*B11 + A12*B21  (2x2 * 2x2)
    C[i1,k2] = A11*B12 + A12*B22  (2x2 * 2x2)  
    C[i1,k3] = A11*B13 + A12*B23  (2x2 * 2x1)
    C[i2,k1] = A21*B11 + A22*B21  (2x2 * 2x2)
    C[i2,k2] = A21*B12 + A22*B22  (2x2 * 2x2)
    C[i2,k3] = A21*B13 + A22*B23  (2x2 * 2x1)
    
    For each 2x2 * 2x2, we can use Strassen (rank 7 instead of 4).
    For each 2x2 * 2x1, we need rank 2 (schoolbook).
    
    But wait, Strassen gives us a way to compute all four 2x2*2x2 blocks together
    if they share the same A and B blocks. Here, the A blocks are A11, A12, A21, A22
    and B blocks are B11, B12, B21, B22, B13, B23.
    
    Actually, let me think differently. The 4x4 * 4x5 can be seen as:
    - 4x4 * 4x2 (for k=0,1,2,3) which can be done with two independent 2x2 Strassen
      (one for columns 0,1 and one for columns 2,3), each using rank 7.
    - 4x4 * 2x1 (for k=4) which uses rank 4 (schoolbook: 4 terms).
    
    Wait, that's not quite right either. Let me think about it as:
    
    The 4x4 * 4x5 multiplication has rank at most 4*4*5 = 80 by schoolbook.
    Using Strassen on the 4x4 * 4x2 part: we can view this as two independent
    4x4 * 2x2 multiplications? No, that's not standard.
    
    Let me use a cleaner approach:
    
    View 4x4 * 4x5 as:
    - Multiply 4x4 by first 4 columns of B (4x4 result): use 2x2 Strassen, rank 7
    - Multiply 4x4 by last 1 column of B (4x1 result): schoolbook, rank 4
    
    Total for A_left @ B_top: 7 + 4 = 11
    
    For A_right @ B_bot (4x1 * 1x5 = 4x5): schoolbook, rank 4
    
    Total: 11 + 4 = 15
    
    That's a huge improvement from 100!
    
    Let me verify the Strassen part more carefully.
    
    4x4 * 4x4 using 2x2 Strassen:
    A = [[A11, A12], [A21, A22]] (each 2x2)
    B = [[B11, B12], [B21, B22]] (each 2x2)
    
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2 * 2x2 product, which itself requires 4 scalar multiplications
    (or 7 if we use Strassen recursively, but let's use schoolbook for 2x2).
    
    Wait, no. In the tensor decomposition, each "product" Mi corresponds to
    rank-4 terms (since it's a 2x2*2x2 multiplication). So the total rank would be
    7 * 4 = 28 for the 4x4*4x4 part? That's worse than schoolbook 64.
    
    I'm confusing myself. Let me restart with a clearer model.
    
    In the tensor decomposition, each term is a rank-1 tensor:
    U[:,t] * V[:,t] * W[:,t]
    
    For schoolbook 4x5 * 5x5:
    For each (i,j,k) triple, we have one term:
    U[i*5+j, t] = 1
    V[j*5+k, t] = 1  
    W[k*4+i, t] = 1
    
    Total: 4*5*5 = 100 terms.
    
    For Strassen 2x2 * 2x2:
    The 7 terms are combinations. Each term involves sums/differences of entries.
    For example, M1 = (A11+A22)(B11+B22) expands to:
    A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    
    So each of the 7 "products" actually corresponds to 4 scalar multiplications
    in the tensor decomposition. Total: 7*4 = 28 terms for 2x2*2x2.
    
    But schoolbook 2x2*2x2 is only 4 terms! So Strassen increases the rank?
    
    No, I'm misunderstanding. In the tensor decomposition sense, Strassen
    reduces the number of independent multiplications, but each "multiplication"
    in Strassen is still a product of two linear combinations, which in the
    bilinear form corresponds to multiple rank-1 terms.
    
    Actually, I think the issue is that Strassen's algorithm reduces the number
    of scalar multiplications, but in the tensor decomposition framework, what
    matters is the rank of the tensor. For 2x2*2x2, the tensor rank is 7 (Strassen
    proved this is optimal), while the "naive" decomposition has rank 4*4=16? No,
    schoolbook 2x2*2x2 has rank 4 (one term per output element, each being a sum
    of 2 products).
    
    I'm getting confused. Let me just implement the schoolbook decomposition
    correctly and verify it works, then try to find a valid reduction.
    
    For now, let me return the correct schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 110
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
