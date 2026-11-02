"""Matrix multiplication 4x4 x 4x8 tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Use Kronecker product structure.
    4 = 2*2, 8 = 2*4
    The tensor for (2x2)*(2x2) has rank 7 (Strassen).
    We can decompose 4x4*4x8 as a Kronecker product of:
      - 2x2 * 2x4 (rank 7 for the 2x2 part, but need to handle the 4 in output)
    
    Actually, let me think more carefully.
    
    Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    where A is 4x4, B is 4x8, C is 4x8.
    
    The tensor T[i,j,k] = e_i ⊗ e_j ⊗ e_k^T (in appropriate basis)
    
    For Kronecker product approach:
    If we write indices as:
      i = i1*2 + i2  (i1, i2 in {0,1})
      j = j1*2 + j2  (j1, j2 in {0,1})  
      k = k1*4 + k2  (k1 in {0,1}, k2 in {0,1,2,3})
    
    Then C[i1*2+i2, k1*4+k2] = sum_{j1,j2} A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k1*4+k2]
    
    This doesn't factor nicely into a simple Kronecker product because B has different dimensions.
    
    Let me try a different approach: use the known rank bounds.
    
    For n x m * m x p, the schoolbook rank is n*m*p = 4*4*8 = 128.
    
    Known results:
    - 2x2*2x2 has rank 7 (Strassen)
    - 2x2*2x3 has rank ? 
    - 4x4*4x4 has rank 49 (Landsberg)
    
    For 4x4*4x8, we can think of it as two 4x4*4x4 multiplications, so rank <= 2*49 = 98.
    But we might do better.
    
    Actually, let's use the Kronecker product more carefully.
    
    Consider A as a 2x2 block matrix where each block is 2x2:
    A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    
    B as a 2x4 block matrix where each block is 2x4:
    B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each Bji is 2x4
    
    Then C = A*B is a 2x4 block matrix:
    C[i1,j1] = sum_{j2} A[i1,j2] * B[j2,j1]
    
    This is getting complex. Let me try a simpler approach: just use Strassen on the 2x2 blocks.
    
    Actually, the cleanest approach: 
    4x4 * 4x8 can be seen as:
    - Split A into 2x2 blocks of 2x2: A = [[A0, A1], [A2, A3]]
    - Split B into 2x4 blocks of 2x4: B = [[B0, B1, B2, B3], [B4, B5, B6, B7]]
    
    C[0,:] = A0*B0 + A1*B4
    C[1,:] = A0*B1 + A1*B5
    C[2,:] = A0*B2 + A1*B6
    C[3,:] = A0*B3 + A1*B7
    C[4,:] = A2*B0 + A3*B4
    C[5,:] = A2*B1 + A3*B5
    C[6,:] = A2*B2 + A3*B6
    C[7,:] = A2*B3 + A3*B7
    
    Wait, I'm confusing myself with the indexing. Let me be more careful.
    
    A is 4x4, so rows 0-3, cols 0-3.
    B is 4x8, so rows 0-3, cols 0-7.
    C is 4x8, so rows 0-3, cols 0-7.
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Let me split i = 2*i1 + i2, j = 2*j1 + j2, k = 4*k1 + k2 where:
    i1, i2 in {0,1}
    j1, j2 in {0,1}
    k1 in {0,1}, k2 in {0,1,2,3}
    
    C[2*i1+i2, 4*k1+k2] = sum_{j1=0}^{1} sum_{j2=0}^{1} A[2*i1+i2, 2*j1+j2] * B[2*j1+j2, 4*k1+k2]
    
    This is still not a clean Kronecker product.
    
    Let me just try to find a good decomposition empirically or use known results.
    
    For now, let's use a rank-128 schoolbook decomposition as the baseline, and try to improve.
    
    Actually, let me think about this differently. The task says "rank-minimal exact tensor decomposition". 
    The score is the rank, so we want to minimize the rank.
    
    Known lower bounds:
    - For 2x2*2x2, rank >= 7
    - For 4x4*4x4, rank >= 49 (Landsberg's bound)
    
    For 4x4*4x8, we can use the fact that it contains 4x4*4x4 as a sub-problem, so rank >= 49.
    But we might be able to do better than 128.
    
    Let me try a rank-56 decomposition using Strassen on 2x2 blocks.
    
    Actually, I'll implement a constructive approach:
    Use the Strassen algorithm for 2x2 matrices, applied to the 2x2 blocks.
    
    For 4x4 * 4x4:
    Split into 2x2 blocks. Use Strassen: 7 multiplications of 2x2*2x2.
    Each 2x2*2x2 uses 7 scalar multiplications.
    Total: 7*7 = 49.
    
    For 4x4 * 4x8:
    We can split B into two 4x4 matrices: B = [B_left, B_right]
    Then C = [A*B_left, A*B_right]
    Each A*B_left and A*B_right is 4x4*4x4, rank 49 each.
    Total rank: 49 + 49 = 98.
    
    But we might do better by not splitting B this way.
    
    Alternatively, split A and B differently.
    
    Let me just implement the rank-98 decomposition for now, as it's a clear improvement over 128.
    
    Actually, I realize I should just return a valid decomposition. Let me use the schoolbook for simplicity and correctness, then try to optimize.
    
    For the schoolbook:
    U[i*m+j, c] = 1 if c corresponds to (i,j,k)
    V[j*p+k, c] = 1 if c corresponds to (i,j,k)
    W[k*n+i, c] = 1 if c corresponds to (i,j,k)
    
    where c = i*m*p + j*p + k
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
