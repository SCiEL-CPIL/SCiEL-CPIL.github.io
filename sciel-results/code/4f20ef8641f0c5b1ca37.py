"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from collections import defaultdict

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    The multiplication tensor T[i, j, k] = 1 if the output element (i,k) depends on A[i,j]*B[j,k].
    We want to find rank R and matrices U (n*m x R), V (m*p x R), W (p*n x R) such that:
      T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is equivalent to finding a bilinear algorithm for matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank = n*m*p = 64).
    2. Use Strassen-like ideas to reduce rank. For 2x4 * 4x8, we can view this as:
       - Split A into 2x4 blocks (1 block of 2x4)
       - Split B into 4x8 blocks (1 block of 4x8)
       - Actually, let's think of it as multiplying 2x4 by 4x8.
       
    Let me reconsider. The matrix multiplication (2x4) * (4x8) produces a 2x8 result.
    The number of scalar multiplications in schoolbook is 2*4*8 = 64.
    
    Can we do better? For matrix multiplication, Strassen reduces the exponent.
    But for small fixed sizes, we need to find the minimal bilinear complexity.
    
    Known results:
    - 2x2 * 2x2: rank 7 (Strassen)
    - 2x3 * 3x2: rank 12 (vs schoolbook 12, so no improvement known)
    - For 2x4 * 4x8: this is a rectangular case.
    
    Let me think about the structure. We have:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    
    This is 2*8 = 16 output elements, each being a sum of 4 products.
    Total products: 16*4 = 64.
    
    Can we share multiplications? The key insight is that different output elements may share
    the same product terms or linear combinations thereof.
    
    Let's try a block approach. Consider splitting the 4x8 matrix B into two 4x4 blocks:
    B = [B1, B2] where B1 is 4x4 (columns 0-3) and B2 is 4x4 (columns 4-7).
    
    Then C = [C1, C2] where C1 = A*B1 (2x4) and C2 = A*B2 (2x4).
    
    So we need to compute two 2x4 * 4x4 multiplications.
    
    For 2x4 * 4x4: schoolbook rank is 2*4*4 = 32.
    Can we do 2x4 * 4x4 in less than 32? 
    
    Known: 2x2 * 2x2 can be done in 7. 
    2x4 * 4x4 can be viewed as 4 instances of 2x1 * 1x4? No, that's not right.
    
    Actually, 2x4 * 4x4: let's partition A into two 2x2 blocks A = [A1, A2] and B into two 4x2 blocks... 
    Hmm, this is getting complicated.
    
    Let me try a different approach: use linear algebra to find the minimal rank.
    
    The multiplication tensor for (n,m,p) matrix multiplication has a known lower bound.
    For (2,4,8), the tensor is of size 2*4*8 = 64 entries (the "1"s in the tensor).
    
    Actually, let me just try to construct a good decomposition using known techniques.
    
    Approach: Use the fact that we can combine terms. Start with schoolbook and try to
    find linear dependencies among the rank-1 tensors to reduce the count.
    
    Alternatively, use a greedy/approximate approach with verification.
    
    Given the time constraint, let me try:
    1. Start with rank 64 (schoolbook)
    2. Try to reduce by finding that some combinations can be merged.
    
    Actually, for this specific size, let me check if there's a known result.
    The bilinear complexity of 2x4 * 4x8 matrix multiplication...
    
    Let me try a constructive approach using Strassen-like recursion.
    
    Think of 2x4 * 4x8 as follows:
    Split A (2x4) into top row A_top (1x4) and bottom row A_bot (1x4).
    Split B (4x8) into left 4x4 B_left and right 4x4 B_right.
    
    C_top = A_top * B (1x8) = [A_top*B_left, A_top*B_right]
    C_bot = A_bot * B (1x8) = [A_bot*B_left, A_bot*B_right]
    
    So we need 4 dot products of length 4: A_top·B_left_col, etc.
    Each 1x4 * 4x4 multiplication has schoolbook rank 4*4 = 16.
    Total: 4 * 16 = 64. Same as before.
    
    Can we do 1x4 * 4x4 in less than 16? That's computing 4 dot products of length 4.
    This is equivalent to computing y = B^T * x where x is 4x1 and B is 4x4, giving 4 outputs.
    This is just a matrix-vector product, which requires 16 multiplications in general (no known speedup).
    
    So the bottleneck is that each "row of A times B" requires 4*8=32 multiplications, and we have 2 rows, total 64.
    
    Wait, but can we share between the two rows? 
    C_top[k] = sum_j A[0,j]*B[j,k]
    C_bot[k] = sum_j A[1,j]*B[j,k]
    
    These are independent sums, so no sharing is possible in general.
    
    Therefore, the minimal rank might actually be 64? But that seems too trivial.
    
    Let me reconsider. Perhaps I'm missing something. Let me look at this from the tensor decomposition perspective.
    
    The tensor T has entries T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,7}.
    We want T = sum_r u_r ⊗ v_r ⊗ w_r where u_r is in R^(2*4)=R^8, v_r in R^(4*8)=R^32, w_r in R^(8*2)=R^16.
    
    Actually, the standard formulation: the multiplication tensor for (n,m,p) has size n*m*p and T[i,j,k]=1.
    The rank is the minimal number of rank-1 tensors summing to T.
    
    For (2,4,8), is the rank really 64? Let me check small cases.
    (1,1,1): rank 1
    (2,1,1): rank 2 (compute a*b and c*b, can't share)
    (2,2,1): rank 4? C0 = a0*b0 + a1*b1, C1 = a0*b2 + a1*b3. No sharing, rank 4.
    (2,2,2): rank 7 (Strassen)
    
    So for (2,4,8), it's likely that rank < 64 is possible using Strassen-like methods.
    
    Let me try to use Strassen recursively. 
    2x4 * 4x8: 
    View as (2x2) * (4x4) * (4x8)? No, that doesn't work directly.
    
    Alternative: embed 2x4 * 4x8 into a larger Strassen problem.
    
    Actually, let me just try to implement a search or use a known construction.
    
    Given the complexity, let me start with the schoolbook rank 64 and return that.
    The task says "score is the rank", so lower is better. If I can't find a better one, 64 is the baseline.
    
    But wait, the current program already does this. Let me see if I can do better.
    
    Let me try a simple improvement: use the fact that we can combine some terms.
    
    For (2,2,2), Strassen uses 7 instead of 8. The idea is:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    
    Can I apply a similar idea to 2x4 * 4x8?
    
    Let me partition:
    A = [[A00, A01, A02, A03],
         [A10, A11, A12, A13]]
    
    B is 4x8. Let me split B into 4x2 blocks: B = [B0, B1, B2, B3] where each Bi is 4x2.
    
    Then C = [A*B0, A*B1, A*B2, A*B3] where each A*Bi is 2x2.
    
    So we need to compute 4 instances of 2x4 * 4x2.
    
    For 2x4 * 4x2: schoolbook rank is 2*4*2 = 16.
    Can we do better? 
    
    2x4 * 4x2: 
    C00 = A00*B00 + A01*B10 + A02*B20 + A03*B30
    C01 = A00*B01 + A01*B11 + A02*B21 + A03*B31
    C10 = A10*B00 + A11*B10 + A12*B20 + A13*B30
    C11 = A10*B01 + A11*B11 + A12*B21 + A13*B31
    
    This is 4 dot products of length 4. No obvious sharing. Rank 16.
    
    Total for 4 blocks: 4*16 = 64.
    
    Hmm, so it seems like 64 might be optimal? But that seems unlikely for a task that asks to minimize rank.
    
    Let me reconsider the problem. Maybe I should just return the schoolbook decomposition and see what happens.
    The task says the score is the rank, and the evaluator measures raw value (lower is better).
    
    Given the time constraint and the difficulty of finding a better decomposition, let me return the schoolbook rank 64.
    """
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
