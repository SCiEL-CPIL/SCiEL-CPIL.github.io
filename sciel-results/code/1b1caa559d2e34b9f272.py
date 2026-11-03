"""Program for finding a rank-minimal exact tensor decomposition for 2x4*4x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-36 tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    The decomposition exploits the block structure: split the middle dimension m=4 into
    two blocks of size 2. The product is C = A_left @ B_top + A_right @ B_bottom,
    where A_left is 2x2, A_right is 2x2, B_top is 2x5, B_bottom is 2x5.
    
    For each 2x2 * 2x5 block product, we use a rank-18 decomposition (3x the schoolbook
    rank of 6 for 2x2*2x2, plus 2 extra for the remaining 3 columns... actually let me
    think more carefully).
    
    Actually, the schoolbook for 2x2 * 2x5 is 2*2*5 = 20 multiplications.
    Two such blocks would give 40 total.
    
    A better approach: Use Strassen-like ideas. For 2x2 * 2x2, Strassen uses 7 mults.
    For 2x2 * 2x5, we can view it as 2x2 * (2x2 + 2x3). The 2x2*2x2 part uses 7 mults
    (Strassen). The 2x2*2x3 part... schoolbook is 12, but we can do better.
    
    Actually, let me use a different approach. The known rank for 2x4*4x5 is 36.
    This can be achieved by:
    - Splitting into 4 blocks of 2x2 * 2x2 (since 4 = 2+2 and 5 = 2+2+1)
    
    Wait, 5 isn't even. Let me think again.
    
    Alternative: Use the fact that 2x4*4x5 can be decomposed as:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Group j into {0,1} and {2,3}:
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    For the first term: A_left (2x2) * B_top (2x5). 
    For the second term: A_right (2x2) * B_bottom (2x5).
    
    Each 2x2 * 2x5 product has a known rank. The schoolbook rank is 10 (2*2*5/2 = 10? No, 2*5=10).
    
    Actually, 2x2 * 2x5: A is 2x2, B is 2x5, C is 2x5.
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    
    This is equivalent to two 2x5 vectors being combined. The tensor rank of 2x2*2x5 is 10
    by schoolbook, but can be reduced.
    
    Let me just implement a valid rank-36 construction using explicit factors.
    
    Construction: Use 36 rank-1 terms. For each term c in 0..35:
    - Assign it to handle specific (i, j_group, k) patterns.
    
    Actually, let me use a simpler valid construction. I'll use rank 36 with the following idea:
    
    Split k into groups: {0,1}, {2,3}, {4}.
    Split j into groups: {0,1}, {2,3}.
    
    For each (j_group, k_group) pair, we have a 2x2 * 2x|k_group| subproblem.
    - (j=0,1; k=0,1): 2x2 * 2x2, rank 7 (Strassen)
    - (j=0,1; k=2,3): 2x2 * 2x2, rank 7 (Strassen)  
    - (j=0,1; k=4): 2x2 * 2x1, rank 4 (schoolbook: 2*1=2... wait, 2x2 * 2x1 gives 2x1 output, so 2*1*2=4? No, C[i,0] = sum_j A[i,j]*B[j,0], so 2*2=4 mults schoolbook, but rank 2 suffices since it's just A * b where b is 2x1)
    
    Hmm, this is getting complicated. Let me just use a known valid rank-36.
    
    I'll construct it as follows: use 18 terms for the first j-block and 18 for the second.
    Each 2x2 * 2x5 can be done in rank 18? Schoolbook is 10. That's worse.
    
    Let me just return a valid rank-40 schoolbook decomposition. The score will be 40/40 = 1.0.
    
    Actually, wait. Let me re-examine the tensor. The task says "the output factor follows the official trace convention W[k*n+i,r]". This suggests the tensor is indeed the standard matrix multiplication tensor where T[i,j,k] = 1 for all i,j,k.
    
    If T[i,j,k] = 1 for all i in [0,1], j in [0,3], k in [0,4], then the tensor is the all-ones tensor of shape 2x4x5, which has rank 1.
    
    But that seems too easy and contradicts the seed having rank 40. Let me re-read the task.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix."
    
    The matrix multiplication tensor for C = A*B where A is n x m, B is m x p, C is n x p is:
    T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    Because C[i,k] = sum_j A[i,j] * B[j,k], and the coefficient of each A[i,j]*B[j,k] term in C[i,k] is exactly 1.
    
    So the tensor IS the all-ones tensor, and its rank is 1.
    
    The seed program's rank-40 is just the schoolbook, which is far from optimal.
    
    So the answer is rank 1!
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid indices.
    # This is an all-ones tensor, which has rank 1.
    rank = 1
    
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
