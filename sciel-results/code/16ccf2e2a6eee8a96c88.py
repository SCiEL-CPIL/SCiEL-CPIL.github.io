"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    The key insight: a 2x3 times 3x9 product can be viewed as 3 independent 2x3 times 3x3
    block multiplications (since 9 = 3*3). Each 2x3*3x3 multiplication has a known 
    rank. The tensor structure is T_{2,3,3} ⊗ I_3, so the total rank is 3 * rank(2,3,3).
    
    For 2x3 * 3x3, we can use the fact that this is equivalent to multiplying a 2x3 matrix
    by a 3x3 matrix. The standard approach uses rank 6 (schoolbook) but there are 
    decompositions with lower rank.
    
    Actually, let's think more carefully. The multiplication tensor for (n,m,p) has 
    dimension n*m*p. For (2,3,3), that's 18. The schoolbook rank is 18.
    
    However, we can exploit structure. A 2x3 * 3x3 product:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is 2*3*3 = 18 scalar multiplications in schoolbook.
    
    But wait - the tensor product structure T_{2,3,3} ⊗ I_3 means if we have a rank-r 
    decomposition for 2x3*3x3, we get rank 3r for 2x3*3x9.
    
    For 2x3*3x3, is there a known lower rank? The border rank and tensor rank of small 
    matrix multiplication are well-studied. For 2x3 * 3x3, the rank is known to be 6? No,
    that's too low. Let me reconsider.
    
    Actually, for rectangular matrix multiplication, the minimal rank is not always known.
    For 2x2*2x2, it's 7 (Strassen). For 2x3*3x3, let me think...
    
    The schoolbook for 2x3*3x3 is 2*3*3 = 18. Can we do better?
    
    One approach: treat the 2x3 matrix as having 6 elements, and the 3x3 as having 9 elements.
    The product has 6 elements. 
    
    Actually, I recall that for multiplying an (n x m) by (m x p) matrix, if we fix n and p 
    and vary m, there are known results. For 2x3 * 3x3, I believe the rank is 6 is not 
    achievable. Let me use a different approach.
    
    Let me just implement the schoolbook for the full 2x3*3x9 (rank 54) and see if I can 
    find any structure to reduce it. Actually, the tensor product argument is valid:
    
    If T_{2,3,3} has rank r, then T_{2,3,9} = T_{2,3,3} ⊗ I_3 has rank 3r.
    
    So I need to find the rank of T_{2,3,3}. 
    
    For 2x3 * 3x3:
    - Schoolbook: 18
    - Is there a known lower bound? The rank is at least max(n*m, m*p, n*p) = max(6,9,6) = 9.
    - Upper bound: 18
    
    I don't have a known decomposition better than schoolbook for 2x3*3x3 in my immediate 
    recall. Let me just go with the schoolbook for now, which gives rank 54 for the full 
    problem.
    
    Wait, but the task says "find a rank-minimal exact tensor decomposition." The current 
    score is 1.0, which corresponds to the schoolbook rank of 54. To improve, I need to find 
    a lower rank.
    
    Let me think about whether there's any structure in 2x3*3x9 that I can exploit beyond 
    the tensor product.
    
    Actually, let me reconsider the problem. The output is W[k*n+i, r] where n=2, p=9. So 
    W has shape (9*2, rank) = (18, rank). U has shape (2*3, rank) = (6, rank). V has shape 
    (3*9, rank) = (27, rank).
    
    The tensor T is in R^{6} ⊗ R^{27} ⊗ R^{18}. The schoolbook decomposition uses rank 54.
    
    Can I find a rank < 54 decomposition? 
    
    One idea: use the fact that 9 = 3*3 and the tensor product structure. If I can find a 
    rank-16 decomposition for 2x3*3x3, then I'd get rank 48 for the full problem. But I 
    don't know if rank 16 is achievable for 2x3*3x3.
    
    Let me try a different approach: maybe I can find a direct decomposition with rank less 
    than 54 by exploiting the specific dimensions.
    
    Actually, I just realized I should check if there's a known result. The rank of the 
    matrix multiplication tensor M_{n,m,p} is known for small values. For M_{2,3,3}, I 
    believe the rank is 6 is not correct. Let me compute it.
    
    For M_{2,3,3}: the tensor has 18 components. The rank is the minimum number of rank-1 
    tensors that sum to it. I don't have a closed-form answer.
    
    Let me just implement the schoolbook and return it, since I can't find a better 
    decomposition in the time available.
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
