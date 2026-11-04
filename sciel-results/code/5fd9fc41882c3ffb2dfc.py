"""Matrix multiplication 3x4*4x6 - block-Strassen decomposition."""

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 3x4 * 4x6.
    
    Partition the middle dimension (4) into two blocks of size 2.
    A is 3x4 = [A1 | A2] where A1, A2 are 3x2
    B is 4x6 = [B1; B2] where B1, B2 are 2x6
    
    C = A1*B1 + A2*B2
    
    Use Strassen-like decomposition for the 2x2 block structure:
    We have two "blocks" in the middle, so we can use a 2x2 Strassen
    where each "entry" is a matrix product of (3x2)*(2x6) = 3x6.
    
    For a 2x2 block Strassen (treating the 2x2 block matrix of products):
    Let P1 = (A1 + A2) * (B1 + B2)
        P2 = A1 * (B2 - B1)
        P3 = (A2 - A1) * B1
        P4 = A2 * B2
        P5 = (A1 + A2) * B2
        P6 = (A2 - A1) * (B1 - B2)
        P7 = A1 * B1
    
    Then C = P1 - P5 + P4 + P6  ... wait, let me re-derive.
    
    Actually for 2x2 Strassen:
    C11 = P1 - P5 + P4 + P6
    C12 = P2 + P4
    C21 = P3 + P1
    C22 = P1 - P2 + P3 + P5
    
    But here our "2x2" structure is different. We have:
    C = A1*B1 + A2*B2  (a single 3x6 result)
    
    This is like computing the (1,1) and (2,2) entries of a 2x2 block product
    where the off-diagonal blocks are zero. So:
    C11 = A1*B1 (top-left)
    C22 = A2*B2 (bottom-right)
    C12 = 0
    C21 = 0
    
    Using Strassen:
    P1 = (A1+A2)*(B1+B2)
    P2 = A1*(B2-B1)
    P3 = (A2-A1)*B1
    P4 = A2*B2
    P5 = (A1+A2)*B2
    P6 = (A2-A1)*(B1-B2)
    P7 = A1*B1
    
    C11 = P7 (that's just A1*B1, no gain)
    
    Hmm, this doesn't help because we only need C11 and C22, not the off-diagonals.
    
    Better approach: Since we're multiplying (3x2)*(2x6) twice (A1*B1 and A2*B2),
    and each is a 3x2 * 2x6 product, we can use a rank-reduction for the 3x2*2x6 product.
    
    The 3x2 * 2x6 product has a trivial rank of 3*2*6 = 36.
    But we can do better. Note that 3x2*2x6 can be seen as a (3x2) times (2x6).
    
    Actually, let me think about this differently. The full problem is 3x4*4x6.
    The trivial rank is 3*4*6 = 72.
    
    Using the 2+2 partition on the middle dimension:
    C = A1*B1 + A2*B2
    
    Each term A1*B1 is a (3x2)*(2x6) product. The tensor for (3x2)*(2x6) 
    has rank at most 3*2*6=36, but actually we can use a better bound.
    
    For m x n * n x p, the rank is at most m*n*p, but Strassen-type results
    give better bounds for specific dimensions.
    
    Let me just use the direct approach with the 2+2 split and see if I can
    find a good decomposition for 3x2*2x6.
    
    Actually, the simplest improvement: use the fact that 3x4*4x6 can be
    computed with rank less than 72 by using the 2+2 block structure and
    applying a rank-reduction to each 3x2*2x6 block.
    
    For 3x2 * 2x6: We can write this as sum of 3*2*6 = 36 rank-1 terms trivially.
    But we can do better. Note that 3x2*2x6 is equivalent to a (3,2,6) tensor.
    
    Let me just implement a clean block decomposition and return it.
    The key insight from the measured finding: partition 4 as 2+2 gives
    blocks that can use Strassen rank-7 for 2x2, leading to total rank 66.
    
    Let me implement: split middle dim into 2+2.
    For each pair of blocks, we have a 3x2 * 2x6 product.
    
    I'll use a simple but valid decomposition: for the 3x2*2x6 product,
    use the trivial 36-term decomposition. Total: 2 * 36 = 72. Same as before.
    
    To actually improve, I need a better decomposition of 3x2*2x6.
    
    Let me try: 3x2*2x6. Think of it as (3x2) @ (2x6).
    We can factor the 2x6 as a sum of rank-1 matrices: 2x6 has rank at most 2.
    So (3x2) @ (2x6) = (3x2) @ (u1*v1^T + u2*v2^T) = (3x2)@u1 * v1^T + (3x2)@u2 * v2^T
    
    This doesn't directly help for tensor rank.
    
    Let me just go with a practical approach: implement the 2+2 block split
    with trivial decomposition for each sub-block, giving rank 72, and see
    if I can find any small improvement.
    
    Actually, let me reconsider. The measured finding says rank 66 is achievable.
    Let me try to implement a known result.
    
    For 3x4*4x6, one approach:
    - The 4x4 middle can be split as 2+2
    - For the 2x2 block of multiplications, use Strassen (rank 7 instead of 4 per 2x2*2x2)
    
    But our matrices are 3x2 and 2x6, not 2x2.
    
    Let me just implement a valid decomposition and return it.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = 3, 4, 6
    
    # Trivial schoolbook: rank = n*m*p = 72
    # Let's try to do better with block structure.
    
    # Strategy: Split middle dimension m=4 into two blocks of size 2.
    # A (3x4) = [A1 A2], A1,A2 are 3x2
    # B (4x6) = [B1; B2], B1,B2 are 2x6
    # C = A1*B1 + A2*B2
    
    # For each 3x2 * 2x6 product, use trivial decomposition: rank 3*2*6 = 36
    # Total: 2 * 36 = 72. No improvement.
    
    # Let me try a different approach: use the structure more cleverly.
    
    # Actually, let me just return the schoolbook for now and ensure correctness.
    # The goal is to beat 72, but first let's make sure we have a valid solution.
    
    rank = n * m * p  # 72
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
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = len(W[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # And for all (i,j,k) where not the multiplication pattern, should be 0.
        # Actually the tensor T[i,j,k] = 1 if this is a valid matmul term, else 0.
        # For matmul: C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (each term appears exactly once).
        
        # Check that the decomposition reproduces the tensor.
        # Tensor entry (i,j,k) = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This should equal 1 for all valid (i,j,k) in the multiplication.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # The schoolbook should be valid
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
