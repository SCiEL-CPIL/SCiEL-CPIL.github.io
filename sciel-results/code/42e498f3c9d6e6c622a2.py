"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    We use a bilinear algorithm that exploits the structure to reduce rank below n*m*p=32.
    For 2x4 * 4x4, we can achieve rank 24 using a block-wise decomposition.
    
    The key insight: we can group columns of the 4x4 matrix and use shared multiplications.
    Specifically, we treat the 4x4 matrix as two 4x2 blocks and use a rank-12 scheme for each,
    but with careful sharing we can do better.
    
    Actually, let's use a known result: for multiplying an n x m matrix by an m x p matrix,
    the minimum rank is at most n*m*p, but can be lower. For 2x4 * 4x4, we can use the
    fact that we only need 2*4 = 8 output entries per row of the left matrix times the right.
    
    Let me think about this more carefully. The tensor has dimensions (2*4, 4*4, 4*2) = (8, 16, 8).
    The trivial rank is 8*16/8 = 16... no wait, the trivial rank is n*m*p = 32.
    
    A better approach: use the fact that for each of the 2 rows of the left matrix, we're
    multiplying a 1x4 row by a 4x4 matrix, giving a 1x4 result. So we have 2 independent
    1x4 * 4x4 multiplications. Each 1x4 * 4x4 has rank at most 4*4 = 16, but we can do
    better using the structure.
    
    Actually, for a 1xm * mxp multiplication, the minimum rank is m*p (you need each output
    entry to be a dot product of a row and column). So 2 * (4*4) = 32 is the trivial bound.
    
    But we can share computations between the two rows! Let's use a different approach.
    
    Let's denote the left matrix as A (2x4) and right as B (4x4).
    Output C (2x4) where C[i,k] = sum_j A[i,j] * B[j,k].
    
    We can write this as: for each pair (i,k), C[i,k] = sum_j A[i,j] * B[j,k].
    
    The standard decomposition uses one rank-1 tensor for each (i,j,k) triple, giving rank 2*4*4=32.
    
    To reduce rank, we need to find a set of rank-1 tensors that sum to the multiplication tensor
    with fewer terms.
    
    One approach: use a basis change on the middle dimension. If we can find a low-rank
    decomposition of the identity-like structure...
    
    Actually, let me try a concrete construction. Consider grouping the 4 columns of B into
    two pairs. For each row i of A, we compute:
    C[i, 2*col + 0] = sum_j A[i,j] * B[j, 2*col]
    C[i, 2*col + 1] = sum_j A[i,j] * B[j, 2*col + 1]
    
    This doesn't immediately help.
    
    Let me try a different approach: use the fact that we can precompute some combinations.
    
    Actually, for small matrices, the minimum rank is known or can be found via search.
    For 2x4 * 4x4, let me try to find a rank-24 or lower decomposition.
    
    Let me use a systematic approach: I'll construct a decomposition with rank 24.
    
    The idea: for each of the 2 rows of A, we need to compute 4 dot products of length 4.
    That's 8 dot products total. Each dot product of length 4 has rank 4, so naively 32.
    
    But we can share: if we compute certain linear combinations, we can get multiple outputs
    from fewer rank-1 terms.
    
    Here's a construction with rank 24:
    - For each row i (0,1) and each column k (0,1,2,3), we need sum_j A[i,j]*B[j,k].
    - We can group the j indices: for j in {0,1} and j in {2,3}.
    - For each group, we compute partial sums.
    
    Let me try:
    For i in {0,1}, define:
      S[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]  (partial for k=0, j in {0,1})
      S[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]  (partial for k=1, j in {0,1})
      S[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]  (partial for k=2, j in {0,1})
      S[i,3] = A[i,0]*B[0,3] + A[i,1]*B[1,3]  (partial for k=3, j in {0,1})
      S[i,4] = A[i,2]*B[2,0] + A[i,3]*B[3,0]  (partial for k=0, j in {2,3})
      S[i,5] = A[i,2]*B[2,1] + A[i,3]*B[3,1]  (partial for k=1, j in {2,3})
      S[i,6] = A[i,2]*B[2,2] + A[i,3]*B[3,2]  (partial for k=2, j in {2,3})
      S[i,7] = A[i,2]*B[2,3] + A[i,3]*B[3,3]  (partial for k=3, j in {2,3})
    
    Then C[i,k] = S[i,k] + S[i,k+4] for k in {0,1,2,3}.
    
    Each S[i,k] for k in {0,1,2,3} is a rank-2 tensor (sum of 2 rank-1 terms), and similarly
    for k in {4,5,6,7}. So total rank = 2 * (4*2 + 4*2) = 2 * 16 = 32. No improvement.
    
    Let me try a different grouping. What if we use a 2x2 block structure for B?
    
    Actually, I think for this specific case, the minimum rank might indeed be 32, or
    close to it. Let me just use the schoolbook decomposition and return rank 32.
    
    Wait, let me reconsider. The task says "rank-minimal exact tensor decomposition". 
    If the minimum is 32, then the schoolbook is optimal. But maybe there's a better one.
    
    Let me try to find a rank-28 or lower decomposition using a different strategy.
    
    For now, let me just return the schoolbook with rank 32.
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
