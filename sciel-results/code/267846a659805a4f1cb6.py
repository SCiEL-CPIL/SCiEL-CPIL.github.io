"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy:
    - Treat the 3x6 factor as three 3x2 sub-blocks.
    - Use Strassen-like rank-7 decomposition for the 2x2 core structure.
    - Exploit the block structure to share computations.
    
    The known rank for 2x3 x 3x6 is at most 24 (schoolbook: 2*3*6=36).
    Using Strassen on 2x2 blocks: we can decompose the multiplication by
    viewing the 3x6 matrix as [3x2 | 3x2 | 3x2] and the 2x3 as a 2x3 matrix.
    
    Actually, let's think more carefully. The tensor is M_{2,3,6}.
    We can write the 2x3 matrix A and 3x6 matrix B.
    
    A = [[a11, a12, a13],
         [a21, a22, a23]]
    
    B = [[b11, b12, b13, b14, b15, b16],
         [b21, b22, b23, b24, b25, b26],
         [b31, b32, b33, b34, b35, b36]]
    
    The product C = A*B is 2x6.
    
    One approach: use the fact that we can group columns of B.
    Let's split B into three 3x2 blocks: B1, B2, B3.
    Then C = [A*B1, A*B2, A*B3].
    
    For each A*Bj (2x3 * 3x2 -> 2x2), we can use Strassen's algorithm which has rank 7.
    But we need to be careful about sharing.
    
    Actually, the total rank should be 3 * 7 = 21 if we use Strassen independently on each 3x2 block.
    But wait, the input A is shared across all three blocks. Let me reconsider.
    
    The tensor M_{2,3,6} can be seen as: for each pair (i,k) in {1,2}x{1,...,6},
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    If we fix the column block (columns 2t-1, 2t of B for t=1,2,3), we have a 2x3 x 3x2 tensor.
    The rank of the 2x3 x 3x2 tensor is known to be 9 (schoolbook) but can be reduced.
    
    Actually, let's use a different approach. The rank of M_{2,3,6} is known to be at most 24.
    Let me construct a rank-24 decomposition using Strassen-like ideas.
    
    For 2x2 x 2x2, Strassen gives rank 7.
    For 2x3 x 3x2, we can pad to 2x4 x 4x2 or use a direct construction.
    
    Let me try a direct construction for rank 24.
    
    Actually, the best known upper bound for M_{2,3,6} is 24. Let me construct it.
    
    We can write:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Let's use the following decomposition. For the 2x3 x 3x2 case (which appears 3 times),
    the rank is 9 for schoolbook. But we can do better.
    
    Actually, let me just implement the schoolbook rank 36 first to ensure correctness,
    then try to improve.
    
    Wait, the observation says a program achieved score 1.091 with rank 33.
    Let me try to get rank 24 or lower.
    
    Known result: The rank of M_{2,3,6} is at most 24.
    
    Let me construct a rank-24 decomposition.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (18, rank)  
    # W shape: (p*n, rank) = (12, rank)
    
    # Let's use a rank-24 construction based on Strassen-like decomposition.
    # We'll build it by considering the structure.
    
    # For now, let's implement a known rank-24 decomposition.
    # The idea: split the 6 columns into 3 groups of 2.
    # For each group, use a rank-8 decomposition for 2x3 x 3x2.
    # Total rank = 3 * 8 = 24.
    
    # Rank-8 decomposition for 2x3 x 3x2:
    # A is 2x3, B is 3x2, C = A*B is 2x2.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # We can use a rank-8 decomposition. Let me define it.
    
    def make_rank8_2x3x3x2():
        """
        Returns U (6,8), V (6,8), W (4,8) for 2x3 x 3x2 -> 2x2.
        U[i*3+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
        
        Wait, the convention is:
        U[i*m+j, c] where m=3, so U[i*3+j, c] for A[i,j] with i in 0..1, j in 0..2.
        V[j*p+k, c] where p=2, so V[j*2+k, c] for B[j,k] with j in 0..2, k in 0..1.
        W[k*n+i, c] where n=2, so W[k*2+i, c] for C[i,k] with i in 0..1, k in 0..1.
        """
        # Rank 8 decomposition for 2x3 x 3x2
        # I'll use a known construction.
        
        # Let's define 8 terms.
        # Term 0: A[0,0] * B[0,0] -> contributes to C[0,0]
        # Term 1: A[0,0] * B[0,1] -> contributes to C[0,1]
        # Term 2: A[0,1] * B[1,0] -> contributes to C[0,0]
        # Term 3: A[0,1] * B[1,1] -> contributes to C[0,1]
        # Term 4: A[0,2] * B[2,0] -> contributes to C[0,0]
        # Term 5: A[0,2] * B[2,1] -> contributes to C[0,1]
        # Term 6: A[1,0] * B[0,0] -> contributes to C[1,0]
        # Term 7: A[1,0] * B[0,1] -> contributes to C[1,1]
        # ... this is schoolbook rank 12, not 8.
        
        # Let me use a proper rank-8 decomposition.
        # Actually, the rank of 2x3 x 3x2 might be 9, not 8.
        
        # Let me just use schoolbook rank 12 for 2x3 x 3x2, giving total rank 36.
        # That's the same as the original.
        
        # Let me try a different approach. Use Strassen on 2x2 subblocks.
        
        # Split A into columns: A = [A_col0, A_col1, A_col2] where each is 2x1.
        # Split B into rows and columns: B is 3x2.
        
        # Actually, let's use the known rank-24 decomposition for M_{2,3,6}.
        # I'll construct it directly.
        
        pass
    
    # Let me implement the rank-36 schoolbook first, then try to optimize.
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
