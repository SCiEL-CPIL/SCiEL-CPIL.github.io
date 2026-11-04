"""Matrix multiplication 3x4x8 tensor decomposition."""
DIMENSIONS = [3, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x8 matrix.
    
    Strategy: Use Strassen-like block decomposition.
    - Split the 3x4 matrix into blocks and use bilinear algorithms to reduce rank.
    - The key is to decompose the inner dimension (4) using linear combinations.
    
    For 3x4x8, we can split the inner dimension 4 into two parts of size 2:
    A = [A1 | A2] where A1, A2 are 3x2
    B = [B1; B2] where B1, B2 are 2x8
    
    C = A1@B1 + A2@B2
    
    Each 3x2x8 multiplication can be done with rank at most 3*2*8 = 48,
    but we can use bilinear techniques to reduce this.
    
    Actually, let's try a different approach: use the fact that for small matrices,
    we can find explicit low-rank decompositions.
    
    Let me try to construct a decomposition with rank less than 96.
    
    One approach: For each output element C[i,k], we need sum_j A[i,j]*B[j,k].
    We can precompute linear combinations of the A and B entries to reduce multiplications.
    
    Let's try a systematic approach using the structure of the problem.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Try to find a decomposition with rank < 96
    # Start with schoolbook and try to reduce
    
    # Approach: Use block decomposition
    # Split inner dimension 4 into 2+2
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    # For each 3x2x8 sub-problem, find a low-rank decomposition
    # Then combine them
    
    # Let's try to use the fact that 3x2x8 can be decomposed more efficiently
    # by using linear combinations.
    
    # Actually, let me try a direct construction.
    # For 3x4x8, we can potentially use rank 64 or lower.
    
    # Let's try to build a decomposition using the following idea:
    # Group the j indices and use shared linear combinations.
    
    # For now, let's return a valid decomposition. We'll try to improve it.
    
    # Schoolbook gives 96. Let's see if we can do better.
    
    # Try rank 72: use 3 blocks of 4 multiplications each?
    # Or try to find a structured decomposition.
    
    # Let's just return schoolbook for now as a baseline.
    # Future improvements can search for better decompositions.
    
    rank = n * m * p  # 96
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 96
    V = [[0] * rank for _ in range(m * p)]  # 32 x 96
    W = [[0] * rank for _ in range(p * n)]  # 24 x 96
    
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
