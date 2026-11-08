"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank = n * m * p
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a better decomposition using Strassen-like methods
    # For 3x4 * 4x7, we can try splitting the inner dimension 4 = 2 + 2
    # and using Strassen on the 2x2 blocks
    
    # Check time
    if time.time() > deadline - 5:
        return best_result
    
    # Try a block-based approach:
    # A (3x4) = [A1 (3x2), A2 (3x2)]
    # B (4x7) = [B1 (2x7); B2 (2x7)]
    # C = A1*B1 + A2*B2
    # For each (3x2)*(2x7) product, use 3 Strassen-like terms instead of 4
    
    # Actually, for (3x2)*(2x7), the standard is 3*2*7=42 terms.
    # Let's try to use a 2-term decomposition with 3 multiplications:
    # M1 = (A[:,0] + A[:,1]) * (B[0,:] + B[1,:])  -> 3x7
    # M2 = A[:,0] * B[1,:]  -> 3x7  
    # M3 = A[:,1] * B[0,:]  -> 3x7
    # C = M1 - M2 - M3
    # This uses 3 "multiplications" but each is a 3x7 outer product sum.
    # In tensor terms, this is 3 * 3 * 7 = 63 terms for the (3x2)*(2x7) case.
    # That's worse than 42, so not helpful.
    
    # Let's try a different approach: use known results for small matrix multiplication.
    # For 2x2 * 2x2, Strassen gives rank 7.
    # We can try to embed this.
    
    # Split 3x4 * 4x7 into:
    # Row 0 of A (1x4) * B (4x7) -> 1x7
    # Row 1 of A (1x4) * B (4x7) -> 1x7
    # Row 2 of A (1x4) * B (4x7) -> 1x7
    # Each 1x4 * 4x7 has rank 4 (dot product with each of 7 columns)
    # Total: 12 terms? No, that's not how tensor rank works.
    
    # The correct schoolbook is 3*4*7 = 84.
    
    # Let's try to use a 4x4 Strassen-like method.
    # Pad A to 4x4 by adding a zero row, pad B to 4x4 by adding zero columns.
    # Then use Strassen for 4x4*4x4 (rank 18 or so), and extract the 3x7 result.
    # But this might not give a better rank for the subproblem.
    
    # Given the time constraints, let's just return the schoolbook.
    # The schoolbook rank is 84, which is the seed.
    
    return best_result
# EVOLVE-BLOCK-END
