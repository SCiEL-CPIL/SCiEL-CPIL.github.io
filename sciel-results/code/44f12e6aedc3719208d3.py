"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # Try a block decomposition approach
    # Split into 2x2 blocks where possible, use Strassen for those
    
    # Strategy: Decompose 4x5 * 5x5 into blocks
    # A (4x5) split by rows: A_top (2x5), A_bot (2x5)
    # B (5x5) split by cols: B_left (5x2), B_mid (5x2), B_right (5x1)
    # C = [A_top*B_left, A_top*B_mid, A_top*B_right;
    #      A_bot*B_left, A_bot*B_mid, A_bot*B_right]
    # Each A_top*B_left is 2x2, use Strassen (rank 7)
    # Each A_top*B_right is 2x1, schoolbook (rank 2)
    
    # But we need to decompose 2x5 * 5x2 into 2x2 * 2x2 blocks
    # 2x5 * 5x2 = sum_{j=0}^{4} A[:,j] * B[j,:]
    # This is not directly 2x2 * 2x2
    
    # Let me try a different approach: just use schoolbook but with some optimization
    # Or use known results
    
    # For now, let's try to construct a decomposition with rank < 100
    # using a simple method: try to find a valid low-rank decomposition
    
    # Given time constraints, let's implement a known construction
    # Use the fact that we can embed smaller multiplications
    
    # Simple approach: Use rank 90 decomposition by splitting 5 = 3+2
    # A (4x5) = [A3 (4x3) | A2 (4x2)]
    # B (5x5) = [B3 (3x5); B2 (2x5)]
    # C = A3*B3 + A2*B2
    # A3*B3: 4x3 * 3x5, schoolbook rank = 4*3*5 = 60
    # A2*B2: 4x2 * 2x5, schoolbook rank = 4*2*5 = 40
    # Total: 100 - same as before
    
    # Try 5 = 4+1:
    # A3*B3: 4x4 * 4x5, rank <= 4*4*5 = 80
    # A1*B1: 4x1 * 1x5, rank <= 4*1*5 = 20
    # Total: 100
    
    # It seems hard to beat 100 with simple block splits
    
    # Let me try a different strategy: use integer linear programming or 
    # a randomized search for a low-rank decomposition
    
    # Given the 150s limit, let's try a simple randomized approach
    # to find a valid decomposition with rank < 100
    
    # Actually, let's just return the schoolbook decomposition as a fallback
    # and try to improve if time permits
    
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
