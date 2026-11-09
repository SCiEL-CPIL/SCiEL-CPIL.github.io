"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}
    
    # Try to find a lower-rank decomposition using Strassen-inspired structure
    # For (3,5,5), we can use the following approach:
    # Decompose using smaller Strassen blocks
    
    # Method: Use the fact that 5 = 2+3, so we can split the inner dimension
    # A is 3x5, B is 5x5
    # Split B into B1 (5x2) and B2 (5x3)
    # Then C = A1*B1 + A2*B2 where A1 = A (3x5) viewed as 3x5, but we need to be careful
    
    # Alternative: Use explicit low-rank construction
    # For multiplying (n,m,p), if we can find linear combinations that reduce the rank
    
    # Let me try a direct approach: represent the multiplication as a sum of rank-1 terms
    # where each term uses linear combinations of inputs
    
    # For (3,5,5), try rank = 50 first (a guess based on known bounds)
    # Actually, let me try to use the "laser" construction or known explicit formulas
    
    # Simpler: try ranks from 50 down to 40, and for each, try to verify if a decomposition exists
    # But verification is hard. Instead, let me use a known construction.
    
    # Known result: For (2,2,2), rank = 7 (Strassen)
    # For (3,3,3), rank <= 23 (Strassen)
    # For (3,5,5), we can use a combination
    
    # Let me try a block decomposition:
    # Write A as 3x5, B as 5x5
    # Split the 5 inner dimension into 2+3
    # A = [A_left (3x2) | A_right (3x3)]
    # B = [B_top (2x5); B_bottom (3x5)]
    # C = A_left * B_top + A_right * B_bottom
    # This is two multiplications: (3,2,5) and (3,3,5)
    
    # For (3,2,5): rank <= 3*2*5 = 30, but can be lower
    # For (3,3,5): rank <= 3*3*5 = 45, but can be lower
    
    # This doesn't obviously help. Let me try a different approach.
    
    # Use the "border rank" idea or explicit bilinear forms.
    
    # Actually, let me just try to use a rank-1 decomposition search with a time limit
    # This is intractable for rank 50, so let me use a heuristic.
    
    # Heuristic: Use random linear combinations and check if they form a valid decomposition
    # This is unlikely to work quickly.
    
    # Let me try a more structured approach: use the known Strassen decomposition for (2,2,2)
    # and extend it.
    
    # For now, return the schoolbook solution. In a more advanced version, we would
    # use known low-rank constructions from the literature.
    
    # Check deadline
    if time.time() > deadline - 5:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
