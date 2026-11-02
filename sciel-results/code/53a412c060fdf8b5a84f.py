"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank using Strassen-like approach for 4x4x4 sub-blocks
    # Split 5 = 4 + 1
    # Use rank 49 for the 4x4x4 part (known optimal or near-optimal)
    # Handle the cross terms and 1x1 part separately
    
    # For a 4x4x4 matrix multiplication, the schoolbook rank is 64.
    # Strassen gives rank 49 for 7x7x7, but for 4x4x4 we can use:
    # Actually, the best known for 4x4x4 is 49 (from Strassen's 7x7x7 decomposition applied differently)
    # Let's use a simpler approach: keep schoolbook for now but try to find any improvement
    
    # Since finding a guaranteed lower rank is complex, let's return the schoolbook decomposition
    # which is always correct.
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
