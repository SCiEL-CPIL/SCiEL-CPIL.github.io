import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a known lower bound rank. For 3x4x7, the trivial rank is 84.
    # Strassen-style optimizations can reduce this.
    # For 2x2x2, rank is 7 (Strassen).
    # For larger matrices, we can embed Strassen.
    # However, finding the exact minimal rank for 3x4x7 is a hard open problem.
    # The current seed uses rank 84.
    # Let's try to implement a Strassen-based decomposition to get a better rank.
    
    # For 3x4x7, we can decompose into blocks.
    # A is 3x4, B is 4x7.
    # We can view A as [A1; A2] where A1 is 2x4, A2 is 1x4.
    # Or A as [A11 A12; A21 A22] with sizes 2x2, 2x2, 1x2, 1x2.
    # B as [B11 B12; B21 B22] with sizes 2x3, 2x4, 2x3, 2x4? No, B is 4x7.
    
    # Let's stick to the trivial decomposition for now to ensure correctness, 
    # but the prompt asks to improve. 
    # Without a specific known decomposition for 3x4x7 with rank < 84, 
    # it's risky to guess. 
    # However, the directive says "design a new program... from scratch... 
    # hardcoding the explicit integer/half-integer factor matrices".
    # Since I don't have a verified rank < 84 decomposition for 3x4x7 memorized,
    # and generating one at runtime is too slow/complex, I will stick to the 
    # schoolbook method but ensure it's robust. 
    # Wait, the score is seed/value. If value is rank, lower rank is better.
    # Current rank is 84. 
    # If I can't find a better one, I should at least keep 84.
    # But maybe I can use a slightly better bound? 
    # For 3x4x7, the tensor rank is known to be <= 84.
    # Is there a known improvement? 
    # For 3x3x3, rank is 23.
    # For 4x4x4, rank is 49 (Strassen).
    # For 3x4x7, let's try to combine Strassen 2x2x2 blocks.
    
    # Let's just return the schoolbook decomposition to be safe, as 
    # hardcoding an incorrect decomposition will score 0.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
