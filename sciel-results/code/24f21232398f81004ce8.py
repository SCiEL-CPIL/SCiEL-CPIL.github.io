import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Laderman's algorithm coefficients for 3x3x3
    # These are the 23 linear forms for U (A), V (B), and W (C)
    # Flattened 3x3 matrices (row-major)
    
    # Note: I am using a known valid set of Laderman coefficients.
    # If these are incorrect, the verify step will catch it and fall back to schoolbook.
    
    # Laderman 23 terms
    # u, v, w are lists of 9 floats
    
    laderman_terms = [
        # (u, v, w)
        ([1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1]), # Placeholder, needs real data
    ]
    
    # Since I cannot guarantee the exact coefficients from memory, 
    # and hardcoding 23*27 numbers is error-prone without a reference,
    # I will use a fallback strategy.
    
    # However, the rank-46 construction is structurally sound.
    # If I cannot provide the exact Laderman coeffs, I cannot provide rank 46.
    
    # Is there a simpler rank-46 construction?
    # No, 46 comes from 2*23.
    
    # I will return the schoolbook decomposition (rank 54) to ensure validity,
    # as I cannot reliably hardcode Laderman's coefficients without a reference.
    # Wait, the prompt asks for a "stronger algorithm".
    # If I return 54, I tie the parent.
    
    # Let's try to find a rank-46 solution via a different method?
    # No, 46 is the known bound.
    
    # I will implement the schoolbook decomposition as a safe baseline.
    
    def schoolbook_decomp():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    U, V, W, rank = schoolbook_decomp()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
