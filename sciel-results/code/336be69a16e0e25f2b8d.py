"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to map indices
    # A is n x m. Index in U: i * m + j
    # B is m x p. Index in V: j * p + k
    # C is p x n. Index in W: k * n + i
    
    # Strassen's 7 products defined by their contributions to U, V, W
    # Each product r is defined by:
    # U contributions: list of (i, j, coeff)
    # V contributions: list of (j, k, coeff)
    # W contributions: list of (k, i, coeff)
    
    products = [
        # r=0 (M1)
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # r=1 (M2)
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, -1), (1, 1, -1)]
        },
        # r=2 (M3)
        {
            'U': [(0, 0, 1)],
            'V': [(1, 0, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        # r=3 (M4)
        {
            'U': [(1, 1, 1)],
            'V': [(0, 1, 1), (0, 0, -1)],
            'W': [(0, 0, -1), (1, 0, 1)]
        },
        # r=4 (M5)
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        # r=5 (M6)
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (1, 0, 1)],
            'W': [(1, 0, -1), (1, 1, 1)]
        },
        # r=6 (M7)
        {
            'U': [(1, 1, 1), (1, 0, -1)],
            'V': [(0, 1, 1), (1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        }
    ]
    
    for r in range(rank):
        prod = products[r]
        for i, j, coeff in prod['U']:
            U[i * m + j, r] = coeff
        for j, k, coeff in prod['V']:
            V[j * p + k, r] = coeff
        for k, i, coeff in prod['W']:
            W[k * n + i, r] = coeff
            
    # Convert to lists as required by interface? 
    # The prompt says "factor matrices of shapes... with integer/half-integer entries"
    # and returns a dict. The seed program returns lists of lists.
    # I will return lists of lists to be safe, or numpy arrays if allowed.
    # Seed program returns lists. I will return lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
