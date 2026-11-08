import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # Each entry is (U_coeffs, V_coeffs, W_coeffs)
    # Indices are (i, j) for U, (j, k) for V, (k, i) for W in local 2x2 coords
    # But we need to map to the specific linear indices.
    
    # Local indices for 2x2:
    # U: (0,0), (0,1), (1,0), (1,1)
    # V: (0,0), (0,1), (1,0), (1,1)
    # W: (0,0), (0,1), (1,0), (1,1) -- Note: W is indexed by k, i in prompt W[k*n+i]
    # So W local index is k*2 + i.
    
    strassen_terms = [
        # M1
        { 'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1} },
        # M2
        { 'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1} },
        # M3
        { 'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1} },
        # M4
        { 'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1} },
        # M5
        { 'U': {(0,0): 1, (1,0): 1}, 'V': {(1,1): 1}, 'W': {(0,1): 1, (0,0): -1} },
        # M6
        { 'U': {(0,1): 1, (0,0): -1}, 'V': {(0,0): 1, (1,0): 1}, 'W': {(1,1): 1} },
        # M7
        { 'U': {(1,0): 1, (1,1): -1}, 'V': {(0,1): 1, (1,1): 1}, 'W': {(0,0): 1} }
    ]
    
    for mu in range(4):
        for j_offset in [0, 2]:
            for term in strassen_terms:
                # Process U
                for (li, lj), val in term['U'].items():
                    gi = li
                    gj = j_offset + lj
                    u_idx = gi * m + gj
                    U[u_idx][col] += val
                
                # Process V
                for (lj, lk), val in term['V'].items():
                    gj = j_offset + lj
                    gk = 2 * mu + lk
                    v_idx = gj * p + gk
                    V[v_idx][col] += val
                
                # Process W
                # W local index is (k, i) -> k*2 + i
                # Prompt: W[k*n+i]
                for (lk, li), val in term['W'].items():
                    gk = 2 * mu + lk
                    gi = li
                    w_idx = gk * n + gi
                    W[w_idx][col] += val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
