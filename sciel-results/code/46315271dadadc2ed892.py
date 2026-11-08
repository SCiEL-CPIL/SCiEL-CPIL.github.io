import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # Local indices:
    # A: r in 0..1, c in 0..1
    # B: r in 0..1, c in 0..1
    # C: r in 0..1, c in 0..1
    
    # P1: A00 * (B00 + B11)
    # P2: (A00 + A10) * B11
    # P3: (A01 + A11) * B00
    # P4: A11 * (B10 + B00)
    # P5: (A00 + A01) * (B11 - B10)
    # P6: (A10 - A00) * (B00 + B01)
    # P7: (A01 - A11) * (B10 + B11)
    
    strassen_factors = [
        # (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
        # Indices are local (r, c) for 2x2
        # U: (r, c)
        # V: (r, c)
        # W: (r, c)
        
        # P1
        {
            'U': [(0, 0, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # P2
        {
            'U': [(0, 0, 1), (1, 0, 1)],
            'V': [(1, 1, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        # P3
        {
            'U': [(0, 1, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        # P4
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, 1)],
            'W': [(0, 0, 1)]
        },
        # P5
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1), (1, 0, -1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        # P6
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 0, 1), (1, 1, 1)]
        },
        # P7
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    col = 0
    for b in range(2): # Main blocks for j: 0,1 and 2,3
        for s in range(3): # Sub blocks for k: 0,1; 2,3; 4,5
            for f in strassen_factors:
                # Apply U
                for (r, c, coeff) in f['U']:
                    # Global A index: row r, col b*2 + c
                    u_idx = r * m + (b * 2 + c)
                    U[u_idx][col] += coeff
                    
                # Apply V
                for (r, c, coeff) in f['V']:
                    # Global B index: row b*2 + r, col s*2 + c
                    v_idx = (b * 2 + r) * p + (s * 2 + c)
                    V[v_idx][col] += coeff
                    
                # Apply W
                for (r, c, coeff) in f['W']:
                    # Global C index: row r, col s*2 + c
                    # Trace convention: W[k*n + i, r] -> index = c_glob * n + r
                    w_idx = (s * 2 + c) * n + r
                    W[w_idx][col] += coeff
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
