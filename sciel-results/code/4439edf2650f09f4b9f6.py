import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # p=14, so 7 blocks of 2 columns
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen decomposition for 2x2 * 2x2
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V local indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W local indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # The 7 Strassen terms: (U_coeffs, V_coeffs, W_coeffs)
    # Each is a list of (index, value) pairs
    terms = [
        # M1: (A00+A11)*(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2: (A10+A11)*B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3: A00*(B01-B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4: A11*(B10-B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5: (A00+A01)*B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6: (A10-A00)*(B00+B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7: (A01-A11)*(B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for b in range(num_blocks):
        # Global V offsets for block b:
        # B00 -> row 0, col 2b -> index 2b
        # B01 -> row 0, col 2b+1 -> index 2b+1
        # B10 -> row 1, col 2b -> index 14 + 2b
        # B11 -> row 1, col 2b+1 -> index 14 + 2b + 1
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # Global W offsets for block b:
        # C00 -> row 2b, col 0 -> index 2*(2b) + 0 = 4b
        # C01 -> row 2b+1, col 0 -> index 2*(2b+1) + 0 = 4b+2
        # C10 -> row 2b, col 1 -> index 2*(2b) + 1 = 4b+1
        # C11 -> row 2b+1, col 1 -> index 2*(2b+1) + 1 = 4b+3
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_terms, v_terms, w_terms in terms:
            for idx, val in u_terms:
                U[idx, col] += val
            for l_idx, val in v_terms:
                V[v_offsets[l_idx], col] += val
            for l_idx, val in w_terms:
                W[w_offsets[l_idx], col] += val
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
