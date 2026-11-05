"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 Strassen decomposition
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen terms for 2x2 block multiplication of 6x6 matrices
    # Blocks are 3x3
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2
    # A22: rows 3-5, cols 3-5
    
    # Helper to get global index for A (row-major)
    def idx_A(r, c):
        return r * 6 + c
        
    # Helper to get global index for B (row-major)
    def idx_B(r, c):
        return r * 6 + c
        
    # Helper to get global index for W (convention k*n+i, i.e., col*6+row)
    def idx_W(r, c):
        return c * 6 + r

    # Define the 7 Strassen multiplications
    # Each entry is a list of (block_name, coeff) for X, Y, and Z contributions to C
    # X is from A, Y is from B.
    # Z contributes to specific C blocks.
    
    # M1 = (A11 + A22)(B11 + B22)
    # C11 += M1, C22 += M1
    strassen_terms = [
        {
            "A": [("11", 1), ("22", 1)],
            "B": [("11", 1), ("22", 1)],
            "C": [("11", 1), ("22", 1)]
        },
        # M2 = (A21 + A22)B11
        # C21 += M2, C22 -= M2
        {
            "A": [("21", 1), ("22", 1)],
            "B": [("11", 1)],
            "C": [("21", 1), ("22", -1)]
        },
        # M3 = A11(B12 - B22)
        # C12 += M3, C22 += M3
        {
            "A": [("11", 1)],
            "B": [("12", 1), ("22", -1)],
            "C": [("12", 1), ("22", 1)]
        },
        # M4 = A22(B21 - B11)
        # C11 += M4, C21 += M4
        {
            "A": [("22", 1)],
            "B": [("21", 1), ("11", -1)],
            "C": [("11", 1), ("21", 1)]
        },
        # M5 = (A11 + A12)B22
        # C11 -= M5, C12 += M5
        {
            "A": [("11", 1), ("12", 1)],
            "B": [("22", 1)],
            "C": [("11", -1), ("12", 1)]
        },
        # M6 = (A21 - A11)(B11 + B12)
        # C22 += M6
        {
            "A": [("21", 1), ("11", -1)],
            "B": [("11", 1), ("12", 1)],
            "C": [("22", 1)]
        },
        # M7 = (A12 - A22)(B21 + B22)
        # C11 += M7
        {
            "A": [("12", 1), ("22", -1)],
            "B": [("21", 1), ("22", 1)],
            "C": [("11", 1)]
        }
    ]
    
    # Block offsets
    # 11: (0,0), 12: (0,3), 21: (3,0), 22: (3,3)
    block_offsets = {
        "11": (0, 0),
        "12": (0, 3),
        "21": (3, 0),
        "22": (3, 3)
    }
    
    for term in strassen_terms:
        for u in range(3):
            for w in range(3):
                for v in range(3):
                    # Process A contributions
                    for block_name, coeff in term["A"]:
                        r_off, c_off = block_offsets[block_name]
                        r = u + r_off
                        c = w + c_off
                        U[idx_A(r, c)][col] += coeff
                        
                    # Process B contributions
                    for block_name, coeff in term["B"]:
                        r_off, c_off = block_offsets[block_name]
                        r = w + r_off
                        c = v + c_off
                        V[idx_B(r, c)][col] += coeff
                        
                    # Process C contributions
                    for block_name, coeff in term["C"]:
                        r_off, c_off = block_offsets[block_name]
                        r = u + r_off
                        c = v + c_off
                        W[idx_W(r, c)][col] += coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
