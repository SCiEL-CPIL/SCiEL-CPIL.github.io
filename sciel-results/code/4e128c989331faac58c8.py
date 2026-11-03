"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 5, 5
    
    # Schoolbook rank is 3*5*5 = 75
    # We will replace one 2x2x2 subblock with Strassen's 7-term decomposition.
    # This reduces the rank by 1 (8 -> 7).
    # Total rank = 75 - 1 = 74.
    
    # We choose the subblock involving:
    # A rows: 0, 1
    # B cols: 0, 1
    # C entries: (0,0), (0,1), (1,0), (1,1)
    
    # The other 67 terms are standard schoolbook terms.
    # The 7 Strassen terms replace the 8 standard terms for the 2x2x2 block.
    
    rank = 74
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    # Helper to add a schoolbook term for (i, j, k)
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    
    # Helper to add a Strassen term
    # U_idx: list of (row, coeff) for U
    # V_idx: list of (row, coeff) for V
    # W_idx: list of (row, coeff) for W
    def add_strassen_term(u_terms, v_terms, w_terms):
        nonlocal col
        for row, coeff in u_terms:
            U[row][col] += coeff
        for row, coeff in v_terms:
            V[row][col] += coeff
        for row, coeff in w_terms:
            W[row][col] += coeff
        col += 1
    
    # Strassen's algorithm for 2x2x2:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b00)
    # M4 = a11(b11 - b01)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Mapping to global indices:
    # A rows 0, 1 -> U rows 0, 1, 5, 6
    #   A[0,0] -> U row 0
    #   A[0,1] -> U row 1
    #   A[1,0] -> U row 5
    #   A[1,1] -> U row 6
    # B cols 0, 1 -> V rows 0, 1, 5, 6
    #   B[0,0] -> V row 0
    #   B[0,1] -> V row 1
    #   B[1,0] -> V row 5
    #   B[1,1] -> V row 6
    # C entries (0,0), (0,1), (1,0), (1,1) -> W rows
    #   C[0,0] -> W row 0
    #   C[0,1] -> W row 3
    #   C[1,0] -> W row 1
    #   C[1,1] -> W row 4
    
    # M1 = (A[0,0] + A[1,1])(B[0,0] + B[1,1])
    u1 = [(0, 1), (6, 1)]
    v1 = [(0, 1), (6, 1)]
    w1 = [(0, 1), (4, 1)]  # C[0,0] and C[1,1]
    add_strassen_term(u1, v1, w1)
    
    # M2 = (A[1,0] + A[1,1])B[0,0]
    u2 = [(5, 1), (6, 1)]
    v2 = [(0, 1)]
    w2 = [(1, 1), (4, -1)]  # C[1,0] (+1), C[1,1] (-1)
    add_strassen_term(u2, v2, w2)
    
    # M3 = A[0,0](B[1,0] - B[0,0])
    u3 = [(0, 1)]
    v3 = [(5, 1), (0, -1)]
    w3 = [(3, 1), (4, 1)]  # C[0,1] (+1), C[1,1] (+1)
    add_strassen_term(u3, v3, w3)
    
    # M4 = A[1,1](B[1,1] - B[0,1])
    u4 = [(6, 1)]
    v4 = [(6, 1), (1, -1)]
    w4 = [(0, 1), (1, 1)]  # C[0,0] (+1), C[1,0] (+1)
    add_strassen_term(u4, v4, w4)
    
    # M5 = (A[0,0] + A[0,1])B[1,1]
    u5 = [(0, 1), (1, 1)]
    v5 = [(6, 1)]
    w5 = [(0, -1), (3, 1)]  # C[0,0] (-1), C[0,1] (+1)
    add_strassen_term(u5, v5, w5)
    
    # M6 = (A[1,0] - A[0,0])(B[0,0] + B[0,1])
    u6 = [(5, 1), (0, -1)]
    v6 = [(0, 1), (1, 1)]
    w6 = [(4, 1)]  # C[1,1] (+1)
    add_strassen_term(u6, v6, w6)
    
    # M7 = (A[0,1] - A[1,1])(B[1,0] + B[1,1])
    u7 = [(1, 1), (6, -1)]
    v7 = [(5, 1), (6, 1)]
    w7 = [(0, 1)]  # C[0,0] (+1)
    add_strassen_term(u7, v7, w7)
    
    # Now add the remaining schoolbook terms.
    # Total 75 terms. 8 were in the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}).
    # We skip those 8.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this term is part of the Strassen-replaced block
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                add_schoolbook(i, j, k)
    
    # Convert to lists of lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
