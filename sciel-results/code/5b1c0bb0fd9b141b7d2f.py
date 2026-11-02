"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose into 8 independent 2x2x2 multiplications using Strassen
    # j_block in {0, 1} splits the m=4 dimension into two 2s
    # k_block in {0, 1, 2, 3} splits the p=8 dimension into four 2s
    # Total sub-problems = 2 * 4 = 8
    # Each uses rank 7 (Strassen)
    rank = 8 * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # U_local: [a00, a01, a10, a11]
    # V_local: [b00, b01, b10, b11]
    # W_local: [c00, c01, c10, c11]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([1, 0, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    col = 0
    for j_block in range(2):
        for k_block in range(4):
            for u_loc, v_loc, w_loc in strassen_terms:
                # U mapping: global row i * m + j
                # i in {0, 1}, j = 2 * j_block + j_loc
                for i in range(n):
                    for j_loc in range(2):
                        val = u_loc[j_loc * 2 + i] # Wait, u_loc is flattened row-major?
                        # u_loc = [a00, a01, a10, a11]
                        # Index 0: a00 (i=0, j_loc=0)
                        # Index 1: a01 (i=0, j_loc=1)
                        # Index 2: a10 (i=1, j_loc=0)
                        # Index 3: a11 (i=1, j_loc=1)
                        # So u_loc[i * 2 + j_loc]
                        if val != 0:
                            u_row = i * m + (2 * j_block + j_loc)
                            U[u_row][col] += val
                
                # V mapping: global row j * p + k
                # j = 2 * j_block + j_loc, k = 4 * k_block + k_loc
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v_loc[j_loc * 2 + k_loc]
                        if val != 0:
                            v_row = (2 * j_block + j_loc) * p + (4 * k_block + k_loc)
                            V[v_row][col] += val
                
                # W mapping: global row k * n + i
                # k = 4 * k_block + k_loc, i in {0, 1}
                for i in range(n):
                    for k_loc in range(2):
                        val = w_loc[k_loc * 2 + i]
                        if val != 0:
                            w_row = (4 * k_block + k_loc) * n + i
                            W[w_row][col] += val
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
