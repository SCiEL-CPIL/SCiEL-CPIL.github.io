"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 74 decomposition using Strassen for 2x2x2 block
    rank = 74
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 block for i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # Helper to set U, V, W for a term
    def set_term(u_entries, v_entries, w_entries):
        nonlocal col
        for (i, j), val in u_entries:
            U[i * m + j][col] = val
        for (j, k), val in v_entries:
            V[j * p + k][col] = val
        for (k, i), val in w_entries:
            W[k * n + i][col] = val
        col += 1

    # M1
    set_term(
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1), ((1,1), 1)]
    )
    # M2
    set_term(
        [((1,0), 1), ((1,1), 1)],
        [((0,0), 1)],
        [((0,1), 1), ((1,1), -1)]
    )
    # M3
    set_term(
        [((0,0), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((1,0), 1), ((1,1), 1)]
    )
    # M4
    set_term(
        [((1,1), 1)],
        [((0,1), 1), ((0,0), -1)],
        [((0,0), 1), ((0,1), 1)]
    )
    # M5
    set_term(
        [((0,0), 1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), -1), ((1,0), 1)]
    )
    # M6
    set_term(
        [((1,1), 1), ((0,0), -1)],
        [((0,0), 1), ((1,0), 1)],
        [((1,1), 1)]
    )
    # M7
    set_term(
        [((0,1), 1), ((1,1), -1)],
        [((1,1), 1), ((0,1), 1)],
        [((0,0), 1)]
    )
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
