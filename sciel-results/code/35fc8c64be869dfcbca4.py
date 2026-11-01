import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=9
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (18, rank)
    # W: (p*n, rank) = (18, rank)
    
    # We use 4 Strassen blocks (2 cols each) and 1 direct block (1 col)
    # Cols: (0,1), (2,3), (4,5), (6,7), (8)
    rank = 4 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Define Strassen coefficients for a block (ka, kb)
    def add_strassen_block(ka, kb):
        nonlocal col
        # M1: (A00+A11)(B0a+B1b) -> C0a(+1), C1b(+1)
        add_term(0, 1, ka, 1, 2*ka, 1)
        add_term(3, 1, 9+kb, 1, 2*kb+1, 1)
        
        # M2: (A10+A11)B0a -> C1a(+1), C1b(-1)
        add_term(2, 1, ka, 1, 2*ka+1, 1)
        add_term(3, 1, ka, 0, 2*kb+1, -1) # v is 0 for M2? No, v is just B0a.
        # Correction: add_term sets one index per call. 
        # My add_term above increments col. This is wrong for multi-index vectors.
        # I need to fill the whole column for a term.
        
    # Re-implementing with direct column filling
    col = 0
    def fill_column(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals:
            U[idx][col] = val
        for idx, val in v_vals:
            V[idx][col] = val
        for idx, val in w_vals:
            W[idx][col] = val
        col += 1

    def add_strassen_block(ka, kb):
        # M1
        fill_column(
            [(0, 1), (3, 1)],
            [(ka, 1), (9+kb, 1)],
            [(2*ka, 1), (2*kb+1, 1)]
        )
        # M2
        fill_column(
            [(2, 1), (3, 1)],
            [(ka, 1)],
            [(2*ka+1, 1), (2*kb+1, -1)]
        )
        # M3
        fill_column(
            [(0, 1)],
            [(9+ka, 1), (9+kb, -1)],
            [(2*kb, 1), (2*kb+1, 1)]
        )
        # M4
        fill_column(
            [(3, 1)],
            [(kb, 1), (ka, -1)],
            [(2*ka, 1), (2*ka+1, 1)]
        )
        # M5
        fill_column(
            [(0, 1), (1, 1)],
            [(9+kb, 1)],
            [(2*ka, -1), (2*kb, 1)]
        )
        # M6
        fill_column(
            [(2, 1), (0, -1)],
            [(ka, 1), (kb, 1)],
            [(2*kb+1, 1)]
        )
        # M7
        fill_column(
            [(1, 1), (3, -1)],
            [(9+ka, 1), (9+kb, 1)],
            [(2*ka, 1)]
        )

    def add_direct_block(k):
        # C0k = A00 B0k + A01 B1k
        # C1k = A10 B0k + A11 B1k
        fill_column(
            [(0, 1)],
            [(k, 1)],
            [(2*k, 1)]
        )
        fill_column(
            [(1, 1)],
            [(9+k, 1)],
            [(2*k, 1)]
        )
        fill_column(
            [(2, 1)],
            [(k, 1)],
            [(2*k+1, 1)]
        )
        fill_column(
            [(3, 1)],
            [(9+k, 1)],
            [(2*k+1, 1)]
        )

    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    add_strassen_block(6, 7)
    add_direct_block(8)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
