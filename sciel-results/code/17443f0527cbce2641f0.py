"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a scalar multiplication term
    # u_idx: list of (i, j, coeff)
    # v_idx: list of (j, k, coeff)
    # w_idx: list of (k, i, coeff)
    def add_term(u_terms, v_terms, w_terms):
        nonlocal col
        for (i, j, c_u) in u_terms:
            U[i * m + j][col] += c_u
        for (j, k, c_v) in v_terms:
            V[j * p + k][col] += c_v
        for (k, i, c_w) in w_terms:
            W[k * n + i][col] += c_w
        col += 1

    # Block A11 (2x2) x B11 (2x2) -> C11 (2x2)
    # Indices: A i in 0,1; j in 0,1. B j in 0,1; k in 0,1. C i in 0,1; k in 0,1.
    # Strassen 1: (A00+A11)(B00+B11)
    add_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    # Strassen 2: (A10+A11)B00
    add_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)])
    # Strassen 3: A00(B01-B11)
    add_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)])
    # Strassen 4: A11(B10-B00)
    add_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(1,1,1)])
    # Strassen 5: (A00+A01)B11
    add_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1)])
    # Strassen 6: (A10-A00)(B00+B01)
    add_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1)])
    # Strassen 7: (A01-A11)(B10+B11)
    add_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(1,1,1)])
    
    # Block A11 (2x2) x B12 (2x2) -> C12 (2x2)
    # B12 cols 2,3.
    # Strassen 1: (A00+A11)(B02+B13)
    add_term([(0,0,1), (1,1,1)], [(0,2,1), (1,3,1)], [(2,0,1), (3,1,1)])
    # Strassen 2: (A10+A11)B02
    add_term([(1,0,1), (1,1,1)], [(0,2,1)], [(2,1,1)])
    # Strassen 3: A00(B03-B13)
    add_term([(0,0,1)], [(0,3,1), (1,3,-1)], [(3,0,1)])
    # Strassen 4: A11(B12-B02)
    add_term([(1,1,1)], [(1,2,1), (0,2,-1)], [(3,1,1)])
    # Strassen 5: (A00+A01)B13
    add_term([(0,0,1), (0,1,1)], [(1,3,1)], [(3,0,1)])
    # Strassen 6: (A10-A00)(B02+B03)
    add_term([(1,0,1), (0,0,-1)], [(0,2,1), (0,3,1)], [(2,1,1)])
    # Strassen 7: (A01-A11)(B12+B13)
    add_term([(0,1,1), (1,1,-1)], [(1,2,1), (1,3,1)], [(3,1,1)])

    # Block A12 (2x1) x B21 (1x2) -> C11 (2x2)
    # A12: i in 0,1; j in 2. B21: j in 2; k in 0,1.
    # Terms: A02*B20, A02*B21, A12*B20, A12*B21
    add_term([(0,2,1)], [(2,0,1)], [(0,0,1)])
    add_term([(0,2,1)], [(2,1,1)], [(1,0,1)])
    add_term([(1,2,1)], [(2,0,1)], [(0,1,1)])
    add_term([(1,2,1)], [(2,1,1)], [(1,1,1)])

    # Block A12 (2x1) x B22 (1x2) -> C12 (2x2)
    # B22: j in 2; k in 2,3.
    add_term([(0,2,1)], [(2,2,1)], [(2,0,1)])
    add_term([(0,2,1)], [(2,3,1)], [(3,0,1)])
    add_term([(1,2,1)], [(2,2,1)], [(2,1,1)])
    add_term([(1,2,1)], [(2,3,1)], [(3,1,1)])

    # Block A21 (1x2) x B11 (2x2) -> C21 (1x2)
    # A21: i in 2; j in 0,1. B11: j in 0,1; k in 0,1.
    # Terms: A20*B00, A20*B01, A21*B10, A21*B11
    add_term([(2,0,1)], [(0,0,1)], [(0,2,1)])
    add_term([(2,0,1)], [(0,1,1)], [(1,2,1)])
    add_term([(2,1,1)], [(1,0,1)], [(0,2,1)])
    add_term([(2,1,1)], [(1,1,1)], [(1,2,1)])

    # Block A21 (1x2) x B12 (2x2) -> C22 (1x2)
    # B12: j in 0,1; k in 2,3.
    add_term([(2,0,1)], [(0,2,1)], [(2,2,1)])
    add_term([(2,0,1)], [(0,3,1)], [(3,2,1)])
    add_term([(2,1,1)], [(1,2,1)], [(2,2,1)])
    add_term([(2,1,1)], [(1,3,1)], [(3,2,1)])

    # Block A22 (1x1) x B21 (1x2) -> C21 (1x2)
    # A22: i in 2; j in 2. B21: j in 2; k in 0,1.
    add_term([(2,2,1)], [(2,0,1)], [(0,2,1)])
    add_term([(2,2,1)], [(2,1,1)], [(1,2,1)])

    # Block A22 (1x1) x B22 (1x2) -> C22 (1x2)
    # B22: j in 2; k in 2,3.
    add_term([(2,2,1)], [(2,2,1)], [(2,2,1)])
    add_term([(2,2,1)], [(2,3,1)], [(3,2,1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
