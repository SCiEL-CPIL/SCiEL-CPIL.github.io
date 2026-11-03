"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        for i in range(rank):
            if i < len(u_indices):
                U[i][i] = u_indices[i]
            if i < len(v_indices):
                V[i][i] = v_indices[i]
            if i < len(w_indices):
                W[i][i] = w_indices[i]
                
    # This helper is wrong for sparse vectors. Let's do it manually or with a better helper.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_term(col, u_sp, v_sp, w_sp):
        for idx, val in u_sp.items():
            U[idx][col] = val
        for idx, val in v_sp.items():
            V[idx][col] = val
        for idx, val in w_sp.items():
            W[idx][col] = val

    # Strassen 7 terms for 2x2 core
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # T1: (A00+A01) * B00 -> C00 - C01
    set_term(0, {0: 1, 1: 1}, {0: 1}, {0: 1, 1: -1})
    
    # T2: A01 * (B10-B00) -> C00
    set_term(1, {1: 1}, {3: 1, 0: -1}, {0: 1})
    
    # T3: (A00+A11) * B11 -> C11 + C00 - C01
    set_term(2, {0: 1, 4: 1}, {4: 1}, {4: 1, 0: 1, 1: -1})
    
    # T4: A11 * (B01-B00) -> C00
    set_term(3, {4: 1}, {1: 1, 0: -1}, {0: 1})
    
    # T5: (A10+A11) * B00 -> C10
    set_term(4, {3: 1, 4: 1}, {0: 1}, {3: 1})
    
    # T6: (A00-A10) * (B01+B11) -> C00 - C10
    set_term(5, {0: 1, 3: -1}, {1: 1, 4: 1}, {0: 1, 3: -1})
    
    # T7: (A01-A11) * (B10+B11) -> C11 + C10
    set_term(6, {1: 1, 4: -1}, {3: 1, 4: 1}, {4: 1, 3: 1})
    
    # Outer 15 terms (Schoolbook)
    
    # C02 (idx 2)
    set_term(7, {0: 1}, {2: 1}, {2: 1})   # A00 B02
    set_term(8, {1: 1}, {5: 1}, {2: 1})   # A01 B12
    set_term(9, {2: 1}, {8: 1}, {2: 1})   # A02 B22
    
    # C12 (idx 5)
    set_term(10, {3: 1}, {2: 1}, {5: 1})  # A10 B02
    set_term(11, {4: 1}, {5: 1}, {5: 1})  # A11 B12
    set_term(12, {5: 1}, {8: 1}, {5: 1})  # A12 B22
    
    # C20 (idx 6)
    set_term(13, {6: 1}, {0: 1}, {6: 1})  # A20 B00
    set_term(14, {7: 1}, {3: 1}, {6: 1})  # A21 B10
    set_term(15, {8: 1}, {6: 1}, {6: 1})  # A22 B20
    
    # C21 (idx 7)
    set_term(16, {6: 1}, {1: 1}, {7: 1})  # A20 B01
    set_term(17, {7: 1}, {4: 1}, {7: 1})  # A21 B11
    set_term(18, {8: 1}, {7: 1}, {7: 1})  # A22 B21
    
    # C22 (idx 8)
    set_term(19, {6: 1}, {2: 1}, {8: 1})  # A20 B02
    set_term(20, {7: 1}, {5: 1}, {8: 1})  # A21 B12
    set_term(21, {8: 1}, {8: 1}, {8: 1})  # A22 B22
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
