"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    # U: (n*m, rank) = (6, 22)
    # V: (m*p, rank) = (12, 22)
    # W: (p*n, rank) = (8, 22)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, c=1):
        for idx, val in zip(u_indices, [c]*len(u_indices)):
            U[idx][current_col] += val
        for idx, val in zip(v_indices, [c]*len(v_indices)):
            V[idx][current_col] += val
        for idx, val in zip(w_indices, [c]*len(w_indices)):
            W[idx][current_col] += val
            
    # Indices for A (2x3): 0:a11, 1:a12, 2:a13, 3:a21, 4:a22, 5:a23
    # Indices for B (3x4): 0:b11, 1:b12, 2:b13, 3:b14, 4:b21, 5:b22, 6:b23, 7:b24, 8:b31, 9:b32, 10:b33, 11:b34
    # Indices for C (2x4): 0:c11, 1:c12, 2:c13, 3:c14, 4:c21, 5:c22, 6:c23, 7:c24
    
    current_col = 0
    
    # Part 1: A22 * B22_1 (cols 0,1 of B)
    # M1: (a11+a22)(b11+b22) -> c11, c22
    add_term([0, 3], [0, 5], [0, 5])
    current_col += 1
    
    # M2: (a21+a22)b11 -> c21, -c22
    add_term([3, 4], [0], [4, 5])
    # Wait, U indices for a21+a22 are 3,4? 
    # A indices: 0:a11, 1:a12, 2:a13, 3:a21, 4:a22, 5:a23
    # a21 is 3, a22 is 4.
    # So U indices [3, 4].
    # W indices: c21 is 4, c22 is 5.
    # Coefficients: c21 gets +1, c22 gets -1.
    # My add_term adds 'c' to all. I need to handle signs.
    
    # Let's rewrite add_term to handle lists of (index, coeff)
    def add_term_full(u_list, v_list, w_list):
        for idx, val in u_list:
            U[idx][current_col] += val
        for idx, val in v_list:
            V[idx][current_col] += val
        for idx, val in w_list:
            W[idx][current_col] += val

    current_col = 0
    
    # M1
    add_term_full([(0,1), (4,1)], [(0,1), (5,1)], [(0,1), (5,1)])
    current_col += 1
    
    # M2: (a21+a22)b11 -> c21(1), c22(-1)
    add_term_full([(3,1), (4,1)], [(0,1)], [(4,1), (5,-1)])
    current_col += 1
    
    # M3: a11(b21+b22) -> c12(1), c22(1)
    add_term_full([(0,1)], [(4,1), (5,1)], [(1,1), (5,1)])
    current_col += 1
    
    # M4: a22(b11-b21) -> c11(1), c21(1)
    add_term_full([(4,1)], [(0,1), (4,-1)], [(0,1), (4,1)])
    current_col += 1
    
    # M5: (a11+a12)b22 -> c11(-1), c12(1)
    add_term_full([(0,1), (1,1)], [(5,1)], [(0,-1), (1,1)])
    current_col += 1
    
    # M6: (a21-a11)(b11+b12) -> c22(-1)
    add_term_full([(3,1), (0,-1)], [(0,1), (1,1)], [(5,-1)])
    current_col += 1
    
    # M7: (a12-a22)(b21+b22) -> c11(1)
    add_term_full([(1,1), (4,-1)], [(4,1), (5,1)], [(0,1)])
    current_col += 1
    
    # Part 2: A22 * B22_2 (cols 2,3 of B)
    # N1: (a11+a22)(b13+b24) -> c13(1), c24(1)
    add_term_full([(0,1), (4,1)], [(2,1), (7,1)], [(2,1), (7,1)])
    current_col += 1
    
    # N2: (a21+a22)b13 -> c23(1), c24(-1)
    add_term_full([(3,1), (4,1)], [(2,1)], [(6,1), (7,-1)])
    current_col += 1
    
    # N3: a11(b23+b24) -> c14(1), c24(1)
    add_term_full([(0,1)], [(6,1), (7,1)], [(3,1), (7,1)])
    current_col += 1
    
    # N4: a22(b13-b23) -> c13(1), c23(1)
    add_term_full([(4,1)], [(2,1), (6,-1)], [(2,1), (6,1)])
    current_col += 1
    
    # N5: (a11+a12)b24 -> c13(-1), c14(1)
    add_term_full([(0,1), (1,1)], [(7,1)], [(2,-1), (3,1)])
    current_col += 1
    
    # N6: (a21-a11)(b13+b14) -> c24(-1)
    add_term_full([(3,1), (0,-1)], [(2,1), (3,1)], [(7,-1)])
    current_col += 1
    
    # N7: (a12-a22)(b23+b24) -> c13(1)
    add_term_full([(1,1), (4,-1)], [(6,1), (7,1)], [(2,1)])
    current_col += 1
    
    # Part 3: A21 * B14
    # T1: a13*b31 -> c11(1)
    add_term_full([(2,1)], [(8,1)], [(0,1)])
    current_col += 1
    
    # T2: a13*b32 -> c12(1)
    add_term_full([(2,1)], [(9,1)], [(1,1)])
    current_col += 1
    
    # T3: a13*b33 -> c13(1)
    add_term_full([(2,1)], [(10,1)], [(2,1)])
    current_col += 1
    
    # T4: a13*b34 -> c14(1)
    add_term_full([(2,1)], [(11,1)], [(3,1)])
    current_col += 1
    
    # T5: a23*b31 -> c21(1)
    add_term_full([(5,1)], [(8,1)], [(4,1)])
    current_col += 1
    
    # T6: a23*b32 -> c22(1)
    add_term_full([(5,1)], [(9,1)], [(5,1)])
    current_col += 1
    
    # T7: a23*b33 -> c23(1)
    add_term_full([(5,1)], [(10,1)], [(6,1)])
    current_col += 1
    
    # T8: a23*b34 -> c24(1)
    add_term_full([(5,1)], [(11,1)], [(7,1)])
    current_col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
