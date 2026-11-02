"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, sign=1):
        nonlocal col
        for idx in u_idx:
            U[idx][col] += sign
        for idx in v_idx:
            V[idx][col] += sign
        for idx in w_idx:
            W[idx][col] += sign
        col += 1

    for J in range(2):
        for K in range(2):
            # Strassen Part (2x2x2) for i in {0,1}
            # Local j' in {0,1}, k' in {0,1}
            # Global j = 2J + j', k = 2K + k'
            
            # T1: (A00+A11)(B00+B11)
            add_term(
                [0*4 + (2*J+0), 1*4 + (2*J+1)], # U: A00, A11
                [(2*J+0)*4 + (2*K+0), (2*J+1)*4 + (2*K+1)], # V: B00, B11
                [(2*K+0)*3 + 0, (2*K+1)*3 + 1], # W: C00, C11
                1
            )
            
            # T2: (A10+A11)B00
            add_term(
                [1*4 + (2*J+0), 1*4 + (2*J+1)], # U: A10, A11
                [(2*J+0)*4 + (2*K+0)], # V: B00
                [(2*K+0)*3 + 1, (2*K+1)*3 + 1], # W: C10 (+), C11 (-)
                1
            )
            # Fix W signs for T2 manually? 
            # My add_term uses same sign for all.
            # T2 contributes +1 to C10 and -1 to C11.
            # So I need to handle signs per index or split terms.
            # It's easier to just add +1 and -1 separately or modify add_term.
            
    # Refactor: add_term should take lists of (index, sign) or similar.
    # Or just call add_term with sign=1 and then subtract?
    # Better: define add_term to take a list of (idx, val) for each matrix.
    
    # Let's rewrite the loop logic cleanly.
    
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_rank1(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals:
            U[idx][col] += val
        for idx, val in v_vals:
            V[idx][col] += val
        for idx, val in w_vals:
            W[idx][col] += val
        col += 1

    for J in range(2):
        for K in range(2):
            # Strassen 2x2x2
            # i=0,1; j'=0,1; k'=0,1
            
            # T1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
            add_rank1(
                [(0*4 + 2*J, 1), (1*4 + 2*J+1, 1)],
                [(2*J*4 + 2*K, 1), ((2*J+1)*4 + 2*K+1, 1)],
                [(2*K*3 + 0, 1), ((2*K+1)*3 + 1, 1)]
            )
            
            # T2: (A10+A11)B00 -> C10 += 1, C11 -= 1
            add_rank1(
                [(1*4 + 2*J, 1), (1*4 + 2*J+1, 1)],
                [(2*J*4 + 2*K, 1)],
                [(2*K*3 + 1, 1), ((2*K+1)*3 + 1, -1)]
            )
            
            # T3: A00(B01-B11) -> C00 -= 1, C01 += 1
            add_rank1(
                [(0*4 + 2*J, 1)],
                [(2*J*4 + 2*K+1, 1), ((2*J+1)*4 + 2*K+1, -1)],
                [(2*K*3 + 0, -1), ((2*K+1)*3 + 0, 1)]
            )
            
            # T4: A11(B10-B00) -> C00 += 1, C10 += 1
            add_rank1(
                [(1*4 + 2*J+1, 1)],
                [((2*J+1)*4 + 2*K, 1), (2*J*4 + 2*K, -1)],
                [(2*K*3 + 0, 1), (2*K*3 + 1, 1)]
            )
            
            # T5: (A00+A01)B11 -> C00 -= 1, C01 += 1
            add_rank1(
                [(0*4 + 2*J, 1), (0*4 + 2*J+1, 1)],
                [((2*J+1)*4 + 2*K+1, 1)],
                [(2*K*3 + 0, -1), ((2*K+1)*3 + 0, 1)]
            )
            
            # T6: (A10-A00)(B00+B01) -> C11 += 1
            add_rank1(
                [(1*4 + 2*J, 1), (0*4 + 2*J, -1)],
                [(2*J*4 + 2*K, 1), (2*J*4 + 2*K+1, 1)],
                [((2*K+1)*3 + 1, 1)]
            )
            
            # T7: (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
            add_rank1(
                [(0*4 + 2*J+1, 1), (1*4 + 2*J+1, -1)],
                [((2*J+1)*4 + 2*K, 1), ((2*J+1)*4 + 2*K+1, 1)],
                [(2*K*3 + 0, 1), ((2*K+1)*3 + 1, -1)]
            )
            
            # Schoolbook 1x2x2 for i=2
            # C20 = A20 B00 + A21 B10
            # C21 = A20 B01 + A21 B11
            
            # T8: A20 B00 -> C20 += 1
            add_rank1(
                [(2*4 + 2*J, 1)],
                [(2*J*4 + 2*K, 1)],
                [(2*K*3 + 2, 1)]
            )
            
            # T9: A21 B10 -> C20 += 1
            add_rank1(
                [(2*4 + 2*J+1, 1)],
                [((2*J+1)*4 + 2*K, 1)],
                [(2*K*3 + 2, 1)]
            )
            
            # T10: A20 B01 -> C21 += 1
            add_rank1(
                [(2*4 + 2*J, 1)],
                [(2*J*4 + 2*K+1, 1)],
                [((2*K+1)*3 + 2, 1)]
            )
            
            # T11: A21 B11 -> C21 += 1
            add_rank1(
                [(2*4 + 2*J+1, 1)],
                [((2*J+1)*4 + 2*K+1, 1)],
                [((2*K+1)*3 + 2, 1)]
            )
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
