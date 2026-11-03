"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    strassen_terms = [
        ({0: 1, 2: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    standard_terms = [
        ({0: 1}, {0: 1}, {0: 1}),
        ({1: 1}, {1: 1}, {0: 1}),
        ({2: 1}, {0: 1}, {1: 1}),
        ({3: 1}, {1: 1}, {1: 1})
    ]

    for part in range(2):
        if part == 0:
            u_map = {0: 0, 1: 1, 2: 4, 3: 5}
        else:
            u_map = {0: 2, 1: 3, 2: 6, 3: 7}
            
        for k in [0, 2, 4]:
            if part == 0:
                v_map = {0: k, 1: k+1, 2: 7+k, 3: 8+k}
            else:
                v_map = {0: 14+k, 1: 15+k, 2: 21+k, 3: 22+k}
                
            w_map = {0: 2*k, 1: 2*k+2, 2: 2*k+1, 3: 2*k+3}
            
            for u_loc, v_loc, w_loc in strassen_terms:
                u_global = {u_map[li]: v for li, v in u_loc.items()}
                v_global = {v_map[li]: v for li, v in v_loc.items()}
                w_global = {w_map[li]: v for li, v in w_loc.items()}
                add_term(u_global, v_global, w_global)
                
        k = 6
        if part == 0:
            v_map_single = {0: 6, 1: 13}
        else:
            v_map_single = {0: 20, 1: 27}
            
        w_map_single = {0: 12, 1: 13}
        
        for u_loc, v_loc, w_loc in standard_terms:
            u_global = {u_map[li]: v for li, v in u_loc.items()}
            v_global = {v_map_single[li]: v for li, v in v_loc.items()}
            w_global = {w_map_single[li]: v for li, v in w_loc.items()}
            add_term(u_global, v_global, w_global)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
