"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen products definition
    # (A_factors, B_factors)
    # A_factors: list of (elem_idx, coeff)
    # B_factors: list of (elem_idx, coeff)
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]
    
    # C reconstruction
    # C_elem: list of (prod_idx, coeff)
    c_reconstruction = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]
    
    # Instances: (a_col_off, b_row_off, b_col_off)
    instances = [
        (0, 0, 0),
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2),
        (0, 0, 4),
        (0, 0, 6),
        (2, 2, 4),
        (2, 2, 6),
    ]
    
    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for prod_idx in range(7):
            a_factors, b_factors = strassen_products[prod_idx]
            for a_elem, a_coeff in a_factors:
                a_row = a_elem // 2
                a_col = a_elem % 2
                u_idx = a_row * m + (a_col_off + a_col)
                U[u_idx, col] += a_coeff
            for b_elem, b_coeff in b_factors:
                b_row = b_elem // 2
                b_col = b_elem % 2
                v_idx = (b_row_off + b_row) * p + (b_col_off + b_col)
                V[v_idx, col] += b_coeff
            col += 1
            
    # W contributions
    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for c_elem in range(4):
            c_row = c_elem // 2
            c_col = c_elem % 2
            w_idx = (b_col_off + c_col) * n + c_row
            for prod_idx, coeff in c_reconstruction[c_elem]:
                W[w_idx, col + prod_idx] += coeff
        col += 7
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
