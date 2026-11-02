"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) -> (4, 56)
    # V: (m*p, rank) -> (32, 56)
    # W: (p*n, rank) -> (32, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen factors for 2x2x2
    # Each factor is a tuple of 3 lists: U_coeffs, V_coeffs, W_coeffs
    # Indices: U[i,j], V[j,k], W[k,i] for 2x2 matrices
    # We map these to the global indices for the 2x2x16 case.
    
    # Strassen factors (u, v, w) where each is a list of (index1, index2, coeff)
    # u: (i, j)
    # v: (j, k)
    # w: (k, i)
    
    factors = [
        # 1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # 2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # 3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # 4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # 5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # 6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # 7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            u_list, v_list, w_list = factors[s]
            
            for i, j, coeff in u_list:
                U[i * m + j][col] = coeff
                
            for j, k, coeff in v_list:
                # k is local index 0 or 1. Global col is 2*t + k
                V[j * p + (2 * t + k)][col] = coeff
                
            for k, i, coeff in w_list:
                # k is local index 0 or 1. Global row index for W is (2*t + k) * n + i
                W[(2 * t + k) * n + i][col] = coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
