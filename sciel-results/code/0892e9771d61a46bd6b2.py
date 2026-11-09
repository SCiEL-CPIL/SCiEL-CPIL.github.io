"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors
    # u_factors: list of 7 lists of (i, j, coeff)
    u_factors = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    v_factors = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    w_factors = [
        [(0, 0, 1), (1, 1, 1), (0, 1, -1), (1, 0, 1)], # z00
        [(0, 0, -1), (0, 1, 1), (1, 0, 1)],           # z01
        [(1, 0, 1), (1, 1, 1), (1, 1, -1), (1, 0, -1)], # z10
        [(1, 0, -1), (1, 1, 1), (0, 0, 1), (1, 1, 1)]   # z11
    ]
    
    # Define the 8 blocks
    # Each block: (a_map, b_map, c_map)
    # a_map: {(0,0): idx, ...}
    # b_map: {(0,0): idx, ...}
    # c_map: {(0,0): idx, ...}
    
    def get_a_idx(i, j):
        return i * m + j
        
    def get_b_idx(i, j):
        return i * p + j
        
    def get_c_idx(k, i):
        return k * n + i

    blocks = []
    
    # Helper to create a block
    def make_block(a_offset, b_j_offset, b_k_offset):
        a_map = {}
        a_map[(0,0)] = get_a_idx(0, 0 + a_offset)
        a_map[(0,1)] = get_a_idx(0, 1 + a_offset)
        a_map[(1,0)] = get_a_idx(1, 0 + a_offset)
        a_map[(1,1)] = get_a_idx(1, 1 + a_offset)
        
        b_map = {}
        b_map[(0,0)] = get_b_idx(0 + b_j_offset, 0 + b_k_offset)
        b_map[(0,1)] = get_b_idx(0 + b_j_offset, 1 + b_k_offset)
        b_map[(1,0)] = get_b_idx(1 + b_j_offset, 0 + b_k_offset)
        b_map[(1,1)] = get_b_idx(1 + b_j_offset, 1 + b_k_offset)
        
        c_map = {}
        c_map[(0,0)] = get_c_idx(0 + b_k_offset, 0)
        c_map[(0,1)] = get_c_idx(1 + b_k_offset, 0)
        c_map[(1,0)] = get_c_idx(0 + b_k_offset, 1)
        c_map[(1,1)] = get_c_idx(1 + b_k_offset, 1)
        
        return (a_map, b_map, c_map)
        
    # a_offset 0 is A_L, 2 is A_R
    # b_j_offset 0 is top half, 2 is bottom half
    # b_k_offset 0, 2, 4, 6 for the 4 column pairs
    
    # Block 0: A_L, B(0,1)x(0,1)
    blocks.append(make_block(0, 0, 0))
    # Block 1: A_L, B(0,1)x(2,3)
    blocks.append(make_block(0, 0, 2))
    # Block 2: A_R, B(2,3)x(0,1)
    blocks.append(make_block(2, 2, 0))
    # Block 3: A_R, B(2,3)x(2,3)
    blocks.append(make_block(2, 2, 2))
    # Block 4: A_L, B(0,1)x(4,5)
    blocks.append(make_block(0, 0, 4))
    # Block 5: A_L, B(0,1)x(6,7)
    blocks.append(make_block(0, 0, 6))
    # Block 6: A_R, B(2,3)x(4,5)
    blocks.append(make_block(2, 2, 4))
    # Block 7: A_R, B(2,3)x(6,7)
    blocks.append(make_block(2, 2, 6))
    
    for b_idx, (a_map, b_map, c_map) in enumerate(blocks):
        col_base = b_idx * 7
        for k in range(7):
            col = col_base + k
            
            # U
            for (i, j, coeff) in u_factors[k]:
                U[a_map[(i, j)]][col] += coeff
                
            # V
            for (i, j, coeff) in v_factors[k]:
                V[b_map[(i, j)]][col] += coeff
                
            # W
            # w_factors[k] is not defined per k, but per output cell.
            # We need to add contributions from m_k to the output cells.
            # m_k is the product u_k v_k.
            # The output Z is a linear combination of m's.
            # So for each output cell z_ij, we sum c * m_k.
            # This means W[z_idx][col] += c.
            
            # Let's iterate over output cells
            for (zi, zj) in [(0,0), (0,1), (1,0), (1,1)]:
                # Find coefficient of m_k in z_ij
                # w_factors is defined as list of 4 lists, one per z_ij
                # w_factors[zi][zj] is list of (m_idx, coeff)
                
                # My w_factors definition above was:
                # w_factors[0] -> z00
                # w_factors[1] -> z01
                # w_factors[2] -> z10
                # w_factors[3] -> z11
                
                z_idx_in_list = zi * 2 + zj
                factors_for_z = w_factors[z_idx_in_list]
                
                coeff = 0
                for (mk, c) in factors_for_z:
                    if mk == k:
                        coeff += c
                        
                if coeff != 0:
                    W[c_map[(zi, zj)]][col] += coeff
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
