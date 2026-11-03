"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    rank = 50
    
    # U shape: (n*m, rank) = (8, 50)
    # V shape: (m*p, rank) = (28, 50)
    # W shape: (p*n, rank) = (14, 50)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to set values in a column
    def set_col(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    for j_block in range(2):
        # j indices: 2*j_block, 2*j_block + 1
        j0 = 2 * j_block
        j1 = 2 * j_block + 1
        
        # U indices for A block
        # A[i, j] -> index i*m + j
        u_a00 = 0 * m + j0
        u_a01 = 0 * m + j1
        u_a10 = 1 * m + j0
        u_a11 = 1 * m + j1
        
        for k_block in range(3):
            # k indices: 2*k_block, 2*k_block + 1
            k0 = 2 * k_block
            k1 = 2 * k_block + 1
            
            # V indices for B block
            # B[j, k] -> index j*p + k
            v_b00 = j0 * p + k0
            v_b01 = j0 * p + k1
            v_b10 = j1 * p + k0
            v_b11 = j1 * p + k1
            
            # W indices for C block
            # C[i, k] -> index k*n + i
            w_z00 = k0 * n + 0
            w_z01 = k1 * n + 0
            w_z10 = k0 * n + 1
            w_z11 = k1 * n + 1
            
            # Strassen Terms
            
            # M1: (a10 + a11) * b00
            set_col(
                {u_a10: 1, u_a11: 1},
                {v_b00: 1},
                {w_z01: 1, w_z11: -1}
            )
            
            # M2: (a10 + a11) * (b01 - b11)
            set_col(
                {u_a10: 1, u_a11: 1},
                {v_b01: 1, v_b11: -1},
                {w_z00: -1, w_z01: 1}
            )
            
            # M3: (a00 + a01) * b10
            set_col(
                {u_a00: 1, u_a01: 1},
                {v_b10: 1},
                {w_z10: 1, w_z11: 1}
            )
            
            # M4: a00 * (b11 - b00)
            set_col(
                {u_a00: 1},
                {v_b11: 1, v_b00: -1},
                {w_z00: 1, w_z10: 1}
            )
            
            # M5: (a01 + a11) * (b00 + b11)
            set_col(
                {u_a01: 1, u_a11: 1},
                {v_b00: 1, v_b11: 1},
                {w_z00: 1, w_z11: 1}
            )
            
            # M6: (a00 - a10) * (b10 + b11)
            set_col(
                {u_a00: 1, u_a10: -1},
                {v_b10: 1, v_b11: 1},
                {w_z00: 1}
            )
            
            # M7: (a01 - a11) * (b00 + b10)
            set_col(
                {u_a01: 1, u_a11: -1},
                {v_b00: 1, v_b10: 1},
                {w_z11: -1}
            )
            
        # Schoolbook part for k=6
        k6 = 6
        
        # V indices for B col 6
        v_b0_6 = j0 * p + k6
        v_b1_6 = j1 * p + k6
        
        # W indices for C col 6
        w_z0_6 = k6 * n + 0
        w_z1_6 = k6 * n + 1
        
        # Term 1: A[0, j0] * B[j0, 6] -> C[0, 6]
        set_col(
            {u_a00: 1},
            {v_b0_6: 1},
            {w_z0_6: 1}
        )
        
        # Term 2: A[0, j1] * B[j1, 6] -> C[0, 6]
        set_col(
            {u_a01: 1},
            {v_b1_6: 1},
            {w_z0_6: 1}
        )
        
        # Term 3: A[1, j0] * B[j0, 6] -> C[1, 6]
        set_col(
            {u_a10: 1},
            {v_b0_6: 1},
            {w_z1_6: 1}
        )
        
        # Term 4: A[1, j1] * B[j1, 6] -> C[1, 6]
        set_col(
            {u_a11: 1},
            {v_b1_6: 1},
            {w_z1_6: 1}
        )
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
