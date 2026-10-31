import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 49 decomposition for 4x4 matrix multiplication
    # Based on Strassen's algorithm applied to 2x2 blocks
    
    # Inner Strassen vectors for 2x2 multiplication
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # u_s corresponds to A (I x J), v_s to B (J x K), w_s to C (I x K)
    inner_u = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    inner_v = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [-1, 0, 1, 0], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    inner_w = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]

    # Outer Strassen vectors for 2x2 block structure
    outer_u = inner_u
    outer_v = inner_v
    outer_w = inner_w

    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to get row, col from flat index for 2x2
    def get_2x2_indices(flat_idx):
        return flat_idx // 2, flat_idx % 2

    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            
            # Process U (A is I x J)
            for idx_u_out, val_u_out in enumerate(outer_u[r]):
                if val_u_out != 0:
                    I_block, J_block = get_2x2_indices(idx_u_out)
                    for idx_u_in, val_u_in in enumerate(inner_u[s]):
                        if val_u_in != 0:
                            i_prime, j_prime = get_2x2_indices(idx_u_in)
                            i = 2 * I_block + i_prime
                            j = 2 * J_block + j_prime
                            flat_idx_u = i * 4 + j
                            U[flat_idx_u][col] += val_u_out * val_u_in

            # Process V (B is J x K)
            for idx_v_out, val_v_out in enumerate(outer_v[r]):
                if val_v_out != 0:
                    J_block, K_block = get_2x2_indices(idx_v_out)
                    for idx_v_in, val_v_in in enumerate(inner_v[s]):
                        if val_v_in != 0:
                            j_prime, k_prime = get_2x2_indices(idx_v_in)
                            j = 2 * J_block + j_prime
                            k = 2 * K_block + k_prime
                            flat_idx_v = j * 4 + k
                            V[flat_idx_v][col] += val_v_out * val_v_in

            # Process W (C is I x K, output index k*n + i)
            for idx_w_out, val_w_out in enumerate(outer_w[r]):
                if val_w_out != 0:
                    I_block, K_block = get_2x2_indices(idx_w_out)
                    for idx_w_in, val_w_in in enumerate(inner_w[s]):
                        if val_w_in != 0:
                            i_prime, k_prime = get_2x2_indices(idx_w_in)
                            i = 2 * I_block + i_prime
                            k = 2 * K_block + k_prime
                            flat_idx_w = k * 4 + i
                            W[flat_idx_w][col] += val_w_out * val_w_in

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
