import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen terms
    # Each term is defined by:
    # A_blocks: dict (I, J) -> coeff
    # B_blocks: dict (J, K) -> coeff
    # C_blocks: dict (I, K) -> coeff
    outer_terms = [
        {
            "A": {(0,0): 1, (1,1): 1},
            "B": {(0,0): 1, (1,1): 1},
            "C": {(0,0): 1, (1,1): 1}
        },
        {
            "A": {(1,0): 1, (1,1): 1},
            "B": {(0,0): 1},
            "C": {(1,0): 1, (1,1): -1}
        },
        {
            "A": {(0,0): 1},
            "B": {(0,1): 1, (1,1): -1},
            "C": {(0,1): 1, (1,1): 1}
        },
        {
            "A": {(1,1): 1},
            "B": {(1,0): 1, (0,0): -1},
            "C": {(0,0): 1, (1,0): 1}
        },
        {
            "A": {(0,0): 1, (0,1): 1},
            "B": {(1,1): 1},
            "C": {(0,1): 1, (0,0): -1}
        },
        {
            "A": {(1,0): 1, (0,0): -1},
            "B": {(0,0): 1, (0,1): 1},
            "C": {(1,1): 1}
        },
        {
            "A": {(0,1): 1, (1,1): -1},
            "B": {(1,0): 1, (1,1): 1},
            "C": {(0,0): 1, (1,1): 1}
        }
    ]
    
    # Inner Strassen terms for 2x2
    # Each term is defined by:
    # Q_indices: list of (i, j, coeff)
    # R_indices: list of (j, k, coeff)
    # P_indices: list of (i, k, coeff)
    inner_terms = [
        {
            "Q": [(0,0,1), (1,1,1)],
            "R": [(0,0,1), (1,1,1)],
            "P": [(0,0,1), (1,1,1)]
        },
        {
            "Q": [(1,0,1), (1,1,1)],
            "R": [(0,0,1)],
            "P": [(1,0,1), (1,1,-1)]
        },
        {
            "Q": [(0,0,1)],
            "R": [(0,1,1), (1,1,-1)],
            "P": [(0,1,1), (1,1,1)]
        },
        {
            "Q": [(1,1,1)],
            "R": [(1,0,1), (0,0,-1)],
            "P": [(0,0,1), (1,0,1)]
        },
        {
            "Q": [(0,0,1), (0,1,1)],
            "R": [(1,1,1)],
            "P": [(0,1,1), (0,0,-1)]
        },
        {
            "Q": [(1,0,1), (0,0,-1)],
            "R": [(0,0,1), (0,1,1)],
            "P": [(1,1,1)]
        },
        {
            "Q": [(0,1,1), (1,1,-1)],
            "R": [(1,0,1), (1,1,1)],
            "P": [(0,0,1), (1,1,1)]
        }
    ]
    
    col = 0
    for t_out in range(7):
        a_blocks = outer_terms[t_out]["A"]
        b_blocks = outer_terms[t_out]["B"]
        c_blocks = outer_terms[t_out]["C"]
        
        for t_in in range(7):
            q_terms = inner_terms[t_in]["Q"]
            r_terms = inner_terms[t_in]["R"]
            p_terms = inner_terms[t_in]["P"]
            
            # Fill U
            for (i, j, v_q) in q_terms:
                for (I, J, v_a) in a_blocks.items():
                    # Q_ij = sum v_a * A_block[I][J]
                    # We need to add v_q * v_a to all entries of A_block[I][J]
                    # A_block[I][J] local indices are i_local, j_local
                    # Global row = 2*I + i_local, col = 2*J + j_local
                    # Global index = (2*I + i_local)*4 + (2*J + j_local)
                    for i_local in range(2):
                        for j_local in range(2):
                            row = (2*I + i_local) * 4 + (2*J + j_local)
                            U[row][col] += v_q * v_a
                            
            # Fill V
            for (j, k, v_r) in r_terms:
                for (J, K, v_b) in b_blocks.items():
                    # R_jk = sum v_b * B_block[J][K]
                    # B_block[J][K] local indices are j_local, k_local
                    # Global row = 2*J + j_local, col = 2*K + k_local
                    # Global index = (2*J + j_local)*4 + (2*K + k_local)
                    for j_local in range(2):
                        for k_local in range(2):
                            row = (2*J + j_local) * 4 + (2*K + k_local)
                            V[row][col] += v_r * v_b
                            
            # Fill W
            for (i, k, v_p) in p_terms:
                for (I, K, v_c) in c_blocks.items():
                    # P_ik = sum v_c * C_block[I][K]
                    # C_block[I][K] local indices are i_local, k_local
                    # Global row = 2*I + i_local, col = 2*K + k_local
                    # W index = col * 4 + row = (2*K + k_local)*4 + (2*I + i_local)
                    for i_local in range(2):
                        for k_local in range(2):
                            row = (2*K + k_local) * 4 + (2*I + i_local)
                            W[row][col] += v_p * v_c
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
