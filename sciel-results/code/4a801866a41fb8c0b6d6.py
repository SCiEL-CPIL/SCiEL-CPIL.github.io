import numpy as np
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    The decomposition is based on the recursive application of Strassen's algorithm
    for 2x2 matrix multiplication.
    
    The output is a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    Entries are integers.
    """
    n = 4
    rank = 49
    
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Strassen 2x2 combinations for A, B, C
    # Each combo is a list of (row, col, sign)
    A_combos = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    B_combos = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    C_combos = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 1, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    # Generate the 49 terms
    for t_outer in range(7):
        for t_inner in range(7):
            r_idx = t_outer * 7 + t_inner
            
            # A (U) part
            for (br, bc, sign) in A_combos[t_outer]:
                for (ir, ik, sign_inner) in A_combos[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    u_idx = i_global * n + k_global
                    U[u_idx, r_idx] += sign * sign_inner
            
            # B (V) part
            for (br, bc, sign) in B_combos[t_outer]:
                for (ir, ik, sign_inner) in B_combos[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    v_idx = i_global * n + k_global
                    V[v_idx, r_idx] += sign * sign_inner
            
            # C (W) part
            for (br, bc, sign) in C_combos[t_outer]:
                for (ir, ik, sign_inner) in C_combos[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    w_idx = k_global * n + i_global
                    W[w_idx, r_idx] += sign * sign_inner

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
