import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Uses block-diagonal Strassen (rank 21) as the base.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    # 1. Construct the base Rank 21 decomposition (Block-diagonal Strassen)
    rank_base = 21
    U_base = np.zeros((nm, rank_base))
    V_base = np.zeros((mp, rank_base))
    W_base = np.zeros((pn, rank_base))

    # Strassen coefficients for 2x2 multiplication
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]

    # Index mappings for the 3 blocks (g=0,1,2)
    v_indices = []
    w_indices = []
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        # V: j*6 + k (j is local row of B, k is local col of B)
        v_indices.append([0 * 6 + k0, 0 * 6 + k1, 1 * 6 + k0, 1 * 6 + k1])
        # W: k*2 + i (k is local row of C, i is local col of C)
        w_indices.append([k0 * 2 + 0, k1 * 2 + 0, k0 * 2 + 1, k1 * 2 + 1])

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            u_list = step['u']
            v_list = step['v']
            w_list = step['w']
            v_idx_list = v_indices[g]
            w_idx_list = w_indices[g]

            for local_idx in range(4):
                if u_list[local_idx] != 0:
                    U_base[local_idx, r] = u_list[local_idx]
                if v_list[local_idx] != 0:
                    V_base[v_idx_list[local_idx], r] = v_list[local_idx]
                if w_list[local_idx] != 0:
                    W_base[w_idx_list[local_idx], r] = w_list[local_idx]

    # Return the rank 21 decomposition
    return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank 21 decomposition generated.")
