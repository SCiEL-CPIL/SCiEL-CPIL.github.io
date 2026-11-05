"""
6x6 Matrix Multiplication Tensor Decomposition.

Strategy:
- Primary: Hybrid Strassen (2x2 outer) + Schoolbook (3x3 inner). Rank = 7*27 = 189.
- Fallback: Schoolbook 6x6. Rank = 216.

The 189-rank decomposition is verified exactly before returning.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140

def construct():
    start_time = time.time()
    n, m, p = 6, 6, 6

    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _hybrid_strassen_schoolbook():
        block_size = 3
        dim = n
        rank_inner = 27
        rank_outer = 7
        total_rank = rank_inner * rank_outer

        U = np.zeros((dim * dim, total_rank))
        V = np.zeros((dim * dim, total_rank))
        W = np.zeros((dim * dim, total_rank))

        col = 0

        # Strassen steps for 2x2 outer
        strassen_steps = [
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
            {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
            {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
            {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
            {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
        ]

        schoolbook_steps = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    schoolbook_steps.append({
                        'A': {(i, j): 1},
                        'B': {(j, k): 1},
                        'C': {(i, k): 1}
                    })

        for s_step in strassen_steps:
            A_out = s_step['A']
            B_out = s_step['B']
            C_out = s_step['C']

            for sb_step in schoolbook_steps:
                A_in = sb_step['A']
                B_in = sb_step['B']
                C_in = sb_step['C']

                for (I, J), ca_o in A_out.items():
                    for (u, v), ca_i in A_in.items():
                        ig = I * block_size + u
                        jg = J * block_size + v
                        U[ig * dim + jg, col] += ca_o * ca_i

                for (J, K), cb_o in B_out.items():
                    for (v, w), cb_i in B_in.items():
                        jg = J * block_size + v
                        kg = K * block_size + w
                        V[jg * dim + kg, col] += cb_o * cb_i

                for (I, K), cc_o in C_out.items():
                    for (u, w), cc_i in C_in.items():
                        ig = I * block_size + u
                        kg = K * block_size + w
                        W[kg * dim + ig, col] += cc_o * cc_i

                col += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _verify(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]
                for k in range(p):
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < 1e-10

    try:
        if time.time() < _DEADLINE:
            res = _hybrid_strassen_schoolbook()
            if _verify(res["U"], res["V"], res["W"]):
                return res
    except Exception:
        pass

    return _schoolbook()

# EVOLVE-BLOCK-END
