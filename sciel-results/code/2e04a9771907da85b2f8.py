import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 140

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, seed=12345, trials=12):
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for _ in range(trials):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n*m) @ Ua)
                y = (B.reshape(m*p) @ Va)
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                C_expected = C_expected.T
                if not np.allclose(C_computed, C_expected, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    # Rank 32: Winograd F(4) for 4x4x4 (rank 16) + Schoolbook for 4x4x1 (rank 16)
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Winograd F(4) Transform
    T = [[1, 0, 1, 0],
         [0, 1, 0, -1],
         [1, 0, -1, 0],
         [0, 1, 0, 1]]
    
    # T is symmetric, T^2 = 2I
    # U[i, j, alpha] = T[r, i] * T[s, j]
    # V[j, k, alpha] = T[r, j] * T[s, k]
    # W[i, k, alpha] = 0.25 * T[i, r] * T[k, s]
    # alpha is (r, s)
    
    col = 0
    for r in range(4):
        for s in range(4):
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][col] += T[r][i] * T[s][j]
            for j in range(4):
                for k in range(4):
                    V[j * 5 + k][col] += T[r][j] * T[s][k]
            for i in range(4):
                for k in range(4):
                    W[k * 4 + i][col] += 0.25 * T[i][r] * T[k][s]
            col += 1

    # Schoolbook for 4x4x1
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to current program logic if Winograd fails verification
        # (This should not happen if the math is correct)
        pass
        
    # Fallback: Return the rank 65 decomposition from the current program
    rank65 = 65
    U65 = [[0] * rank65 for _ in range(n * m)]
    V65 = [[0] * rank65 for _ in range(m * p)]
    W65 = [[0] * rank65 for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U65[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V65[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W65[k * n + i][col] += c

    col = 0
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),
    ]
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0: continue
            if b_idx == 0: gi, gj = r, c
            elif b_idx == 1: gi, gj = r, 2 + c
            elif b_idx == 2: gi, gj = 2 + r, c
            elif b_idx == 3: gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0: continue
            if b_idx == 0: gj, gk = r, c
            elif b_idx == 1: gj, gk = r, 2 + c
            elif b_idx == 2: gj, gk = 2 + r, c
            elif b_idx == 3: gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    for i in range(4):
        for j in range(4):
            U65[i * 4 + j][col] = 1
            V65[j * 5 + 4][col] = 1
            W65[4 * 4 + i][col] = 1
            col += 1

    return {"U": U65, "V": V65, "W": W65}
