import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m, dtype=float)
        v_col = np.zeros(m_p, dtype=float)
        w_col = np.zeros(p_n, dtype=float)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def schoolbook(n_prime, m_prime, p_prime, i_offset, j_offset, k_offset):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])

    def strassen_2x2x2(i_offset, j_offset, k_offset):
        def gI(i_l, j_l): return (i_offset + i_l) * m + (j_offset + j_l)
        def gJ(j_l, k_l): return (j_offset + j_l) * p + (k_offset + k_l)
        def gK(k_l, i_l): return (k_offset + k_l) * n + (i_offset + i_l)

        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        # M2: (A10+A11)B00 -> C10, -C11
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        # M3: A00(B01-B11) -> C01, C11
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        # M4: A11(B10-B00) -> C00, C10
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        # M5: (A00+A01)B11 -> -C00, C01
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )

    # Assemble 4x5x5 as 4 blocks of 2x5x5
    # Block 1: i in {0,1}, j in {0..4}, k in {0..4}
    #   Sub-blocks: (2,2,2) Strassen, (2,3,2) school, (2,2,3) school, (2,3,3) school
    strassen_2x2x2(0, 0, 0)       # 2x2x2
    schoolbook(2, 3, 2, 0, 2, 0)  # 2x3x2
    schoolbook(2, 2, 3, 0, 0, 2)  # 2x2x3
    schoolbook(2, 3, 3, 0, 2, 2)  # 2x3x3

    # Block 2: i in {2,3}, j in {0..4}, k in {0..4}
    strassen_2x2x2(2, 0, 0)       # 2x2x2
    schoolbook(2, 3, 2, 2, 2, 0)  # 2x3x2
    schoolbook(2, 2, 3, 2, 0, 2)  # 2x2x3
    schoolbook(2, 3, 3, 2, 2, 2)  # 2x3x3

    if not U_cols:
        return {"U": np.zeros((n_m, 1)).tolist(),
                "V": np.zeros((m_p, 1)).tolist(),
                "W": np.zeros((p_n, 1)).tolist()}

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
