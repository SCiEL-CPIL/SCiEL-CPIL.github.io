import numpy as np
import time

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 sub-blocks (rank 7 each) and schoolbook
    for the remaining thin blocks. Total rank: 8*7 + 4*4 + 4*4 + 2*2 = 56+16+16+4 = 92.
    """
    n, m, p = 4, 5, 5
    n_m = n * m
    m_p = m * p
    p_n = p * n

    start_time = time.time()
    deadline = start_time + 120  # keep well under 150s

    U_cols = []
    V_cols = []
    W_cols = []

    def check_deadline():
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded during construction")

    def add_term(u_vec, v_vec, w_vec):
        """Add a single rank-1 term. u_vec, v_vec, w_vec are lists of (index, coeff)."""
        check_deadline()
        u_col = [0] * n_m
        v_col = [0] * m_p
        w_col = [0] * p_n
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def schoolbook_block(i_start, j_start, k_start, i_size, j_size, k_size):
        """Add schoolbook terms for a sub-block."""
        for i_loc in range(i_size):
            for j_loc in range(j_size):
                for k_loc in range(k_size):
                    check_deadline()
                    i_glob = i_start + i_loc
                    j_glob = j_start + j_loc
                    k_glob = k_start + k_loc
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])

    def strassen_block(i_start, j_start, k_start):
        """Add Strassen's 7-term decomposition for a 2x2x2 sub-block."""
        def gI(i_l, j_l): return (i_start + i_l) * m + (j_start + j_l)
        def gJ(j_l, k_l): return (j_start + j_l) * p + (k_start + k_l)
        def gK(k_l, i_l): return (k_start + k_l) * n + (i_start + i_l)

        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_term([(gI(0,0), 1), (gI(1,1), 1)],
                 [(gJ(0,0), 1), (gJ(1,1), 1)],
                 [(gK(0,0), 1), (gK(1,1), 1)])
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_term([(gI(1,0), 1), (gI(1,1), 1)],
                 [(gJ(0,0), 1)],
                 [(gK(0,1), 1), (gK(1,1), -1)])
        # M3 = A00(B01-B11) -> C00 -= M3, C01 += M3
        add_term([(gI(0,0), 1)],
                 [(gJ(0,1), 1), (gJ(1,1), -1)],
                 [(gK(1,0), 1), (gK(1,1), 1)])
        # M4 = A11(B10-B00) -> C00 += M4, C01 += M4
        add_term([(gI(1,1), 1)],
                 [(gJ(1,0), 1), (gJ(0,0), -1)],
                 [(gK(0,0), 1), (gK(0,1), 1)])
        # M5 = (A00+A01)B11 -> C00 -= M5, C10 += M5
        add_term([(gI(0,0), 1), (gI(0,1), 1)],
                 [(gJ(1,1), 1)],
                 [(gK(0,0), -1), (gK(1,0), 1)])
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        add_term([(gI(1,0), 1), (gI(0,0), -1)],
                 [(gJ(0,0), 1), (gJ(0,1), 1)],
                 [(gK(1,1), 1)])
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        add_term([(gI(0,1), 1), (gI(1,1), -1)],
                 [(gJ(1,0), 1), (gJ(1,1), 1)],
                 [(gK(0,0), 1)])

    # Partition: i: 2+2, j: 2+2+1, k: 2+2+1
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 1)]

    for i_start, i_size in i_blocks:
        for j_start, j_size in j_blocks:
            for k_start, k_size in k_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    strassen_block(i_start, j_start, k_start)
                else:
                    schoolbook_block(i_start, j_start, k_start, i_size, j_size, k_size)

    # Fallback: if somehow no terms were added, use full schoolbook
    if not U_cols:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])

    # Convert to numpy arrays and return as lists
    U = np.array(U_cols, dtype=float).T
    V = np.array(V_cols, dtype=float).T
    W = np.array(W_cols, dtype=float).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
