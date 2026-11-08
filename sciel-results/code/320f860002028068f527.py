import numpy as np
import time

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    """
    start_time = time.time()
    DEADLINE = 120  # seconds

    n, m, p = 4, 4, 9
    u_dim = n * m  # 16
    v_dim = m * p  # 36
    w_dim = p * n  # 36

    U_list = []
    V_list = []
    W_list = []

    # Strassen 2x2x2 terms: (A_mask, B_mask, C_mask)
    s2_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def add_strassen_4x4(b_col_off):
        for oa, ob, oc in s2_terms:
            for ia, ib, ic in s2_terms:
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36

                for (oor, ooc), c_ao in oa.items():
                    for (ior, ioc), c_ia in ia.items():
                        r = oor * 2 + ior
                        c = ooc * 2 + ioc
                        u[r * 4 + c] += c_ao * c_ia

                for (oor, ooc), c_bo in ob.items():
                    for (ior, ioc), c_ib in ib.items():
                        r = oor * 2 + ior
                        c = b_col_off + ooc * 2 + ioc
                        v[r * 9 + c] += c_bo * c_ib

                for (oor, ooc), c_co in oc.items():
                    for (ior, ioc), c_ic in ic.items():
                        r = oor * 2 + ior
                        c = b_col_off + ooc * 2 + ioc
                        w[c * 4 + r] += c_co * c_ic

                add_term(u, v, w)

    def add_schoolbook_4x1(b_col):
        for i in range(4):
            for j in range(4):
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36

                u[i * 4 + j] = 1.0
                v[j * 9 + b_col] = 1.0
                w[b_col * 4 + i] = 1.0

                add_term(u, v, w)

    # Add Strassen for B cols 0-3
    add_strassen_4x4(0)

    # Add Strassen for B cols 4-7
    add_strassen_4x4(4)

    # Add Schoolbook for B col 8
    add_schoolbook_4x1(8)

    rank = len(U_list)

    # Build output matrices
    U = np.array(U_list).T  # shape (16, rank)
    V = np.array(V_list).T  # shape (36, rank)
    W = np.array(W_list).T  # shape (36, rank)

    # Convert to lists for the required output format
    U_list_out = U.tolist()
    V_list_out = V.tolist()
    W_list_out = W.tolist()

    return {"U": U_list_out, "V": V_list_out, "W": W_list_out}
