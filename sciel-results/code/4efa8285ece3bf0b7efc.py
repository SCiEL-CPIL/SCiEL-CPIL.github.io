import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    # Strassen 2x2*2x2 decomposition (rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: M1 = (a00+a11)(b00+b11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1: M2 = (a10+a11)*b00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2: M3 = a00*(b01-b11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3: M4 = a11*(b10-b00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4: M5 = (a00+a01)*b11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5: M6 = (a10-a00)*(b00+b01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6: M7 = (a01-a11)*(b10+b11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen for 4x4*4x4 (rank 49)
    A_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0: A00 + A11
        [(1, 1, 0), (1, 1, 1)],   # P1: A10 + A11
        [(1, 0, 0)],              # P2: A00
        [(1, 1, 1)],              # P3: A11
        [(1, 0, 0), (1, 0, 1)],   # P4: A00 + A01
        [(1, 1, 0), (-1, 0, 0)],  # P5: A10 - A00
        [(1, 0, 1), (-1, 1, 1)],  # P6: A01 - A11
    ]

    B_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0: B00 + B11
        [(1, 0, 0)],              # P1: B00
        [(1, 0, 1), (-1, 1, 1)],  # P2: B01 - B11
        [(1, 1, 0), (-1, 0, 0)],  # P3: B10 - B00
        [(1, 1, 1)],              # P4: B11
        [(1, 0, 0), (1, 0, 1)],   # P5: B00 + B01
        [(1, 1, 0), (1, 1, 1)],   # P6: B10 + B11
    ]

    C_specs = [
        {(0, 0): 1, (1, 1): 1},        # P0: C00 + C11
        {(1, 0): 1, (1, 1): -1},       # P1: C10 - C11
        {(0, 1): 1, (1, 1): 1},        # P2: C01 + C11
        {(0, 0): 1, (1, 0): 1},        # P3: C00 + C10
        {(0, 0): -1, (0, 1): 1},       # P4: -C00 + C01
        {(1, 1): 1},                   # P5: C11
        {(0, 0): 1},                   # P6: C00
    ]

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_dict, v_dict, w_dict):
        U_row = np.zeros(nm)
        V_row = np.zeros(mp)
        W_row = np.zeros(pn)
        for (i, j), val in u_dict.items():
            U_row[i * m + j] += val
        for (j, k), val in v_dict.items():
            V_row[j * p + k] += val
        for (k, i), val in w_dict.items():
            W_row[k * n + i] += val
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    def strassen_4x4x4(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        for k_p in range(7):
            for m_s in range(7):
                u_dict = {}
                v_dict = {}
                w_dict = {}

                for (coeff, br, bc) in A_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            u_val = U_2[r * 2 + c, m_s]
                            if u_val != 0:
                                gi = A_row_off + br * 2 + r
                                gj = A_col_off + bc * 2 + c
                                u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + coeff * u_val

                for (coeff, br, bc) in B_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            v_val = V_2[r * 2 + c, m_s]
                            if v_val != 0:
                                gj = B_row_off + br * 2 + r
                                gk = B_col_off + bc * 2 + c
                                v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + coeff * v_val

                for (cbr, cbc), coeff_c in C_specs[k_p].items():
                    for r in range(2):
                        for c in range(2):
                            w_val = W_2[r * 2 + c, m_s]
                            if w_val != 0:
                                gi = C_row_off + cbr * 2 + r
                                gk = C_col_off + cbc * 2 + c
                                w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + coeff_c * w_val

                if u_dict and v_dict and w_dict:
                    add_term(u_dict, v_dict, w_dict)

    # Apply for B_left (cols 0-3) and B_right (cols 4-7)
    strassen_4x4x4(A_row_off=0, A_col_off=0, B_row_off=0, B_col_off=0, C_row_off=0, C_col_off=0)
    strassen_4x4x4(A_row_off=0, A_col_off=0, B_row_off=0, B_col_off=4, C_row_off=0, C_col_off=4)

    rank = len(U_list)
    U = np.array(U_list).T  # shape (nm, rank)
    V = np.array(V_list).T  # shape (mp, rank)
    W = np.array(W_list).T  # shape (pn, rank)

    # Fast vectorized verification
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want T[i,j,k] = 1 for all valid (i,j,k)
    # Precompute indices
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_flat_idx = i_idx * m + j_idx  # shape (n*m*p,)
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    def verify_fast(U_mat, V_mat, W_mat):
        """Vectorized verification. Returns max absolute error."""
        # For each (i,j,k), compute sum_r U[u_flat_idx[r], r] * V[v_flat_idx[r], r] * W[w_flat_idx[r], r]
        # This is: element-wise product of U[u_flat_idx, :], V[v_flat_idx, :], W[w_flat_idx, :] then sum over rank
        prod = U_mat[u_flat_idx, :] * V_mat[v_flat_idx, :] * W_mat[w_flat_idx, :]
        tensor_vals = prod.sum(axis=1)  # shape (n*m*p,)
        return np.max(np.abs(tensor_vals - 1.0))

    error = verify_fast(U, V, W)
    if error > 1e-8:
        U_school = np.zeros((nm, 128))
        V_school = np.zeros((mp, 128))
        W_school = np.zeros((pn, 128))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

    # Greedy rank reduction with vectorized verification
    if time.time() < deadline - 30:
        removed = set()
        current_rank = rank
        
        # Precompute norms for ordering
        norms = np.array([
            np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            for r in range(rank)
        ])
        order = np.argsort(norms)
        
        # Multiple passes
        for pass_num in range(3):
            if time.time() > deadline - 25:
                break
            progress = False
            for r in order:
                if r in removed or time.time() > deadline - 25:
                    continue
                
                # Build test matrices excluding r and removed terms
                keep_mask = np.ones(rank, dtype=bool)
                keep_mask[r] = False
                keep_mask[list(removed)] = False
                n_keep = keep_mask.sum()
                
                if n_keep == 0:
                    break
                
                U_test = U[:, keep_mask]
                V_test = V[:, keep_mask]
                W_test = W[:, keep_mask]
                
                err = verify_fast(U_test, V_test, W_test)
                if err < 1e-7:
                    removed.add(r)
                    progress = True
                    current_rank -= 1
            
            if not progress:
                break
        
        if removed:
            keep = [r for r in range(rank) if r not in removed]
            U_red = U[:, keep]
            V_red = V[:, keep]
            W_red = W[:, keep]
            
            # Final verification
            err_red = verify_fast(U_red, V_red, W_red)
            if err_red < 1e-7:
                U = U_red
                V = V_red
                W = W_red

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
