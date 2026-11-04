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

    # Standard Strassen 2x2*2x2 (Rank 7)
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

    # Block Strassen for 4x4*4x4 (Rank 49)
    A_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0
        [(1, 1, 0), (1, 1, 1)],   # P1
        [(1, 0, 0)],              # P2
        [(1, 1, 1)],              # P3
        [(1, 0, 0), (1, 0, 1)],   # P4
        [(1, 1, 0), (-1, 0, 0)],  # P5
        [(1, 0, 1), (-1, 1, 1)],  # P6
    ]

    B_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0
        [(1, 0, 0)],              # P1
        [(1, 0, 1), (-1, 1, 1)],  # P2
        [(1, 1, 0), (-1, 0, 0)],  # P3
        [(1, 1, 1)],              # P4
        [(1, 0, 0), (1, 0, 1)],   # P5
        [(1, 1, 0), (1, 1, 1)],   # P6
    ]

    C_specs = [
        {(0, 0): 1, (1, 1): 1},        # P0
        {(1, 0): 1, (1, 1): -1},       # P1
        {(0, 1): 1, (1, 1): 1},        # P2
        {(0, 0): 1, (1, 0): 1},        # P3
        {(0, 0): -1, (0, 1): 1},       # P4
        {(1, 1): 1},                   # P5
        {(0, 0): 1},                   # P6
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
                                gk = C_row_off + cbc * 2 + c
                                w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + coeff_c * w_val

                if u_dict and v_dict and w_dict:
                    add_term(u_dict, v_dict, w_dict)

    # Apply for B_left and B_right
    strassen_4x4x4(A_row_off=0, A_col_off=0, B_row_off=0, B_col_off=0, C_row_off=0, C_col_off=0)
    strassen_4x4x4(A_row_off=0, A_col_off=0, B_row_off=0, B_col_off=4, C_row_off=0, C_col_off=4)

    # Verify the current decomposition
    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    def verify(U, V, W):
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-10:
                        error += abs(val - 1)
        return error

    error = verify(U, V, W)
    if error > 1e-8:
        # Fallback
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

    # Attempt rank reduction via local search
    # This is a heuristic approach. We try to merge terms that have identical U and V vectors,
    # or try to remove terms that are linearly dependent in a way that preserves the tensor.
    # Given the time budget, we'll try a simple greedy removal if possible, but mostly we rely
    # on the structural decomposition.
    
    # A more sophisticated approach would be to check if any column of U, V, W is zero, or if 
    # two columns are identical in U and V but different in W (which can be merged).
    # However, the Strassen construction is already quite tight.
    
    # Let's try to merge identical columns in U and V
    merged_U = []
    merged_V = []
    merged_W = []
    used = [False] * rank
    
    for i in range(rank):
        if used[i]:
            continue
        for j in range(i+1, rank):
            if not used[j]:
                if np.allclose(U[:, i], U[:, j]) and np.allclose(V[:, i], V[:, j]):
                    # Merge: W[:, i] += W[:, j]
                    W[:, i] += W[:, j]
                    used[j] = True
        
        if not used[i]:
            merged_U.append(U[:, i])
            merged_V.append(V[:, i])
            merged_W.append(W[:, i])
    
    if len(merged_U) < rank:
        U = np.column_stack(merged_U)
        V = np.column_stack(merged_V)
        W = np.column_stack(merged_W)
        error = verify(U, V, W)
        if error <= 1e-8:
            rank = U.shape[1]
        else:
            # Revert
            U = np.array(U_list).T
            V = np.array(V_list).T
            W = np.array(W_list).T
            rank = len(U_list)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
