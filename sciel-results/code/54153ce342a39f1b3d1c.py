import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # Target rank based on structural decomposition
    # Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # Split B (4x5) into B1 (2x5, rows 0-1) and B2 (2x5, rows 2-3)
    # C = A1*B1 + A2*B2
    # Each term is (2x2) * (2x5)
    # Split (2x5) into (2x2), (2x2), (2x1)
    # Rank for (2x2)*(2x2) is 7 (Strassen)
    # Rank for (2x2)*(2x1) is 4 (Schoolbook)
    # Rank per block = 7 + 7 + 4 = 18
    # Total rank = 18 + 18 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0

    # Helper to add a rank-1 term defined by sparse dictionaries
    # u_dict: {global_U_index: coeff}
    # v_dict: {global_V_index: coeff}
    # w_dict: {global_W_index: coeff}
    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    # Define index mapping helpers
    # A index: i*m + j
    # B index: j*p + k
    # C index (W): k*n + i
    
    def idx_a(i, j):
        return i * m + j

    def idx_b(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    # Add Strassen block for (2x2) * (2x2)
    # a_i, a_j: global row, col start of A block
    # b_j, b_k: global row, col start of B block
    # c_i, c_k: global row, col start of C block (output)
    def add_strassen_2x2(a_i, a_j, b_j, b_k, c_i, c_k):
        # Local indices for 2x2:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # M1 = (A00+A11)(B00+B11)
        u = {idx_a(a_i, a_j): 1, idx_a(a_i+1, a_j+1): 1}
        v = {idx_b(b_j, b_k): 1, idx_b(b_j+1, b_k+1): 1}
        w = {idx_w(c_k, c_i): 1, idx_w(c_k+1, c_i+1): 1}
        add_rank1_term(u, v, w)

        # M2 = (A10+A11)B00
        u = {idx_a(a_i+1, a_j): 1, idx_a(a_i+1, a_j+1): 1}
        v = {idx_b(b_j, b_k): 1}
        w = {idx_w(c_k, c_i+1): 1, idx_w(c_k+1, c_i+1): -1}
        add_rank1_term(u, v, w)

        # M3 = A00(B01-B11)
        u = {idx_a(a_i, a_j): 1}
        v = {idx_b(b_j, b_k+1): 1, idx_b(b_j+1, b_k+1): -1}
        w = {idx_w(c_k+1, c_i): 1, idx_w(c_k+1, c_i+1): 1}
        add_rank1_term(u, v, w)

        # M4 = A11(B10-B00)
        u = {idx_a(a_i+1, a_j+1): 1}
        v = {idx_b(b_j+1, b_k): 1, idx_b(b_j, b_k): -1}
        w = {idx_w(c_k, c_i): 1, idx_w(c_k, c_i+1): 1}
        add_rank1_term(u, v, w)

        # M5 = (A00+A01)B11
        u = {idx_a(a_i, a_j): 1, idx_a(a_i, a_j+1): 1}
        v = {idx_b(b_j+1, b_k+1): 1}
        w = {idx_w(c_k, c_i): -1, idx_w(c_k+1, c_i): 1}
        add_rank1_term(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = {idx_a(a_i+1, a_j): 1, idx_a(a_i, a_j): -1}
        v = {idx_b(b_j, b_k): 1, idx_b(b_j, b_k+1): 1}
        w = {idx_w(c_k+1, c_i+1): 1}
        add_rank1_term(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = {idx_a(a_i, a_j+1): 1, idx_a(a_i+1, a_j+1): -1}
        v = {idx_b(b_j+1, b_k): 1, idx_b(b_j+1, b_k+1): 1}
        w = {idx_w(c_k, c_i): 1}
        add_rank1_term(u, v, w)

    # Add Schoolbook block for (2x2) * (2x1)
    # A is 2x2, B is 2x1, C is 2x1
    # a_i, a_j: global row, col start of A block
    # b_j, b_k: global row, col start of B block (only 1 col)
    # c_i, c_k: global row, col start of C block (only 1 col)
    def add_schoolbook_2x2_1(a_i, a_j, b_j, b_k, c_i, c_k):
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # Term 1: A00 * B00 -> C00
        u = {idx_a(a_i, a_j): 1}
        v = {idx_b(b_j, b_k): 1}
        w = {idx_w(c_k, c_i): 1}
        add_rank1_term(u, v, w)
        
        # Term 2: A01 * B10 -> C00
        u = {idx_a(a_i, a_j+1): 1}
        v = {idx_b(b_j+1, b_k): 1}
        w = {idx_w(c_k, c_i): 1}
        add_rank1_term(u, v, w)
        
        # Term 3: A10 * B00 -> C10
        u = {idx_a(a_i+1, a_j): 1}
        v = {idx_b(b_j, b_k): 1}
        w = {idx_w(c_k, c_i+1): 1}
        add_rank1_term(u, v, w)
        
        # Term 4: A11 * B10 -> C10
        u = {idx_a(a_i+1, a_j+1): 1}
        v = {idx_b(b_j+1, b_k): 1}
        w = {idx_w(c_k, c_i+1): 1}
        add_rank1_term(u, v, w)

    # Block 1: A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-4)
    # A1 is at a_i=0, a_j=0
    # B1 is at b_j=0, b_k=0
    # C1 is at c_i=0, c_k=0
    
    # Sub-block 1.1: A1 * B1[:, 0:2] (2x2 * 2x2)
    # A1 is 2x2 at (0,0)
    # B1_sub1 is 2x2 at (0,0)
    # C1_sub1 is 2x2 at (0,0)
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # Sub-block 1.2: A1 * B1[:, 2:4] (2x2 * 2x2)
    # A1 is 2x2 at (0,0)
    # B1_sub2 is 2x2 at (0,2)
    # C1_sub2 is 2x2 at (0,2)
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    
    # Sub-block 1.3: A1 * B1[:, 4:5] (2x2 * 2x1)
    # A1 is 2x2 at (0,0)
    # B1_sub3 is 2x1 at (0,4)
    # C1_sub3 is 2x1 at (0,4)
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)

    # Block 2: A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-4)
    # A2 is at a_i=0, a_j=2
    # B2 is at b_j=2, b_k=0
    # C2 is at c_i=0, c_k=0 (Note: C columns are determined by B columns, so C2 adds to same C cols as C1)
    
    # Sub-block 2.1: A2 * B2[:, 0:2] (2x2 * 2x2)
    # A2 is 2x2 at (0,2)
    # B2_sub1 is 2x2 at (2,0)
    # C2_sub1 is 2x2 at (0,0)
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # Sub-block 2.2: A2 * B2[:, 2:4] (2x2 * 2x2)
    # A2 is 2x2 at (0,2)
    # B2_sub2 is 2x2 at (2,2)
    # C2_sub2 is 2x2 at (0,2)
    add_strassen_2x2(0, 2, 2, 2, 0, 2)
    
    # Sub-block 2.3: A2 * B2[:, 4:5] (2x2 * 2x1)
    # A2 is 2x2 at (0,2)
    # B2_sub3 is 2x1 at (2,4)
    # C2_sub3 is 2x1 at (0,4)
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)

    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Construct tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i, j, k] == 1 for all i, j, k
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U_np[i * m + j, :]
                v_vec = V_np[j * p + k, :]
                w_vec = W_np[k * n + i, :]
                # Term is sum_r u_vec[r] * v_vec[r] * w_vec[r]
                val = np.sum(u_vec * v_vec * w_vec)
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    # print(f"Invalid at i={i}, j={j}, k={k}, val={val}")
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_schoolbook = n * m * p
        U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
        V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
        W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
