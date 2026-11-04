import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin from 150s limit

    n, m, p = 4, 4, 8
    nm = n * m  # 16
    mp = m * p  # 32
    pn = p * n  # 32
    
    # Base 2x2 Strassen decomposition
    # U, V, W are 4x7 matrices corresponding to 2x2 blocks
    # Indices 0:00, 1:01, 2:10, 3:11
    
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

    # Pre-compute non-zero indices for U_2, V_2, W_2 to speed up construction
    # For each term c in 0..6, we need lists of (index, value)
    U_2_sparse = []
    V_2_sparse = []
    W_2_sparse = []
    for c in range(7):
        u_list = [(idx, U_2[idx, c]) for idx in range(4) if U_2[idx, c] != 0]
        v_list = [(idx, V_2[idx, c]) for idx in range(4) if V_2[idx, c] != 0]
        w_list = [(idx, W_2[idx, c]) for idx in range(4) if W_2[idx, c] != 0]
        U_2_sparse.append(u_list)
        V_2_sparse.append(v_list)
        W_2_sparse.append(w_list)

    U_list = []
    V_list = []
    W_list = []

    def build_4x4_block(B_col_offset, C_col_offset):
        """
        Builds the 49 terms for one 4x4x4 sub-problem.
        A is fixed at rows 0-3, cols 0-3.
        B is rows 0-3, cols B_col_offset to B_col_offset+3.
        C is rows 0-3, cols C_col_offset to C_col_offset+3.
        """
        for k_p in range(7):
            for m_s in range(7):
                U_row = np.zeros(nm)
                V_row = np.zeros(mp)
                W_row = np.zeros(pn)
                
                # A part
                # A_specs logic from original code:
                # We iterate over the 4 elements of the 2x2 block in A
                # But we can just use the sparse lists and map indices
                
                # The mapping from local 2x2 index (r, c) to global 4x4 index (i, j)
                # for a block at (br, bc) is i = br*2 + r, j = bc*2 + c
                
                # However, the original code used specific A_specs which combined terms.
                # Let's stick to the explicit expansion to be safe and match the known working logic.
                # Original A_specs:
                # P0 (k_p=0): A00+A11 -> (1,0,0) and (1,1,1)
                # P1 (k_p=1): A10+A11 -> (1,1,0) and (1,1,1)
                # P2 (k_p=2): A00 -> (1,0,0)
                # P3 (k_p=3): A11 -> (1,1,1)
                # P4 (k_p=4): A00+A01 -> (1,0,0) and (1,0,1)
                # P5 (k_p=5): A10-A00 -> (1,1,0) and (-1,0,0)
                # P6 (k_p=6): A01-A11 -> (1,0,1) and (-1,1,1)
                
                # This corresponds to:
                # A_indices[k_p] = list of (coeff, br, bc)
                A_specs = [
                    [(1, 0, 0), (1, 1, 1)],
                    [(1, 1, 0), (1, 1, 1)],
                    [(1, 0, 0)],
                    [(1, 1, 1)],
                    [(1, 0, 0), (1, 0, 1)],
                    [(1, 1, 0), (-1, 0, 0)],
                    [(1, 0, 1), (-1, 1, 1)]
                ]
                B_specs = [
                    [(1, 0, 0), (1, 1, 1)],
                    [(1, 0, 0)],
                    [(1, 0, 1), (-1, 1, 1)],
                    [(1, 1, 0), (-1, 0, 0)],
                    [(1, 1, 1)],
                    [(1, 0, 0), (1, 0, 1)],
                    [(1, 1, 0), (1, 1, 1)]
                ]
                C_specs = [
                    {(0, 0): 1, (1, 1): 1},
                    {(1, 0): 1, (1, 1): -1},
                    {(0, 1): 1, (1, 1): 1},
                    {(0, 0): 1, (1, 0): 1},
                    {(0, 0): -1, (0, 1): 1},
                    {(1, 1): 1},
                    {(0, 0): 1}
                ]

                # Process A
                for coeff, br, bc in A_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            # Global A index: (br*2+r, bc*2+c)
                            gi = br * 2 + r
                            gj = bc * 2 + c
                            # We need to sum contributions from U_2[r*2+c, m_s]
                            # Wait, U_2 indices are row-major for 2x2: 0:00, 1:01, 2:10, 3:11
                            # So local index in U_2 is r*2 + c
                            u_val = U_2[r * 2 + c, m_s]
                            if u_val != 0:
                                U_row[gi * 4 + gj] += coeff * u_val

                # Process B
                for coeff, br, bc in B_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            gj = br * 2 + r
                            gk = B_col_offset + bc * 2 + c
                            v_val = V_2[r * 2 + c, m_s]
                            if v_val != 0:
                                V_row[gj * 8 + gk] += coeff * v_val

                # Process C
                for (cbr, cbc), coeff_c in C_specs[k_p].items():
                    for r in range(2):
                        for c in range(2):
                            gi = cbr * 2 + r
                            gk = C_col_offset + cbc * 2 + c
                            w_val = W_2[r * 2 + c, m_s]
                            if w_val != 0:
                                W_row[gk * 4 + gi] += coeff_c * w_val

                if np.any(U_row) and np.any(V_row) and np.any(W_row):
                    U_list.append(U_row)
                    V_list.append(V_row)
                    W_list.append(W_row)

    # Build left half (B cols 0-3, C cols 0-3)
    build_4x4_block(0, 0)
    
    # Check time
    if time.time() > deadline:
         # Fallback to schoolbook if construction is unexpectedly slow (unlikely)
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
         return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Build right half (B cols 4-7, C cols 4-7)
    build_4x4_block(4, 4)

    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    # Fast verification using einsum
    # U: (16, R) -> (4, 4, R)
    # V: (32, R) -> (4, 8, R)
    # W: (32, R) -> (8, 4, R)
    U_44 = U.reshape(4, 4, rank)
    V_48 = V.reshape(4, 8, rank)
    W_84 = W.reshape(8, 4, rank)
    
    # M[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: W is indexed by k, i. In W_84, indices are (k, i).
    M = np.einsum('ijr,jkr,kir->ijk', U_44, V_48, W_84)
    
    if not np.allclose(M, 1.0, atol=1e-10):
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
         return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
