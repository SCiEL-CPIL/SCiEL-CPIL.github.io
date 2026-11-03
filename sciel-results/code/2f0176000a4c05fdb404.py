import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Construct 49-rank decomposition for 4x4x4
    U_44, V_44, W_44 = construct_4x4x4_strassen()
    
    # Construct 98-rank decomposition for 4x4x8
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First half
    U[:, 0:49] = U_44
    V[0:16, 0:49] = V_44
    W[0:16, 0:49] = W_44
    
    # Second half
    U[:, 49:98] = U_44
    V[16:32, 49:98] = V_44
    W[16:32, 49:98] = W_44
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_4x4x4_strassen():
    # Returns U, V, W for 4x4x4 rank 49
    # U: (16, 49), V: (16, 49), W: (16, 49)
    
    # We use the recursive Strassen structure.
    # M1..M7 are 2x2x2 multiplications.
    # Each 2x2x2 multiplication has 7 factors.
    # Total 49 factors.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Helper to get 2x2x2 factors
    def get_2x2_factors():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        u[0,0]=1; u[3,0]=1
        v[0,0]=1; v[3,0]=1
        w[0,0]=1; w[1,0]=-1; w[2,0]=-1; w[3,0]=1
        u[2,1]=1; u[3,1]=1
        v[0,1]=1
        w[2,1]=1; w[3,1]=-1
        u[0,2]=1
        v[1,2]=1; v[3,2]=-1
        w[1,2]=1; w[3,2]=-1
        u[3,3]=1
        v[2,3]=1; v[0,3]=-1
        w[0,3]=-1; w[2,3]=1
        u[0,4]=1; u[1,4]=1
        v[3,4]=1
        w[1,4]=1; w[3,4]=1
        u[2,5]=1; u[0,5]=-1
        v[0,5]=1; v[1,5]=1
        w[0,5]=1
        u[1,6]=1; u[3,6]=-1
        v[2,6]=1; v[3,6]=1
        w[0,6]=1
        return u, v, w

    u2, v2, w2 = get_2x2_factors()
    
    # Mapping of blocks to indices in 4x4
    # A:
    # 11: 0,1,2,3
    # 12: 0,1,4,5 -> No, row major:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # Row 3: 12,13,14,15
    # A11: 0,1,2,3 (rows 0,1 cols 0,1) -> 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # B:
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # C:
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    # Indices for 2x2 blocks in 4x4:
    # Block 11: [0,1,4,5]
    # Block 12: [2,3,6,7]
    # Block 21: [8,9,12,13]
    # Block 22: [10,11,14,15]
    
    def get_block_indices(block_id):
        if block_id == 11: return [0,1,4,5]
        if block_id == 12: return [2,3,6,7]
        if block_id == 21: return [8,9,12,13]
        if block_id == 22: return [10,11,14,15]

    # M1: (A11+A22)(B11+B22)
    # For each of the 7 factors in u2, v2, w2:
    # U: add u2 to A11 indices and A22 indices
    # V: add v2 to B11 indices and B22 indices
    # W: add w2 to C11, C12, C21, C22 with appropriate signs
    
    for m in range(7):
        base_col = m * 7
        # U
        for i in range(4):
            u_val = u2[i, m]
            if u_val != 0:
                # Add to A11 and A22
                for idx in get_block_indices(11):
                    U[idx, base_col + i] += u_val
                for idx in get_block_indices(22):
                    U[idx, base_col + i] += u_val
        # V
        for i in range(4):
            v_val = v2[i, m]
            if v_val != 0:
                for idx in get_block_indices(11):
                    V[idx, base_col + i] += v_val
                for idx in get_block_indices(22):
                    V[idx, base_col + i] += v_val
        # W
        for i in range(4):
            w_val = w2[i, m]
            if w_val != 0:
                # C11 gets +, C12 gets -, C21 gets -, C22 gets +
                for idx in get_block_indices(11):
                    W[idx, base_col + i] += w_val
                for idx in get_block_indices(12):
                    W[idx, base_col + i] -= w_val
                for idx in get_block_indices(21):
                    W[idx, base_col + i] -= w_val
                for idx in get_block_indices(22):
                    W[idx, base_col + i] += w_val
    
    # M2: (A21+A22)B11
    for m in range(7):
        base_col = 7 + m * 7
        for i in range(4):
            u_val = u2[i, m]
            if u_val != 0:
                for idx in get_block_indices(21):
                    U[idx, base_col + i] += u_val
                for idx in get_block_indices(22):
                    U[idx, base_col + i] += u_val
        for i in range(4):
            v_val = v2[i, m]
            if v_val != 0:
                for idx in get_block_indices(11):
                    V[idx, base_col + i] += v_val
        for i in range(4):
            w_val = w2[i, m]
            if w_val != 0:
                for idx in get_block_indices(21):
                    W[idx, base_col + i] += w_val
                for idx in get_block_indices(22):
                    W[idx, base_col + i] -= w_val

    # M3: A11(B12-B22)
    for m in range(7):
        base_col = 14 + m * 7
        for i in range(4):
            u_val = u2[i, m]
            if u_val != 0:
                for idx in get_block_indices(11):
                    U[idx, base_col + i] += u_val
        for i in range(4):
            v_val = v2[i, m]
            if v_val != 0:
                for idx in get_block_indices(12):
                    V[idx, base_col + i] += v_val
                for idx in get_block_indices(22):
                    V[idx, base_col + i] -= v_val
        for i in range(4):
            w_val = w2[i, m]
            if w_val != 0:
                for idx in get_block_indices(12):
                    W[idx, base_col + i] += w_val
                for idx in get_block_indices(22):
                    W[idx, base_col + i] -= w_val

    # M4: A22(B21-B11)
    for m in range(7):
        base_col = 21 + m * 7
        for i in range(4):
            u_val = u2[i, m]
            if u_val != 0:
                for idx in get_block_indices(22):
                    U[idx, base_col + i] += u_val
        for i in range(4):
            v_val = v2[i, m]
            if v_val != 0:
                for idx in get_block_indices(21):
                    V[idx, base_col + i] += v_val
                for idx in get_block_indices(11):
                    V[idx, base_col + i] -= v_val
        for i in range(4):
            w_val = w2[i, m]
            if w_val != 0:
                for idx in get_block_indices(11):
                    W[idx, base_col + i] -= w_val
                for idx in get_block_indices(21):
                    W[idx, base_col + i] += w_val

    # M5: (A11+A12)B22
    for m in range(7):
        base_col = 28 + m * 7
        for i in range(4):
            u_val = u2[i, m]
            if u_val != 0:
                for idx in get_block_indices(11):
                    U[idx, base_col + i] += u_val
                for idx in get_block_indices(12):
                    U[idx, base_col + i] += u_val
        for i in range(4):
            v_val = v2[i, m]
            if v_val != 0:
                for idx in get_block_indices(22):
                    V[idx, base_col + i] += v_val
        for i in range(4):
            w_val = w2[i, m]
            if w_val != 0:
                for idx in get_block_indices(12):
                    W[idx, base_col + i] += w_val
                for idx in get_block_indices(22):
                    W[idx, base_col + i] += w_val

    # M6: (A21-A11)(B11+B12)
    for m in range(7):
        base_col = 35 + m * 7
        for i in range(4):
            u_val = u2[i, m]
            if u_val != 0:
                for idx in get_block_indices(21):
                    U[idx, base_col + i] += u_val
                for idx in get_block_indices(11):
                    U[idx, base_col + i] -= u_val
        for i in range(4):
            v_val = v2[i, m]
            if v_val != 0:
                for idx in get_block_indices(11):
                    V[idx, base_col + i] += v_val
                for idx in get_block_indices(12):
                    V[idx, base_col + i] += v_val
        for i in range(4):
            w_val = w2[i, m]
            if w_val != 0:
                for idx in get_block_indices(11):
                    W[idx, base_col + i] += w_val

    # M7: (A12-A22)(B21+B22)
    for m in range(7):
        base_col = 42 + m * 7
        for i in range(4):
            u_val = u2[i, m]
            if u_val != 0:
                for idx in get_block_indices(12):
                    U[idx, base_col + i] += u_val
                for idx in get_block_indices(22):
                    U[idx, base_col + i] -= u_val
        for i in range(4):
            v_val = v2[i, m]
            if v_val != 0:
                for idx in get_block_indices(21):
                    V[idx, base_col + i] += v_val
                for idx in get_block_indices(22):
                    V[idx, base_col + i] += v_val
        for i in range(4):
            w_val = w2[i, m]
            if w_val != 0:
                for idx in get_block_indices(11):
                    W[idx, base_col + i] += w_val

    return U, V, W

# EVOLVE-BLOCK-END
