"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    # Partitions
    # n=3 -> [1, 2]
    # m=5 -> [2, 2, 1]
    # p=5 -> [2, 2, 1]
    
    I_parts = [1, 2]
    J_parts = [2, 2, 1]
    K_parts = [2, 2, 1]
    
    # Calculate total rank
    # We will build the matrices dynamically to determine rank
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_rows, v_rows, w_rows, u_vals, v_vals, w_vals):
        """
        u_rows: list of global (i, j) indices
        v_rows: list of global (j, k) indices
        w_rows: list of global (k, i) indices
        u_vals: values for U
        v_vals: values for V
        w_vals: values for W
        """
        # This is a bit slow if done one by one. 
        # Better to construct full vectors for the block and append.
        pass

    # More efficient: Construct local matrices for each block, then map to global.
    
    # Pre-allocate global matrices? We don't know rank yet.
    # Let's collect local blocks first.
    
    blocks = []
    
    # Offsets
    I_offsets = [0]
    current = 0
    for size in I_parts:
        current += size
        I_offsets.append(current)
        
    J_offsets = [0]
    current = 0
    for size in J_parts:
        current += size
        J_offsets.append(current)
        
    K_offsets = [0]
    current = 0
    for size in K_parts:
        current += size
        K_offsets.append(current)

    # Iterate through blocks
    for idx_i, ni in enumerate(I_parts):
        for idx_j, mj in enumerate(J_parts):
            for idx_k, pk in enumerate(K_parts):
                # Local block size ni x mj x pk
                # Determine decomposition
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen Rank 7
                    # Local indices i in 0..1, j in 0..1, k in 0..1
                    # U_local shape (2*2, 7)
                    # V_local shape (2*2, 7)
                    # W_local shape (2*2, 7)
                    
                    U_l = np.zeros((4, 7))
                    V_l = np.zeros((4, 7))
                    W_l = np.zeros((4, 7))
                    
                    # M1 = (A00 + A11)(B00 + B11)
                    U_l[0, 0] = 1 # A00
                    U_l[3, 0] = 1 # A11
                    V_l[0, 0] = 1 # B00
                    V_l[3, 0] = 1 # B11
                    W_l[0, 0] = 1 # C00
                    W_l[3, 0] = 1 # C11
                    
                    # M2 = (A10 + A11) B00
                    U_l[2, 1] = 1 # A10
                    U_l[3, 1] = 1 # A11
                    V_l[0, 1] = 1 # B00
                    W_l[2, 1] = 1 # C10
                    
                    # M3 = A00 (B10 + B11)
                    U_l[0, 2] = 1 # A00
                    V_l[2, 2] = 1 # B10
                    V_l[3, 2] = 1 # B11
                    W_l[1, 2] = 1 # C01
                    W_l[3, 2] = 1 # C11
                    
                    # M4 = A11 (B01 - B11)
                    U_l[3, 3] = 1 # A11
                    V_l[1, 3] = 1 # B01
                    V_l[3, 3] = -1 # B11
                    W_l[0, 3] = 1 # C00
                    W_l[2, 3] = 1 # C10
                    
                    # M5 = (A00 + A01) B11
                    U_l[0, 4] = 1 # A00
                    U_l[1, 4] = 1 # A01
                    V_l[3, 4] = 1 # B11
                    W_l[0, 4] = -1 # C00
                    W_l[1, 4] = 1 # C01
                    
                    # M6 = (A10 - A00)(B00 + B01)
                    U_l[2, 5] = 1 # A10
                    U_l[0, 5] = -1 # A00
                    V_l[0, 5] = 1 # B00
                    V_l[1, 5] = 1 # B01
                    W_l[3, 5] = 1 # C11
                    
                    # M7 = (A01 - A11)(B10 + B11)
                    U_l[1, 6] = 1 # A01
                    U_l[3, 6] = -1 # A11
                    V_l[2, 6] = 1 # B10
                    V_l[3, 6] = 1 # B11
                    W_l[0, 6] = 1 # C00
                    W_l[1, 6] = 1 # C01
                    
                    blocks.append((idx_i, idx_j, idx_k, U_l, V_l, W_l))
                
                else:
                    # Schoolbook
                    rank = ni * mj * pk
                    U_l = np.zeros((ni * mj, rank))
                    V_l = np.zeros((mj * pk, rank))
                    W_l = np.zeros((pk * ni, rank))
                    
                    col = 0
                    for i in range(ni):
                        for j in range(mj):
                            for k in range(pk):
                                U_l[i * mj + j, col] = 1
                                V_l[j * pk + k, col] = 1
                                W_l[k * ni + i, col] = 1
                                col += 1
                    blocks.append((idx_i, idx_j, idx_k, U_l, V_l, W_l))

    # Now assemble global matrices
    total_rank = sum(block[3].shape[1] for block in blocks)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    for idx_i, idx_j, idx_k, U_l, V_l, W_l in blocks:
        rank_b = U_l.shape[1]
        
        ni = I_parts[idx_i]
        mj = J_parts[idx_j]
        pk = K_parts[idx_k]
        
        i_off = I_offsets[idx_i]
        j_off = J_offsets[idx_j]
        k_off = K_offsets[idx_k]
        
        # Map local indices to global rows
        # U row: i * m + j
        # V row: j * p + k
        # W row: k * n + i
        
        for r in range(rank_b):
            # U
            for i_l in range(ni):
                for j_l in range(mj):
                    val = U_l[i_l * mj + j_l, r]
                    if val != 0:
                        i_g = i_off + i_l
                        j_g = j_off + j_l
                        U[i_g * m + j_g, current_rank + r] = val
                        
            # V
            for j_l in range(mj):
                for k_l in range(pk):
                    val = V_l[j_l * pk + k_l, r]
                    if val != 0:
                        j_g = j_off + j_l
                        k_g = k_off + k_l
                        V[j_g * p + k_g, current_rank + r] = val
                        
            # W
            for k_l in range(pk):
                for i_l in range(ni):
                    val = W_l[k_l * ni + i_l, r]
                    if val != 0:
                        k_g = k_off + k_l
                        i_g = i_off + i_l
                        W[k_g * n + i_g, current_rank + r] = val
                        
        current_rank += rank_b
        
    if time.time() > deadline:
        # Fallback to schoolbook if we are slow (unlikely)
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
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
