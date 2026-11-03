# ... (imports)

def _construct_rank_40():
    # Try to find rank 10 for 2x2x3
    U_223, V_223, W_223 = _find_rank_10_2x2x3(max_time=10.0)
    if U_223 is None:
        return None
        
    n, m, p = 2, 4, 6
    rank = 40
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Map the 4 blocks
    # Block 0: A1 (0:2, 0:2) * B1a (0:2, 0:3) -> C0 (0:2, 0:3)
    # Block 1: A1 (0:2, 0:2) * B1b (0:2, 3:6) -> C1 (0:2, 3:6)
    # Block 2: A2 (0:2, 2:4) * B2a (2:4, 0:3) -> C2 (0:2, 0:3) ... Wait.
    
    # A is 2x4.
    # A1 = A[:, 0:2] (2x2)
    # A2 = A[:, 2:4] (2x2)
    
    # B is 4x6.
    # B1 = B[0:2, :] (2x6)
    # B2 = B[2:4, :] (2x6)
    
    # C is 2x6.
    # C = A1 B1 + A2 B2
    
    # B1 = [B1a B1b] where B1a is 2x3, B1b is 2x3.
    # B2 = [B2a B2b] where B2a is 2x3, B2b is 2x3.
    
    # Block 0: A1 * B1a. A1 is 2x2. B1a is 2x3.
    # U indices for A1: 0,1,2,3 (flattened 2x2)
    # V indices for B1a: 0,1,2,3,4,5 (flattened 2x3)
    # W indices for C0: 0,1,2,3,4,5 (flattened 2x3)
    
    # We need to map the local indices to global indices.
    
    def map_block(block_id, U_local, V_local, W_local):
        # block_id 0: A1, B1a
        # block_id 1: A1, B1b
        # block_id 2: A2, B2a
        # block_id 3: A2, B2b
        
        offset = block_id * 10
        
        # U mapping
        if block_id == 0 or block_id == 1:
            # A1 is A[:, 0:2]
            # Global A indices: i*4 + j. i in 0..1, j in 0..1.
            # Local A indices: i*2 + j.
            for i in range(2):
                for j in range(2):
                    u_global = i * 4 + j
                    u_local = i * 2 + j
                    U[u_global, offset:offset+10] += U_local[u_local, :]
        else:
            # A2 is A[:, 2:4]
            # Global A indices: i*4 + j. i in 0..1, j in 2..3.
            # Local A indices: i*2 + (j-2).
            for i in range(2):
                for j in range(2):
                    u_global = i * 4 + (j + 2)
                    u_local = i * 2 + j
                    U[u_global, offset:offset+10] += U_local[u_local, :]
                    
        # V mapping
        if block_id == 0 or block_id == 2:
            # B1a or B2a (cols 0:3)
            # Global B indices: j*6 + k. j in 0..1 (for B1) or 2..3 (for B2). k in 0..2.
            # Local B indices: j*3 + k.
            if block_id == 0:
                j_offset = 0
            else:
                j_offset = 2
            for j in range(2):
                for k in range(3):
                    v_global = (j + j_offset) * 6 + k
                    v_local = j * 3 + k
                    V[v_global, offset:offset+10] += V_local[v_local, :]
        else:
            # B1b or B2b (cols 3:6)
            if block_id == 1:
                j_offset = 0
            else:
                j_offset = 2
            for j in range(2):
                for k in range(3):
                    v_global = (j + j_offset) * 6 + (k + 3)
                    v_local = j * 3 + k
                    V[v_global, offset:offset+10] += V_local[v_local, :]
                    
        # W mapping
        # C is 2x6.
        # C0 is C[:, 0:3] (cols 0:3)
        # C1 is C[:, 3:6] (cols 3:6)
        # Global C indices: k*2 + i. k in 0..5, i in 0..1.
        # Local C indices: k*2 + i. k in 0..2, i in 0..1.
        
        if block_id == 0 or block_id == 2:
            # Cols 0:3
            for k in range(3):
                for i in range(2):
                    w_global = k * 2 + i
                    w_local = k * 2 + i
                    W[w_global, offset:offset+10] += W_local[w_local, :]
        else:
            # Cols 3:6
            for k in range(3):
                for i in range(2):
                    w_global = (k + 3) * 2 + i
                    w_local = k * 2 + i
                    W[w_global, offset:offset+10] += W_local[w_local, :]
                    
    map_block(0, U_223, V_223, W_223)
    map_block(1, U_223, V_223, W_223)
    map_block(2, U_223, V_223, W_223)
    map_block(3, U_223, V_223, W_223)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
