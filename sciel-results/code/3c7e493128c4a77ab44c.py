"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # We will construct a decomposition with rank 64 using block schoolbook.
    # Structure:
    # B (4x8) -> B_L (4x4), B_R (4x4)
    # For each B_part (4x4):
    #   A (3x4) -> A_top (2x4), A_bot (1x4)
    #   A_top (2x4) * B_part (4x4) -> 
    #      B_part -> B_top (2x4), B_bot (2x4)
    #      A_top * B_top (2x4) -> 
    #         B_top -> B_t1 (2x2), B_t2 (2x2)
    #         A_top * B_t1 (2x2) -> Rank 4
    #         A_top * B_t2 (2x2) -> Rank 4
    #      A_top * B_bot (2x4) -> Rank 8
    #   A_bot (1x4) * B_part (4x4) -> Rank 16
    # Total per B_part: 16 (top) + 16 (bot) = 32
    # Total Rank: 64

    rank = 64
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx, col] = val
        V[v_idx, col] = val
        W[w_idx, col] = val
        col += 1

    # Helper to add schoolbook terms for a sub-problem
    # n_sub, m_sub, p_sub
    # u_offset, v_offset, w_offset
    # The indices in the global U, V, W correspond to the local indices of the sub-problem.
    # Local U index: i * m_sub + j
    # Local V index: j * p_sub + k
    # Local W index: k * n_sub + i
    
    def add_schoolbook(n_sub, m_sub, p_sub, u_off, v_off, w_off):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u_idx = u_off + i * m_sub + j
                    v_idx = v_off + j * p_sub + k
                    w_idx = w_off + k * n_sub + i
                    add_term(u_idx, v_idx, w_idx, 1.0)

    # Global Indices:
    # U: i (0..2) * 4 + j (0..3)
    # V: j (0..3) * 8 + k (0..7)
    # W: k (0..7) * 3 + i (0..2)

    # Split B into L (k=0..3) and R (k=4..7)
    # B_L: j (0..3), k (0..3). V offset 0.
    # B_R: j (0..3), k (4..7). V offset 4*8 = 32? No.
    # V index is j * p + k.
    # For B_L, k is 0..3. V indices: j*8 + 0..3.
    # For B_R, k is 4..7. V indices: j*8 + 4..7.
    
    # Process B_L
    # A_top: i=0,1. j=0..3.
    # A_bot: i=2. j=0..3.
    
    # --- A_top * B_L ---
    # A_top: n=2, m=4. U offset 0.
    # B_L: m=4, p=4. V offset 0.
    # Result C_top_L: n=2, p=4. W offset 0.
    
    # Split B_L into B_L_top (j=0,1) and B_L_bot (j=2,3)?
    # No, B is 4x4. Rows are j.
    # A_top is 2x4.
    # C = A_top * B_L.
    # C[i, k] = sum_j A_top[i, j] * B_L[j, k].
    # j ranges 0..3.
    # Split j into 0,1 and 2,3.
    # B_L_top: rows 0,1. V indices: 0*8+0..3, 1*8+0..3.
    # B_L_bot: rows 2,3. V indices: 2*8+0..3, 3*8+0..3.
    
    # A_top * B_L_top (2x4 * 2x4 -> 2x4)
    # n_sub=2, m_sub=2, p_sub=4.
    # U offset 0.
    # V offset 0.
    # W offset 0.
    # But B_L_top is 2x4.
    # We can further split p (k) into 0,1 and 2,3.
    # B_L_top_left: k=0,1.
    # B_L_top_right: k=2,3.
    
    # A_top * B_L_top_left (2x2 * 2x2 -> 2x2)
    # n=2, m=2, p=2.
    # U offset 0.
    # V offset 0.
    # W offset 0.
    add_schoolbook(2, 2, 2, 0, 0, 0)
    
    # A_top * B_L_top_right (2x2 * 2x2 -> 2x2)
    # n=2, m=2, p=2.
    # U offset 0.
    # V offset: j is 0,1. k is 2,3.
    # V index = j*8 + k.
    # j=0, k=2 -> 2. j=0, k=3 -> 3.
    # j=1, k=2 -> 10. j=1, k=3 -> 11.
    # The V offset for the sub-problem is not a simple offset because of the stride.
    # However, add_schoolbook assumes contiguous indices.
    # We must map the indices correctly.
    
    # Let's refine add_schoolbook to take explicit index mappings or handle strides.
    # It's easier to just call add_term directly for the small blocks.
    
    # Reset col? No, we are appending.
    # col is currently 4 (from the first add_schoolbook).
    
    # A_top * B_L_top_right
    # i in 0..1, j in 0..1, k in 2..3
    for i in range(2):
        for j in range(2):
            for k in range(2, 4):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx, 1.0)
                
    # A_top * B_L_bot_left (j=2,3; k=0,1)
    for i in range(2):
        for j in range(2, 4):
            for k in range(2):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx, 1.0)
                
    # A_top * B_L_bot_right (j=2,3; k=2,3)
    for i in range(2):
        for j in range(2, 4):
            for k in range(2, 4):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx, 1.0)

    # --- A_bot * B_L ---
    # i=2. j=0..3. k=0..3.
    for i in range(2, 3):
        for j in range(4):
            for k in range(4):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx, 1.0)

    # --- B_R (k=4..7) ---
    # A_top * B_R
    # i=0,1. j=0..3. k=4..7.
    # Split j into 0,1 and 2,3.
    # Split k into 4,5 and 6,7.
    
    # A_top * B_R_top_left (j=0,1; k=4,5)
    for i in range(2):
        for j in range(2):
            for k in range(4, 6):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx, 1.0)
                
    # A_top * B_R_top_right (j=0,1; k=6,7)
    for i in range(2):
        for j in range(2):
            for k in range(6, 8):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx, 1.0)
                
    # A_top * B_R_bot_left (j=2,3; k=4,5)
    for i in range(2):
        for j in range(2, 4):
            for k in range(4, 6):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx, 1.0)
                
    # A_top * B_R_bot_right (j=2,3; k=6,7)
    for i in range(2):
        for j in range(2, 4):
            for k in range(6, 8):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx, 1.0)

    # A_bot * B_R
    for i in range(2, 3):
        for j in range(4):
            for k in range(4, 8):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx, 1.0)

    # Check rank
    if col != rank:
        # Adjust rank to actual used
        rank = col
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]
        
    # Verify
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    recon = np.tensordot(U, V, axes=([1], [1]))
    recon = np.tensordot(recon, W, axes=([2], [1]))
    
    if not np.allclose(recon, T):
        # Fallback to schoolbook
        U = np.zeros((nm, nm * mp))
        V = np.zeros((mp, nm * mp))
        W = np.zeros((pn, nm * mp))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = col
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
