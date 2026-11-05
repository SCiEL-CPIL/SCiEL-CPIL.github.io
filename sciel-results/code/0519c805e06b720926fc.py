"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 88
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Helper to add a schoolbook 1xm x mxp term
    def add_schoolbook_1mp(i_offset, j_range, k_range, u_base, v_base, w_base):
        # u_base, v_base, w_base are not used directly if we use global indices
        # We pass the global indices directly
        pass

    # Part 1: 1 x 4 x 8 (Rank 32)
    # A row 0 (i=0), B all (j=0..3, k=0..7), C row 0 (i=0, k=0..7)
    for j in range(4):
        for k in range(8):
            u_idx = 0 * m + j
            v_idx = j * p + k
            w_idx = k * n + 0
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Part 2: 2 x 4 x 8 (Rank 56)
    # A rows 1,2 (i=1,2), B all (j=0..3, k=0..7), C rows 1,2 (i=1,2, k=0..7)
    # Split into two 2 x 4 x 4
    for k_block in range(2):
        # k_block 0: k=0,1; k_block 1: k=2,3
        k_start = k_block * 4
        # Wait, p=8. Split into 4 and 4.
        # k_block 0: k=0..3
        # k_block 1: k=4..7
        
        # Inside 2 x 4 x 4:
        # Split into two 2 x 4 x 2
        for k_sub in range(2):
            # k_sub 0: k=k_start..k_start+1
            # k_sub 1: k=k_start+2..k_start+3
            k_curr_start = k_start + k_sub * 2
            
            # 2 x 4 x 2 decomposition
            # A: rows 1,2 (i=1,2), cols 0..3 (j=0..3)
            # B: rows 0..3 (j=0..3), cols k_curr_start, k_curr_start+1
            # C: rows 1,2 (i=1,2), cols k_curr_start, k_curr_start+1
            
            # Split A into A_L (j=0,1) and A_R (j=2,3)
            # Split B into B_T (j=0,1) and B_B (j=2,3)
            
            # Term 1: A_L B_T (2x2 x 2x2 -> 2x2)
            # A_L: i=1,2; j=0,1
            # B_T: j=0,1; k=k_curr_start, k_curr_start+1
            # C_part: i=1,2; k=k_curr_start, k_curr_start+1
            
            # Strassen for A_L * B_T
            # Local A indices: (i_local, j_local) -> (0,0), (0,1), (1,0), (1,1)
            # Global A indices:
            # (0,0) -> i=1, j=0 -> 1*4+0 = 4
            # (0,1) -> i=1, j=1 -> 1*4+1 = 5
            # (1,0) -> i=2, j=0 -> 2*4+0 = 8
            # (1,1) -> i=2, j=1 -> 2*4+1 = 9
            
            # Local B indices: (j_local, k_local) -> (0,0), (0,1), (1,0), (1,1)
            # Global B indices:
            # (0,0) -> j=0, k=k_curr_start -> 0*8 + k_curr_start
            # (0,1) -> j=0, k=k_curr_start+1 -> 0*8 + k_curr_start+1
            # (1,0) -> j=1, k=k_curr_start -> 1*8 + k_curr_start
            # (1,1) -> j=1, k=k_curr_start+1 -> 1*8 + k_curr_start+1
            
            # Local C indices: (i_local, k_local)
            # Global C indices:
            # (0,0) -> i=1, k=k_curr_start -> k_curr_start*3 + 1
            # (0,1) -> i=1, k=k_curr_start+1 -> (k_curr_start+1)*3 + 1
            # (1,0) -> i=2, k=k_curr_start -> k_curr_start*3 + 2
            # (1,1) -> i=2, k=k_curr_start+1 -> (k_curr_start+1)*3 + 2
            
            # Define mappings
            def get_u_idx(i_l, j_l):
                return (1 + i_l) * m + (0 + j_l) # i=1..2, j=0..1 for A_L
            
            def get_v_idx(j_l, k_l):
                return (0 + j_l) * p + (k_curr_start + k_l) # j=0..1, k=k_curr_start..+1 for B_T
            
            def get_w_idx(i_l, k_l):
                return (k_curr_start + k_l) * n + (1 + i_l) # k=k_curr_start..+1, i=1..2
            
            # Strassen Terms
            # M1: (A00+A11)(B00+B11)
            U[get_u_idx(0,0), col] = 1
            U[get_u_idx(1,1), col] = 1
            V[get_v_idx(0,0), col] = 1
            V[get_v_idx(1,1), col] = 1
            W[get_w_idx(0,0), col] = 1 # Z00
            W[get_w_idx(1,1), col] = 1 # Z11
            col += 1
            
            # M2: (A10+A11)B00
            U[get_u_idx(1,0), col] = 1
            U[get_u_idx(1,1), col] = 1
            V[get_v_idx(0,0), col] = 1
            W[get_w_idx(1,0), col] = 1 # Z10
            W[get_w_idx(1,1), col] = -1 # Z11
            col += 1
            
            # M3: A00(B01-B11)
            U[get_u_idx(0,0), col] = 1
            V[get_v_idx(0,1), col] = 1
            V[get_v_idx(1,1), col] = -1
            W[get_w_idx(0,1), col] = 1 # Z01
            W[get_w_idx(1,1), col] = 1 # Z11
            col += 1
            
            # M4: A11(B10-B00)
            U[get_u_idx(1,1), col] = 1
            V[get_v_idx(1,0), col] = 1
            V[get_v_idx(0,0), col] = -1
            W[get_w_idx(0,0), col] = 1 # Z00
            W[get_w_idx(1,0), col] = 1 # Z10
            col += 1
            
            # M5: (A00+A01)B11
            U[get_u_idx(0,0), col] = 1
            U[get_u_idx(0,1), col] = 1
            V[get_v_idx(1,1), col] = 1
            W[get_w_idx(0,0), col] = -1 # Z00
            W[get_w_idx(0,1), col] = 1 # Z01
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[get_u_idx(1,0), col] = 1
            U[get_u_idx(0,0), col] = -1
            V[get_v_idx(0,0), col] = 1
            V[get_v_idx(0,1), col] = 1
            W[get_w_idx(1,1), col] = 1 # Z11
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[get_u_idx(0,1), col] = 1
            U[get_u_idx(1,1), col] = -1
            V[get_v_idx(1,0), col] = 1
            V[get_v_idx(1,1), col] = 1
            W[get_w_idx(0,0), col] = 1 # Z00
            col += 1
            
            # Term 2: A_R B_B (2x2 x 2x2 -> 2x2)
            # A_R: i=1,2; j=2,3
            # B_B: j=2,3; k=k_curr_start, k_curr_start+1
            # C_part: i=1,2; k=k_curr_start, k_curr_start+1
            
            def get_u_idx_r(i_l, j_l):
                return (1 + i_l) * m + (2 + j_l) # i=1..2, j=2..3 for A_R
            
            def get_v_idx_b(j_l, k_l):
                return (2 + j_l) * p + (k_curr_start + k_l) # j=2..3, k=k_curr_start..+1 for B_B
            
            # W indices are the same as before, because it's the same C block
            # def get_w_idx(i_l, k_l): ... same ...
            
            # M1: (A00+A11)(B00+B11)
            U[get_u_idx_r(0,0), col] = 1
            U[get_u_idx_r(1,1), col] = 1
            V[get_v_idx_b(0,0), col] = 1
            V[get_v_idx_b(1,1), col] = 1
            W[get_w_idx(0,0), col] = 1
            W[get_w_idx(1,1), col] = 1
            col += 1
            
            # M2: (A10+A11)B00
            U[get_u_idx_r(1,0), col] = 1
            U[get_u_idx_r(1,1), col] = 1
            V[get_v_idx_b(0,0), col] = 1
            W[get_w_idx(1,0), col] = 1
            W[get_w_idx(1,1), col] = -1
            col += 1
            
            # M3: A00(B01-B11)
            U[get_u_idx_r(0,0), col] = 1
            V[get_v_idx_b(0,1), col] = 1
            V[get_v_idx_b(1,1), col] = -1
            W[get_w_idx(0,1), col] = 1
            W[get_w_idx(1,1), col] = 1
            col += 1
            
            # M4: A11(B10-B00)
            U[get_u_idx_r(1,1), col] = 1
            V[get_v_idx_b(1,0), col] = 1
            V[get_v_idx_b(0,0), col] = -1
            W[get_w_idx(0,0), col] = 1
            W[get_w_idx(1,0), col] = 1
            col += 1
            
            # M5: (A00+A01)B11
            U[get_u_idx_r(0,0), col] = 1
            U[get_u_idx_r(0,1), col] = 1
            V[get_v_idx_b(1,1), col] = 1
            W[get_w_idx(0,0), col] = -1
            W[get_w_idx(0,1), col] = 1
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[get_u_idx_r(1,0), col] = 1
            U[get_u_idx_r(0,0), col] = -1
            V[get_v_idx_b(0,0), col] = 1
            V[get_v_idx_b(0,1), col] = 1
            W[get_w_idx(1,1), col] = 1
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[get_u_idx_r(0,1), col] = 1
            U[get_u_idx_r(1,1), col] = -1
            V[get_v_idx_b(1,0), col] = 1
            V[get_v_idx_b(1,1), col] = 1
            W[get_w_idx(0,0), col] = 1
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
