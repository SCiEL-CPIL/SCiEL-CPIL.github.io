import numpy as np
import time

def construct():
    # Deadline guard: The evaluator kills after 150s.
    # We should return the best result found before 120s to be safe.
    t0 = time.time()
    limit = t0 + 115.0  # 5s margin

    n, m, p = 4, 4, 8

    # Rank-98 decomposition: two independent Strassen-based 4x4x4 blocks.
    # A is 4x4, B is 4x8 (split into two 4x4 halves), C is 8x4 (two 4x4 halves).
    # Each 4x4x4 uses a rank-49 decomposition built from 7 "super-Strassen"
    # products, each of which is a 2x2x2 Strassen product (rank 7).
    # Total rank = 2 * 49 = 98.

    rank = 98
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 2x2x2 factor matrices (4 entries each side, 7 products)
    # Index convention: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # P0 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # P1 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    # P2 = A00(B01-B11)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    # P3 = A11(B10-B00)
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    # P4 = (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    # P5 = (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # P6 = (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1

    # For each 4x4x4 block, define 7 "super-products" P_k
    # A_eff = sum of specified 2x2 blocks of A
    # B_eff = sum of specified 2x2 blocks of B
    # C contribution: P_k is added/subtracted from specified C blocks
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)

    A_spec = [
        [(1,0),(1,3)],  # P0: A00+A11
        [(1,2),(1,3)],  # P1: A10+A11
        [(1,0)],        # P2: A00
        [(1,3)],        # P3: A11
        [(1,0),(1,1)],  # P4: A00+A01
        [(1,2),(-1,0)], # P5: A10-A00
        [(1,1),(-1,3)], # P6: A01-A11
    ]
    B_spec = [
        [(1,0),(1,3)],  # P0: B00+B11
        [(1,0)],        # P1: B00
        [(1,1),(-1,3)], # P2: B01-B11
        [(1,2),(-1,0)], # P3: B10-B00
        [(1,3)],        # P4: B11
        [(1,0),(1,1)],  # P5: B00+B01
        [(1,2),(1,3)],  # P6: B10+B11
    ]
    C_spec = [
        [(1,0),(1,3)],   # P0: C00+C11
        [(1,2),(-1,3)],  # P1: C10-C11
        [(1,1),(1,3)],   # P2: C01+C11
        [(1,0),(1,2)],   # P3: C00+C10
        [(-1,0),(1,1)],  # P4: -C00+C01
        [(1,3)],         # P5: C11
        [(1,0)],         # P6: C00
    ]

    # Precompute the 2x2 block-to-element mapping
    # For a 2x2 block with top-left at (br, bc) in a 4x4 matrix:
    #   element (r,c) in block -> global (br*2+r, bc*2+c)
    block_offsets = []
    for bi in range(4):
        br, bc = divmod(bi, 2)
        block_offsets.append((br, bc))

    # Precompute pattern matrices for each kp (0..6)
    # U_pattern[kp] has shape (16, 7), where col ms corresponds to the 7th Strassen product
    U_patterns = []
    V_patterns = []
    W_patterns = []

    for kp in range(7):
        Up = np.zeros((16, 7))
        Vp = np.zeros((32, 7))
        Wp = np.zeros((32, 7))
        
        for ms in range(7):
            # --- U: Input A (4x4) ---
            for ca, bidx in A_spec[kp]:
                br, bc = block_offsets[bidx]
                for r in range(2):
                    for c in range(2):
                        a_idx = (br*2+r)*4 + (bc*2+c)
                        Up[a_idx, ms] += ca * U2[r*2+c, ms]

            # --- V: Input B (4x8) ---
            # For half=0, B col is bc*2+c. For half=1, it is bc*2+c+4.
            # We compute for half=0 here.
            for cb, bidx in B_spec[kp]:
                br, bc = block_offsets[bidx]
                for r in range(2):
                    for c in range(2):
                        b_idx = (br*2+r)*8 + (bc*2+c)
                        Vp[b_idx, ms] += cb * V2[r*2+c, ms]

            # --- W: Output C (8x4) ---
            # C index convention: W[k*n+i, r] where k is col of C (0..7), i is row of C (0..3)
            # So C element at row i, col k -> index k*4 + i
            # For half=0, C col is bj*2+c. For half=1, it is bj*2+c+4.
            for cc, cidx in C_spec[kp]:
                bi, bj = divmod(cidx, 2)
                for r in range(2):
                    for c in range(2):
                        C_row = bi*2 + r
                        C_col = bj*2 + c
                        w_idx = C_col*4 + C_row
                        Wp[w_idx, ms] += cc * W2[r*2+c, ms]
                        
        U_patterns.append(Up)
        V_patterns.append(Vp)
        W_patterns.append(Wp)

    # Fill U, V, W for both halves
    for half in range(2):
        base_col = half * 49
        for kp in range(7):
            col_start = base_col + kp * 7
            col_end = col_start + 7
            
            U[:, col_start:col_end] = U_patterns[kp]
            
            # V needs offset for the second half
            if half == 0:
                V[:, col_start:col_end] = V_patterns[kp]
            else:
                # Shift columns 0..3 by +4 in the B index (which is the lower 4 bits of the row index for 4x8)
                # Row index in V is (br*2+r)*8 + (bc*2+c).
                # For half=1, we want (br*2+r)*8 + (bc*2+c+4).
                # This is equivalent to V_patterns[kp] but with rows shifted?
                # No, V_patterns[kp] is indexed by global B index for half=0.
                # We need to map these to half=1 indices.
                # Let V_half0 = V_patterns[kp].
                # We create V_half1 such that V_half1[row, ms] = V_half0[row, ms] if row < 32?
                # Wait, V has 32 rows (4*8).
                # Half 0 uses B cols 0..3. Half 1 uses B cols 4..7.
                # In the V matrix, row index is r_B * 8 + c_B.
                # For half 0, c_B in 0..3.
                # For half 1, c_B in 4..7.
                # So for a given logical position (r_B, c_B_local), 
                # half 0 index is r_B*8 + c_B_local.
                # half 1 index is r_B*8 + c_B_local + 4.
                
                # We can just copy the pattern and shift the rows?
                # No, the pattern V_patterns[kp] already has zeros for c_B >= 4.
                # So we can take V_patterns[kp], and for each non-zero entry at row i, 
                # put it at row i+4 in the target?
                # Yes.
                V_half1 = np.zeros_like(V_patterns[kp])
                # Rows 0..15 correspond to c_B=0..3? No.
                # Row index = r*8 + c.
                # If c=0, row=r*8. If c=1, row=r*8+1...
                # The "shift" is +4 in the column part of the index.
                # So row_new = row_old + 4?
                # Let's check: 
                # r=0, c=0 -> idx 0. Half 1: r=0, c=4 -> idx 4.
                # r=0, c=1 -> idx 1. Half 1: r=0, c=5 -> idx 5.
                # r=1, c=0 -> idx 8. Half 1: r=1, c=4 -> idx 12.
                # Yes, simply adding 4 to the row index works for the first 16 rows?
                # What about r=2, c=0 -> idx 16. Half 1: r=2, c=4 -> idx 20.
                # Yes, row + 4 works for all rows where c < 4.
                # Since V_patterns only has non-zeros for c < 4 (because B_spec uses blocks 0..3 which map to c=0..3), 
                # this shift is valid.
                V_half1[4:, :] = V_patterns[kp][:28, :] # Wait, shape is (32,7). 
                # Rows 0..31.
                # Non-zeros are in rows where c<4.
                # c = row % 8.
                # So rows where row % 8 < 4 are non-zero.
                # We want to move them to row+4.
                # V_half1[row+4, ms] = V_patterns[kp][row, ms]
                
                # Easier:
                V_half1 = np.zeros_like(V_patterns[kp])
                for r in range(32):
                    if V_patterns[kp][r, :].any():
                        V_half1[r+4, :] = V_patterns[kp][r, :]
                V[:, col_start:col_end] = V_half1

            # W needs offset for the second half
            if half == 0:
                W[:, col_start:col_end] = W_patterns[kp]
            else:
                # W index: k*4 + i. k is col of C.
                # Half 0: k in 0..3. Half 1: k in 4..7.
                # So shift k by +4.
                # W index = k*4 + i.
                # New index = (k+4)*4 + i = k*4 + i + 16.
                # So shift row index by +16.
                W_half1 = np.zeros_like(W_patterns[kp])
                for r in range(32):
                    if W_patterns[kp][r, :].any():
                        W_half1[r+16, :] = W_patterns[kp][r, :]
                W[:, col_start:col_end] = W_half1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
