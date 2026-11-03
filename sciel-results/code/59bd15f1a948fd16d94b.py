import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Mapping of 2x2 block indices to global indices
    # A is 4x4, index = row*m + col
    # Block (br, bc) contains local (lr, lc)
    # Global row = br*2 + lr, Global col = bc*2 + lc
    # Global index = (br*2 + lr)*4 + (bc*2 + lc)
    
    def get_a_idx(br, bc, lr, lc):
        return (br * 2 + lr) * 4 + (bc * 2 + lc)
        
    def get_b_idx(br, bc, lr, lc):
        # B is 4x4 (for the first part)
        # Rows of B correspond to cols of A? No, B is m x p = 4 x 4.
        # Index = row*p + col? No, standard matrix mult C = A B.
        # A is n x m. B is m x p.
        # A index: i*m + j.
        # B index: j*p + k.
        # Here n=4, m=4, p=4 for the first part.
        # B index: j*4 + k.
        # Block (br, bc) in B?
        # B rows are j (0..3), cols are k (0..3).
        # We split B into 2x2 blocks based on j and k.
        # Block (br, bc) -> j in [br*2, br*2+1], k in [bc*2, bc*2+1].
        # Local (lr, lc) -> j = br*2 + lr, k = bc*2 + lc.
        # Index = j*4 + k.
        return (br * 2 + lr) * 4 + (bc * 2 + lc)

    def get_c_idx(br, bc, lr, lc):
        # C is n x p = 4 x 4.
        # Index = i*p + k?
        # Task says: W[k*n+i, r].
        # So C index is k*n + i.
        # C rows are i (0..3), cols are k (0..3).
        # Block (br, bc) -> i in [br*2, br*2+1], k in [bc*2, bc*2+1].
        # Local (lr, lc) -> i = br*2 + lr, k = bc*2 + lc.
        # Index = k*4 + i.
        return (bc * 2 + lc) * 4 + (br * 2 + lr)

    # Strassen 4x4x4
    # 7 outer terms
    outer_terms = [
        # (A_blocks, B_blocks, C_blocks, signs)
        # A: list of (block, sign)
        # B: list of (block, sign)
        # C: list of (block, sign)
        # Block is (br, bc)
        
        # M1 = (A00 + A11)(B00 + B11) -> C00(+1), C11(+1)
        # Wait, C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the contribution of each Mk to the C blocks.
        # Then inside Mk, we use Strassen 2x2x2.
    ]
    
    # Define outer Strassen steps
    # Each step k has:
    # X_k = sum sa * A_block
    # Y_k = sum sb * B_block
    # Z_k = X_k Y_k
    # Z_k contributes to C blocks with weights.
    
    steps = [
        # k=0: M1
        {
            'X': [((0,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)] # C00 += M1, C11 += M1
        },
        # k=1: M2
        {
            'X': [((1,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), -1)] # C10 += M2, C11 -= M2
        },
        # k=2: M3
        {
            'X': [((0,0), 1)],
            'Y': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1), ((1,1), 1)] # C01 += M3, C11 += M3
        },
        # k=3: M4
        {
            'X': [((1,1), 1)],
            'Y': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)] # C00 += M4, C10 += M4
        },
        # k=4: M5
        {
            'X': [((0,0), 1), ((0,1), 1)],
            'Y': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)] # C00 -= M5, C01 += M5
        },
        # k=5: M6
        {
            'X': [((1,0), 1), ((0,0), -1)],
            'Y': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)] # C11 += M6
        },
        # k=6: M7
        {
            'X': [((0,1), 1), ((1,1), -1)],
            'Y': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)] # C00 += M7
        }
    ]
    
    # Inner Strassen 2x2x2
    # X, Y are 2x2. Z = XY.
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    inner_steps = [
        # j=0: S1
        {
            'X': [((0,0), 1), ((1,1), 1)], # X00 + X11
            'Y': [((0,0), 1), ((1,1), 1)], # Y00 + Y11
            'Z': [((0,0), 1), ((1,1), 1)]  # Z00 += S1, Z11 += S1
        },
        # j=1: S2
        {
            'X': [((1,0), 1), ((1,1), 1)], # X10 + X11
            'Y': [((0,0), 1)],              # Y00
            'Z': [((1,0), 1), ((1,1), -1)]  # Z10 += S2, Z11 -= S2
        },
        # j=2: S3
        {
            'X': [((0,0), 1)],              # X00
            'Y': [((0,1), 1), ((1,1), -1)], # Y01 - Y11
            'Z': [((0,1), 1), ((1,1), 1)]   # Z01 += S3, Z11 += S3
        },
        # j=3: S4
        {
            'X': [((1,1), 1)],              # X11
            'Y': [((1,0), 1), ((0,0), -1)], # Y10 - Y00
            'Z': [((0,0), 1), ((1,0), 1)]   # Z00 += S4, Z10 += S4
        },
        # j=4: S5
        {
            'X': [((0,0), 1), ((0,1), 1)],  # X00 + X01
            'Y': [((1,1), 1)],              # Y11
            'Z': [((0,0), -1), ((0,1), 1)]  # Z00 -= S5, Z01 += S5
        },
        # j=5: S6
        {
            'X': [((1,0), 1), ((0,0), -1)], # X10 - X00
            'Y': [((0,0), 1), ((0,1), 1)],  # Y00 + Y01
            'Z': [((1,1), 1)]               # Z11 += S6
        },
        # j=6: S7
        {
            'X': [((0,1), 1), ((1,1), -1)], # X01 - X11
            'Y': [((1,0), 1), ((1,1), 1)],  # Y10 + Y11
            'Z': [((0,0), 1)]               # Z00 += S7
        }
    ]
    
    col = 0
    for step in steps:
        # Construct linear combos for X_k and Y_k in terms of global A and B indices
        # X_k is a 2x2 matrix (conceptually), but we are multiplying blocks.
        # Actually, X_k is a sum of A blocks. Each A block is 2x2.
        # So X_k is a 2x2 matrix of "values" which are linear combos of A entries?
        # No.
        # Let's trace the indices.
        
        # For a fixed outer step k:
        # X_k = sum_{(ba, sa)} sa * A_{ba}
        # Y_k = sum_{(bb, sb)} sb * B_{bb}
        # Z_k = X_k Y_k
        
        # Z_k is a 2x2 matrix (block).
        # We decompose Z_k using inner Strassen.
        # The "entries" of X_k and Y_k in the inner Strassen are the entries of the blocks.
        
        # Let's expand.
        # X_k has entries X_k[lr][lc] = sum_{(ba, sa)} sa * A_{ba}[lr][lc]
        # Y_k has entries Y_k[lr][lc] = sum_{(bb, sb)} sb * B_{bb}[lr][lc]
        
        # For each inner step j:
        # S_j = (sum_{lx, sx} sx * X_k[lx]) * (sum_{ly, sy} sy * Y_k[ly])
        # Here lx, ly are local indices (0..3) representing (lr, lc).
        # Wait, inner Strassen operates on the 2x2 entries.
        # X_k is 2x2. Its entries are scalars (which are linear combos of A).
        # So X_k[0] = X_k[0,0], X_k[1] = X_k[0,1], X_k[2] = X_k[1,0], X_k[3] = X_k[1,1].
        
        # Let's precompute the linear combo for each local entry of X_k and Y_k.
        
        # X_k local entries:
        # X_k[0] (0,0) = sum sa * A_{ba}[0][0]
        # X_k[1] (0,1) = sum sa * A_{ba}[0][1]
        # X_k[2] (1,0) = sum sa * A_{ba}[1][0]
        # X_k[3] (1,1) = sum sa * A_{ba}[1][1]
        
        # We need to map A_{ba}[lr][lc] to global index.
        
        x_combos = [[0]*4 for _ in range(4)] # x_combos[local_idx] is list of (global_idx, coeff)
        # Actually, we just need to add to U.
        
        # Let's iterate inner steps.
        for inner in inner_steps:
            # Determine U coefficients (from X_k and inner X)
            # U will be the product of coefficients from X_k and inner X.
            
            # Inner X is a linear combo of X_k's local entries.
            # inner['X'] is list of (local_idx, coeff)
            
            # For each (lx, sx) in inner['X']:
            #   sx * X_k[lx]
            #   X_k[lx] is sum_{(ba, sa)} sa * A_{ba}[lx_row][lx_col]
            
            # So U is sum of terms: sx * sa * A_global_idx
            
            u_coeffs = {} # global_idx -> coeff
            for lx, sx in inner['X']:
                lx_row = lx // 2
                lx_col = lx % 2
                for ba, sa in step['X']:
                    ba_row, ba_col = ba
                    global_idx = get_a_idx(ba_row, ba_col, lx_row, lx_col)
                    coeff = sx * sa
                    u_coeffs[global_idx] = u_coeffs.get(global_idx, 0) + coeff
            
            # Determine V coefficients (from Y_k and inner Y)
            v_coeffs = {}
            for ly, sy in inner['Y']:
                ly_row = ly // 2
                ly_col = ly % 2
                for bb, sb in step['Y']:
                    bb_row, bb_col = bb
                    global_idx = get_b_idx(bb_row, bb_col, ly_row, ly_col)
                    coeff = sy * sb
                    v_coeffs[global_idx] = v_coeffs.get(global_idx, 0) + coeff
            
            # Determine W coefficients (from C blocks and inner Z)
            # Z_k contributes to C blocks.
            # inner['Z'] is list of (local_idx, coeff)
            # Z_k[local_idx] contributes to C_block[local_idx]
            # And C_block contributes to global C.
            
            w_coeffs = {}
            for lz, sz in inner['Z']:
                lz_row = lz // 2
                lz_col = lz % 2
                for bc, sc in step['C']:
                    bc_row, bc_col = bc
                    global_idx = get_c_idx(bc_row, bc_col, lz_row, lz_col)
                    coeff = sz * sc
                    w_coeffs[global_idx] = w_coeffs.get(global_idx, 0) + coeff
            
            # Add rank-1 term
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] = val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] = val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] = val
            col += 1
            
    # Now add schoolbook for the remaining 3 columns (indices 4,5,6)
    # B columns 4,5,6. C columns 4,5,6.
    # A is full 4x4.
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
