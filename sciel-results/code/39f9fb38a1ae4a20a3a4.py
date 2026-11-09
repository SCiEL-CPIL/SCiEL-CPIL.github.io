import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # We construct a rank-56 decomposition based on 4 independent pairs of 2x2 Strassen multiplications.
    # A is 2x4, split into A_left (2x2) and A_right (2x2).
    # B is 4x8, split vertically into B_top (2x8) and B_bottom (2x8).
    # C = A_left * B_top + A_right * B_bottom.
    # C is 2x8, split into 4 blocks C_k (2x2).
    # C_k = A_left * B_top_k + A_right * B_bottom_k.
    # Each term is a 2x2 * 2x2 multiplication.
    # Strassen rank for 2x2 is 7.
    # Since variables are disjoint between the two terms, we need 7+7=14 per block.
    # 4 blocks * 14 = 56 total rank.

    # Strassen 2x2x2 decomposition coefficients
    # Inputs X (2x2) = [[x00, x01], [x10, x11]]
    # Inputs Y (2x2) = [[y00, y01], [y10, y11]]
    # Outputs Z (2x2) = [[z00, z01], [z10, z11]]
    
    # Linear forms for X
    # m1 = (x00 + x11) * (y00 + y11)
    # m2 = (x10 + x11) * y00
    # m3 = x00 * (y01 - y11)
    # m4 = x11 * (y10 - y00)
    # m5 = (x00 + x01) * y11
    # m6 = (x10 - x00) * (y00 + y10)
    # m7 = (x01 - x11) * (y11 + y01)
    
    # We need to map these to the global indices.
    # For each block k (0..3):
    #   Term 1: A_left * B_top_k
    #   Term 2: A_right * B_bottom_k
    
    # A_left indices in A (2x4):
    # A[i, j] where i in {0,1}, j in {0,1} for Left, j in {2,3} for Right.
    # Global A index: i*4 + j
    
    # B_top_k indices in B (4x8):
    # B[j, k*2 + col] where j in {0,1} for Top, j in {2,3} for Bottom.
    # Global B index: j*8 + col_in_block + k*2
    
    # C indices in C (2x8):
    # C[i, k*2 + col]
    # Global C index: k*2 + col ... wait, trace convention W[k*n+i, r].
    # So C index is (k*n + i). Here n=2. So index is 2*k + i.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen steps
    # Each step has coefficients for X, Y, and Z
    # X coeffs: [x00, x01, x10, x11]
    # Y coeffs: [y00, y01, y10, y11]
    # Z coeffs: [z00, z01, z10, z11]
    
    strassen_steps = [
        # 1: (x00+x11)(y00+y11)
        {'x': [1, 0, 0, 1], 'y': [1, 0, 0, 1], 'z': [1, 0, 0, -1]},
        # 2: (x10+x11)y00
        {'x': [0, 0, 1, 1], 'y': [1, 0, 0, 0], 'z': [0, 0, 0, 1]},
        # 3: x00(y01-y11)
        {'x': [1, 0, 0, 0], 'y': [0, 1, 0, -1], 'z': [0, 1, 0, 0]},
        # 4: x11(y10-y00)
        {'x': [0, 0, 0, 1], 'y': [-1, 0, 1, 0], 'z': [0, 0, 1, 0]},
        # 5: (x00+x01)y11
        {'x': [1, 1, 0, 0], 'y': [0, 0, 0, 1], 'z': [1, -1, 0, 0]},
        # 6: (x10-x00)(y00+y10)
        {'x': [-1, 0, 1, 0], 'y': [1, 0, 1, 0], 'z': [0, -1, 1, 0]},
        # 7: (x01-x11)(y11+y01)
        {'x': [0, 1, 0, -1], 'y': [0, 1, 0, 1], 'z': [0, 1, 0, -1]},
    ]
    
    r = 0
    for k in range(4):
        # Process Term 1: A_left * B_top_k
        for s in range(7):
            # U: A_left
            # A_left is A[i, j] for i in 0..1, j in 0..1
            # Global A idx: i*4 + j
            for i in range(2):
                for j in range(2):
                    a_idx = i * 4 + j
                    # Map local j (0..1) to strassen x index (0..3)
                    # Local matrix [[x00, x01], [x10, x11]]
                    # j=0 -> col 0, j=1 -> col 1
                    x_idx = i * 2 + j
                    U[a_idx][r] = strassen_steps[s]['x'][x_idx]
            
            # V: B_top_k
            # B_top is B[j, col] for j in 0..1, col in 0..1 relative to block k
            # Global B idx: j*8 + (k*2 + col)
            for j in range(2):
                for col in range(2):
                    b_idx = j * 8 + (k * 2 + col)
                    # Map local col (0..1) to strassen y index
                    # Local matrix [[y00, y01], [y10, y11]]
                    # j=0 -> row 0, j=1 -> row 1
                    # col=0 -> col 0, col=1 -> col 1
                    y_idx = j * 2 + col
                    V[b_idx][r] = strassen_steps[s]['y'][y_idx]
            
            # W: C_k
            # C_k is C[i, col] for i in 0..1, col in 0..1 relative to block k
            # Global C idx: (k*2 + col)*2 + i ? No, W[k*n+i, r]
            # The problem statement says W[k*n+i, r].
            # Wait, "output factor follows the official trace convention W[k*n+i,r]".
            # Usually trace is sum over k of C[i,k] * W[k*n+i].
            # Here C is 2x8. So k goes 0..7.
            # The index in W is k*n + i.
            # For block k (0..3), the columns are 2k and 2k+1.
            # So for local col c, global col is 2k+c.
            # W index is (2k+c)*2 + i.
            
            for i in range(2):
                for col in range(2):
                    c_idx = (k * 2 + col) * n + i
                    # Map local (i, col) to strassen z index
                    # Local matrix [[z00, z01], [z10, z11]]
                    # i=0 -> row 0, i=1 -> row 1
                    # col=0 -> col 0, col=1 -> col 1
                    z_idx = i * 2 + col
                    W[c_idx][r] = strassen_steps[s]['z'][z_idx]
            
            r += 1
        
        # Process Term 2: A_right * B_bottom_k
        for s in range(7):
            # U: A_right
            # A_right is A[i, j] for i in 0..1, j in 2..3
            # Global A idx: i*4 + j
            for i in range(2):
                for j in range(2):
                    a_idx = i * 4 + (2 + j)
                    x_idx = i * 2 + j
                    U[a_idx][r] = strassen_steps[s]['x'][x_idx]
            
            # V: B_bottom_k
            # B_bottom is B[j, col] for j in 2..3, col in 0..1 relative to block k
            # Global B idx: j*8 + (k*2 + col)
            for j in range(2):
                for col in range(2):
                    b_idx = (2 + j) * 8 + (k * 2 + col)
                    y_idx = j * 2 + col
                    V[b_idx][r] = strassen_steps[s]['y'][y_idx]
            
            # W: C_k (same as before, add to W)
            for i in range(2):
                for col in range(2):
                    c_idx = (k * 2 + col) * n + i
                    z_idx = i * 2 + col
                    W[c_idx][r] = strassen_steps[s]['z'][z_idx]
            
            r += 1

    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
