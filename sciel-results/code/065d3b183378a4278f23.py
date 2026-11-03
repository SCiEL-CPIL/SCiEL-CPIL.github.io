"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Get Strassen factors for 2x2
    U2, V2, W2 = get_strassen_2x2()
    
    # Outer Strassen steps for 4x4
    # Each step is (u_coeffs, v_coeffs, w_block_index)
    # u_coeffs: [c0, c1, c2, c3] for A blocks
    # v_coeffs: [c0, c1, c2, c3] for B blocks
    # w_block_index: which of the 4 output blocks this contributes to, and with what sign?
    # Actually, the inner Strassen W2 handles the distribution within the 2x2 block.
    # The outer step determines WHICH 2x2 block of the 4x4 output is affected.
    
    # Outer steps:
    # 1. (A0+A3)(B0+B3) -> C0 (+1), C3 (+1) ? No.
    # Let's use the standard Strassen formulas for 2x2 blocks.
    # C0 = M1 - M4 + M6 - M7
    # C1 = M4 + M5
    # C2 = M2 + M3
    # C3 = M1 - M2 + M3 + M7
    
    # M1 = (A0+A3)(B0+B3)
    # M2 = (A2+A3)B3
    # M3 = A0(B2-B3)
    # M4 = (A1-A0)(B0+B2)
    # M5 = (A0+A1)B3
    # M6 = (A2-A0)(B0+B1)
    # M7 = (A1+A3)(B1-B3)
    
    outer_steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1 -> C0(+1), C3(+1) is handled by W2? No.
        # W2 for M1 is [1, 0, 0, 1] which means C00(+1), C11(+1).
        # In the 4x4 context, C00 of the block corresponds to C00 of the 4x4.
        # C11 of the block corresponds to C11 of the 4x4? No.
        # The block is 2x2. The result of M1 is a 2x2 matrix.
        # This 2x2 matrix is added to C0 (top-left) and C3 (bottom-right)?
        # No, Strassen computes the 4 blocks C0, C1, C2, C3 of the 4x4 matrix.
        # M1 is used in C0 and C3.
        # So the result of M1 (a 2x2 matrix) is added to C0 with +1 and C3 with +1.
        
        # Therefore, for each outer step, we need to know which output blocks it contributes to.
        # And the W2 vector tells us how the 2x2 result maps to the 2x2 block.
        
        # Let's refine the outer steps list to include output block indices and signs.
        # Format: (u_coeffs, v_coeffs, [(block_idx, sign), ...])
    ]
    
    # M1: (A0+A3)(B0+B3)
    # Contributes to C0 (+1) and C3 (+1)
    # M2: (A2+A3)B3
    # Contributes to C2 (+1) and C3 (-1)
    # M3: A0(B2-B3)
    # Contributes to C2 (+1) and C3 (+1)
    # M4: (A1-A0)(B0+B2)
    # Contributes to C0 (-1) and C1 (+1)
    # M5: (A0+A1)B3
    # Contributes to C1 (+1)
    # M6: (A2-A0)(B0+B1)
    # Contributes to C0 (-1)
    # M7: (A1+A3)(B1-B3)
    # Contributes to C0 (-1) and C3 (+1)
    
    outer_steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 1), (3, 1)]),
        ([0, 0, 1, 1], [0, 0, 0, 1], [(2, 1), (3, -1)]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [(2, 1), (3, 1)]),
        ([-1, 1, 0, 0], [1, 0, 1, 0], [(0, -1), (1, 1)]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [(1, 1)]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(0, -1)]),
        ([0, 1, 0, 1], [0, 1, 0, -1], [(0, -1), (3, 1)])
    ]
    
    rank = 49
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    col = 0
    for u_c, v_c, w_blocks in outer_steps:
        for j in range(7):
            # u_c is 4-vector, u_inner is 4-vector
            # Full U is 16-vector
            U4[:, col] = np.concatenate([u_c[i] * U2[:, j] for i in range(4)])
            V4[:, col] = np.concatenate([v_c[i] * V2[:, j] for i in range(4)])
            
            # W is scattered
            W_full = np.zeros(16)
            for block_idx, sign in w_blocks:
                # Map W2[:, j] (4-vector) to block_idx in W_full
                # Block 0: indices 0,1,4,5
                # Block 1: indices 2,3,6,7
                # Block 2: indices 8,9,12,13
                # Block 3: indices 10,11,14,15
                
                w_vec = W2[:, j] * sign
                if block_idx == 0:
                    W_full[0] = w_vec[0]
                    W_full[1] = w_vec[1]
                    W_full[4] = w_vec[2]
                    W_full[5] = w_vec[3]
                elif block_idx == 1:
                    W_full[2] = w_vec[0]
                    W_full[3] = w_vec[1]
                    W_full[6] = w_vec[2]
                    W_full[7] = w_vec[3]
                elif block_idx == 2:
                    W_full[8] = w_vec[0]
                    W_full[9] = w_vec[1]
                    W_full[12] = w_vec[2]
                    W_full[13] = w_vec[3]
                elif block_idx == 3:
                    W_full[10] = w_vec[0]
                    W_full[11] = w_vec[1]
                    W_full[14] = w_vec[2]
                    W_full[15] = w_vec[3]
            
            W4[:, col] = W_full
            col += 1
            
    # Now construct 4x4x8 by combining two 4x4x4
    U_total = np.zeros((16, 98))
    V_total = np.zeros((32, 98))
    W_total = np.zeros((32, 98))
    
    # First 49 columns
    U_total[:, 0:49] = U4
    # V_total for first 49: B1 (cols 0-3 of B)
    # B1 indices in V_total: j*8 + k for j in 0..3, k in 0..3
    # U4 indices: j*4 + k for j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            V_total[j*8 + k, 0:49] = V4[j*4 + k, 0:49]
            
    # W_total for first 49: C1 (cols 0-3 of C)
    # C1 indices in W_total: k*4 + i for k in 0..3, i in 0..3
    # W4 indices: k*4 + i for k in 0..3, i in 0..3
    W_total[0:16, 0:49] = W4[:, 0:49] # Wait, W4 is 16x49.
    # W4 row index is k*4+i.
    # W_total row index is k*4+i for k in 0..3.
    # So W_total[0:16, 0:49] = W4[:, 0:49] is correct.
    
    # Next 49 columns
    U_total[:, 49:98] = U4
    # V_total for next 49: B2 (cols 4-7 of B)
    for j in range(4):
        for k in range(4):
            V_total[j*8 + (k+4), 49:98] = V4[j*4 + k, 0:49]
            
    # W_total for next 49: C2 (cols 4-7 of C)
    # C2 indices in W_total: k*4 + i for k in 4..7.
    # So row index is k*4 + i.
    # W4 row index is k'*4 + i for k' in 0..3.
    # Map k' to k = k' + 4.
    for k_prime in range(4):
        for i in range(4):
            W_total[(k_prime+4)*4 + i, 49:98] = W4[k_prime*4 + i, 0:49]
            
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

def get_strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [0, 0, 0, 1]
    W[:, 1] = [0, 0, 1, -1]
    
    # M3
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 0, 1, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    # M4
    U[:, 3] = [-1, 1, 0, 0]
    V[:, 3] = [1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    # M5
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [0, 1, 0, 0]
    
    # M6
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [-1, 0, 0, 0]
    
    # M7
    U[:, 6] = [0, 1, 0, 1]
    V[:, 6] = [0, 1, 0, -1]
    W[:, 6] = [-1, 0, 0, 1]
    
    return U, V, W
# EVOLVE-BLOCK-END
