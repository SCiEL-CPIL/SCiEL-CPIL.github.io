import time
import numpy as np

def construct():
    """
    Rank-28 decomposition for 2x4 * 4x8 matrix multiplication.
    Strategy:
    1. Split B (4x8) into B_L (4x4) and B_R (4x4).
    2. Split A (2x4) into A_L (2x2) and A_R (2x2).
    3. Split B_L into B_L_top (2x4) and B_L_bot (2x4).
    4. Split B_R into B_R_top (2x4) and B_R_bot (2x4).
    5. Compute C_L = A_L B_L_top + A_R B_L_bot.
    6. Compute C_R = A_L B_R_top + A_R B_R_bot.
    7. Each product (e.g., A_L B_L_top) is a 2x2 by 2x4 multiplication.
       This is decomposed into two 2x2 by 2x2 Strassen products (rank 7 each),
       totaling rank 14.
    8. Total rank = 14 (for C_L) + 14 (for C_R) = 28.
    """
    deadline = time.time() + 120
    
    n, m, p = 2, 4, 8
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for 2x2x2
    # U_s: (4, 7), V_s: (4, 7), W_s: (4, 7)
    # Indices for 2x2x2: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # Term 2: M3 = A00(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # Helper to add a 2x2x2 Strassen block to the global tensors
    # a_offset: offset for A indices (i*2 + j) in global A (i*4 + j_global)
    # b_offset: offset for B indices (j*2 + k) in global B (j_global*8 + k_global)
    # c_offset: offset for C indices (k*2 + i) in global C (k_global*2 + i)
    # col_start: starting column in U, V, W
    
    def add_strassen_block(a_row_offset, b_row_offset, c_row_offset, col_start):
        # a_row_offset: 0 for A_L (j=0,1), 2 for A_R (j=2,3)
        # b_row_offset: 0 for top (j=0,1), 2 for bot (j=2,3)
        # c_row_offset: 0 for left (k=0..3), 4 for right (k=4..7)
        # Note: c_row_offset is for the k index in the 2x2 block.
        # The global k is k_global = c_row_offset + k_local.
        
        for s in range(7):
            col = col_start + s
            # U: A part
            # Local A index: i*2 + j
            # Global A index: i*4 + (a_row_offset + j)
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        global_i = i
                        global_j = a_row_offset + j
                        U[global_i * 4 + global_j, col] = val
            
            # V: B part
            # Local B index: j*2 + k
            # Global B index: (b_row_offset + j)*8 + (c_row_offset + k)
            # Wait, c_row_offset is for the k block.
            # The 2x2 block of B has k in {0,1} or {2,3} relative to the 4x4 block?
            # No, the 2x2 block of B is 2x2.
            # The 2x4 block of B is split into two 2x2 blocks.
            # So we call this function twice for each 2x4 block.
            # Let's assume this function handles a 2x2x2 block.
            
            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        global_j = b_row_offset + j
                        global_k = c_row_offset + k
                        V[global_j * 8 + global_k, col] = val
            
            # W: C part
            # Local C index: k*2 + i
            # Global C index: (c_row_offset + k)*2 + i
            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        global_k = c_row_offset + k
                        global_i = i
                        W[global_k * 2 + global_i, col] += val

    # We need to add 4 such blocks, each of which is a 2x2x4 product decomposed into two 2x2x2.
    # So 8 Strassen blocks total.
    # Block 1: A_L, B_L_top, C_L_left (k=0,1)
    add_strassen_block(0, 0, 0, 0)
    # Block 2: A_L, B_L_top, C_L_right (k=2,3)
    add_strassen_block(0, 0, 2, 7)
    # Block 3: A_R, B_L_bot, C_L_left (k=0,1)
    add_strassen_block(2, 2, 0, 14)
    # Block 4: A_R, B_L_bot, C_L_right (k=2,3)
    add_strassen_block(2, 2, 2, 21)
    # Block 5: A_L, B_R_top, C_R_left (k=4,5)
    add_strassen_block(0, 0, 4, 28) # Wait, rank is 28. This exceeds.
    
    # Ah, I made a mistake.
    # Rank of T_2,2,4 is 14.
    # T_2,4,8 = T_2,2,4 (for A_L B_L) + T_2,2,4 (for A_R B_L) + ...
    # No.
    # C_L = A_L B_L_top + A_R B_L_bot.
    # A_L B_L_top is 2x2 by 2x4. Rank 14.
    # A_R B_L_bot is 2x2 by 2x4. Rank 14.
    # Total for C_L is 28.
    # C_R is another 28.
    # Total 56.
    
    # Where is the sharing?
    # A_L is used in C_L and C_R.
    # A_L B_L_top and A_L B_R_top.
    # Both use A_L.
    # The Strassen U-factors for A_L are the same.
    # So we compute 7 U-factors for A_L.
    # Then we multiply them with B_L_top (7 mults) and B_R_top (7 mults).
    # Total 14 mults for A_L part.
    # Similarly for A_R: 14 mults.
    # Total 28 mults.
    
    # So the rank IS 28.
    # My previous count of 56 was because I didn't share the U-factors.
    # But in the tensor decomposition, the U matrix is fixed.
    # If we use the same U-factors for two different V-blocks, we still need separate terms?
    # No. The term is $u(A) v(B)$.
    # If $u(A)$ is the same, but $v(B)$ is different, they are different terms.
    # Term 1: $u_1(A_L) v_1(B_{L,top})$
    # Term 2: $u_1(A_L) v_2(B_{R,top})$
    # These are two distinct multiplications.
    # So the rank is indeed the number of multiplications.
    # 7 (for A_L, B_L) + 7 (for A_L, B_R) = 14 for A_L.
    # 7 (for A_R, B_L) + 7 (for A_R, B_R) = 14 for A_R.
    # Total 28.
    
    # So my code above should work with rank 28.
    # But I called add_strassen_block 5 times above, which is 35 terms.
    # I need to call it 4 times?
    # No, each add_strassen_block adds 7 terms.
    # I need 28 terms. So 4 calls.
    # But I have 4 products: A_L B_L, A_R B_L, A_L B_R, A_R B_R.
    # Each is 2x2 by 2x4.
    # Each is decomposed into TWO 2x2x2 blocks.
    # So 4 products * 2 blocks/product = 8 blocks.
    # 8 blocks * 7 terms/block = 56 terms.
    
    # Contradiction!
    # If I don't share, it's 56.
    # If I share U-factors, do I save rank?
    # Rank is the number of terms $r$.
    # Term $r$ is defined by $u_r, v_r, w_r$.
    # If $u_1 = u_2$, but $v_1 \neq v_2$, they are different terms.
    # So sharing U-factors does NOT reduce the rank.
    # It reduces the number of linear combinations computed, but not the number of multiplications.
    # The rank is the number of multiplications.
    
    # So the rank is 56?
    # Then why did I think 28?
    # Maybe there is a better decomposition.
    
    # Let's check the rank of T_2,2,4 again.
    # Is it 14?
    # Yes, 7+7=14.
    # So T_2,4,8 is 4 * 14 = 56.
    
    # Is there a way to get lower?
    # What if we use a different split?
    # T_2,4,8.
    # Maybe T_2,4,8 rank is 48?
    # Let's try to find a rank 48 decomposition.
    
    # For now, I will return the rank 56 solution as it is correct.
    # The current seed is 56.
    # If I can't find better, I return 56.
    
    # Let's re-verify the seed.
    # Seed is 56.
    # My construction is 56.
    # It's the same.
    
    # I will just return the seed logic but cleaned up.
    
    pass

# Re-implementing the seed logic correctly
def construct():
    deadline = time.time() + 120
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # 4 products of 2x2 by 2x4.
    # Each 2x2 by 2x4 is two 2x2x2 Strassen blocks.
    # Total 8 Strassen blocks.
    
    # Product 1: A_L (j=0,1) * B_L_top (j=0,1, k=0,1,2,3)
    # Block 1.1: k=0,1
    col = 0
    for s in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, s] != 0:
                    U[i*4 + j, col+s] = U_s[i*2+j, s]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, s] != 0:
                    V[(j)*8 + k, col+s] = V_s[j*2+k, s]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, s] != 0:
                    W[(k)*2 + i, col+s] += W_s[k*2+i, s]
    
    # Block 1.2: k=2,3
    col = 7
    for s in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, s] != 0:
                    U[i*4 + j, col+s] = U_s[i*2+j, s]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, s] != 0:
                    V[(j)*8 + (k+2), col+s] = V_s[j*2+k, s]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, s] != 0:
                    W[(k+2)*2 + i, col+s] += W_s[k*2+i, s]

    # Product 2: A_R (j=2,3) * B_L_bot (j=2,3, k=0,1,2,3)
    # Block 2.1: k=0,1
    col = 14
    for s in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, s] != 0:
                    U[i*4 + (j+2), col+s] = U_s[i*2+j, s]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, s] != 0:
                    V[(j+2)*8 + k, col+s] = V_s[j*2+k, s]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, s] != 0:
                    W[(k)*2 + i, col+s] += W_s[k*2+i, s]

    # Block 2.2: k=2,3
    col = 21
    for s in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, s] != 0:
                    U[i*4 + (j+2), col+s] = U_s[i*2+j, s]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, s] != 0:
                    V[(j+2)*8 + (k+2), col+s] = V_s[j*2+k, s]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, s] != 0:
                    W[(k+2)*2 + i, col+s] += W_s[k*2+i, s]

    # Product 3: A_L (j=0,1) * B_R_top (j=0,1, k=4,5,6,7)
    # Block 3.1: k=4,5
    col = 28
    for s in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, s] != 0:
                    U[i*4 + j, col+s] = U_s[i*2+j, s]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, s] != 0:
                    V[(j)*8 + (k+4), col+s] = V_s[j*2+k, s]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, s] != 0:
                    W[(k+4)*2 + i, col+s] += W_s[k*2+i, s]

    # Block 3.2: k=6,7
    col = 35
    for s in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, s] != 0:
                    U[i*4 + j, col+s] = U_s[i*2+j, s]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, s] != 0:
                    V[(j)*8 + (k+6), col+s] = V_s[j*2+k, s]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, s] != 0:
                    W[(k+6)*2 + i, col+s] += W_s[k*2+i, s]

    # Product 4: A_R (j=2,3) * B_R_bot (j=2,3, k=4,5,6,7)
    # Block 4.1: k=4,5
    col = 42
    for s in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, s] != 0:
                    U[i*4 + (j+2), col+s] = U_s[i*2+j, s]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, s] != 0:
                    V[(j+2)*8 + (k+4), col+s] = V_s[j*2+k, s]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, s] != 0:
                    W[(k+4)*2 + i, col+s] += W_s[k*2+i, s]

    # Block 4.2: k=6,7
    col = 49
    for s in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, s] != 0:
                    U[i*4 + (j+2), col+s] = U_s[i*2+j, s]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, s] != 0:
                    V[(j+2)*8 + (k+6), col+s] = V_s[j*2+k, s]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, s] != 0:
                    W[(k+6)*2 + i, col+s] += W_s[k*2+i, s]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

