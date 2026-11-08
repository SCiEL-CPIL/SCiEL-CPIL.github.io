import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 6
    nm = n * m # 8
    mp = m * p # 24
    pn = p * n # 12
    
    # We will construct U, V, W of shape (nm, 42), (mp, 42), (pn, 42)
    rank = 42
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We define the local vectors for A (2x2), B (2x2), C (2x2) for each M
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # a_idx = i*2 + j
    # b_idx = j*2 + k
    # c_idx = k*2 + i
    
    # M1: a=(1,0,1,0) -> [1,0,1,0] for a00,a01,a10,a11?
    # Let's list a vectors (size 4: a00, a01, a10, a11)
    # b vectors (size 4: b00, b01, b10, b11)
    # c vectors (size 4: c00, c01, c10, c11)
    
    strassen_terms = [
        # (a_vec, b_vec, c_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # M1 contributes to c00? 
        # Wait, the c vector in the tensor decomposition represents the coefficient of C.
        # The tensor T is sum u_r v_r w_r.
        # w_r is the vector in C space.
        # For M1, the contribution to C is:
        # c00 += 1, c01 += 0, c10 += 0, c11 += 1? 
        # Let's check Strassen formulas again.
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # So for M1:
        # a_vec = [1, 0, 0, 1] (a00 + a11)
        # b_vec = [1, 0, 0, 1] (b00 + b11)
        # c_vec = [1, 0, 0, 1] (c00 + c11)
        
        # For M2:
        # a_vec = [0, 0, 1, 1] (a10 + a11)
        # b_vec = [1, 0, 0, 0] (b00)
        # c_vec = [0, 0, 1, -1] (c10 - c11)
        
        # For M3:
        # a_vec = [1, 0, 0, 0] (a00)
        # b_vec = [0, 1, 0, -1] (b01 - b11)
        # c_vec = [0, 1, 0, 1] (c01 + c11)
        
        # For M4:
        # a_vec = [0, 0, 0, 1] (a11)
        # b_vec = [0, 0, 1, -1] (b10 - b00)
        # c_vec = [1, 0, 1, 0] (c00 + c10)
        
        # For M5:
        # a_vec = [1, 1, 0, 0] (a00 + a01)
        # b_vec = [0, 0, 0, 1] (b11)
        # c_vec = [-1, 1, 0, 0] (-c00 + c01)
        
        # For M6:
        # a_vec = [0, 0, 1, -1] (a10 - a00)
        # b_vec = [1, 1, 0, 0] (b00 + b01)
        # c_vec = [0, 0, 0, 1] (c11)
        
        # For M7:
        # a_vec = [0, 1, 0, -1] (a01 - a11)
        # b_vec = [0, 0, 1, 1] (b10 + b11)
        # c_vec = [1, 0, 0, 0] (c00)
    ]
    
    # Let's double check M6 c_vec.
    # c11 = M1 - M2 + M3 + M6.
    # So M6 contributes +1 to c11.
    # c_vec for M6 should be [0, 0, 0, 1]. Correct.
    
    # Let's double check M7 c_vec.
    # c00 = M1 + M4 - M5 + M7.
    # So M7 contributes +1 to c00.
    # c_vec for M7 should be [1, 0, 0, 0]. Correct.
    
    # Let's double check M5 c_vec.
    # c00 has -M5. c01 has +M5.
    # c_vec for M5 should be [-1, 1, 0, 0]. Correct.
    
    # Let's double check M4 c_vec.
    # c00 has +M4. c10 has +M4.
    # c_vec for M4 should be [1, 0, 1, 0]. Correct.
    
    # Let's double check M3 c_vec.
    # c01 has +M3. c11 has +M3.
    # c_vec for M3 should be [0, 1, 0, 1]. Correct.
    
    # Let's double check M2 c_vec.
    # c10 has +M2. c11 has -M2.
    # c_vec for M2 should be [0, 0, 1, -1]. Correct.
    
    # Let's double check M1 c_vec.
    # c00 has +M1. c11 has +M1.
    # c_vec for M1 should be [1, 0, 0, 1]. Correct.

    current_rank = 0
    
    # Define the 6 blocks
    # Block 0: A_left (cols 0,1), B_top (rows 0,1), C_cols (0,1)
    # Block 1: A_left (cols 0,1), B_top (rows 0,1), C_cols (2,3)
    # Block 2: A_left (cols 0,1), B_top (rows 0,1), C_cols (4,5)
    # Block 3: A_right (cols 2,3), B_bottom (rows 2,3), C_cols (0,1)
    # Block 4: A_right (cols 2,3), B_bottom (rows 2,3), C_cols (2,3)
    # Block 5: A_right (cols 2,3), B_bottom (rows 2,3), C_cols (4,5)
    
    blocks = [
        (0, 0, 0), # i_offset, j_offset, k_offset
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]
    
    for i_off, j_off, k_off in blocks:
        for a_vec, b_vec, c_vec in strassen_terms:
            # a_vec corresponds to local A indices (0,1,2,3) -> (0,0), (0,1), (1,0), (1,1)
            # b_vec corresponds to local B indices (0,1,2,3) -> (0,0), (0,1), (1,0), (1,1)
            # c_vec corresponds to local C indices (0,1,2,3) -> (0,0), (0,1), (1,0), (1,1)
            
            # Map local A indices to global A indices
            # Local (li, lj) -> Global (li + i_off, lj + j_off)
            # Global A index a = (li + i_off) * 4 + (lj + j_off)
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    val = a_vec[local_idx]
                    if val != 0:
                        global_a = (li + i_off) * 4 + (lj + j_off)
                        U[global_a, current_rank] += val
            
            # Map local B indices to global B indices
            # Local (lj, lk) -> Global (lj + j_off, lk + k_off)
            # Global B index b = (lj + j_off) * 6 + (lk + k_off)
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    val = b_vec[local_idx]
                    if val != 0:
                        global_b = (lj + j_off) * 6 + (lk + k_off)
                        V[global_b, current_rank] += val
            
            # Map local C indices to global C indices
            # Local (lk, li) -> Global (lk + k_off, li + i_off)
            # Global C index c = (lk + k_off) * 2 + (li + i_off)
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    val = c_vec[local_idx]
                    if val != 0:
                        global_c = (lk + k_off) * 2 + (li + i_off)
                        W[global_c, current_rank] += val
            
            current_rank += 1
            
            if time.time() > deadline:
                # If we run out of time, return what we have (should not happen for 42 steps)
                pass
    
    # Verify
    # T[a,b,c] should be sum U[a,r] V[b,r] W[c,r]
    # We can do a quick check
    # T is sparse.
    
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
