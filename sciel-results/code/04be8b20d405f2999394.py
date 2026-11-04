import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank 98 decomposition for 4x4x8
    # Split into two 4x4x4 multiplications (Left and Right)
    
    # Inner Strassen factors for 2x2x2
    # u, v, w are length 4 vectors
    inner_u = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    inner_v = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, 0, 1, -1],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    inner_w = [
        [1, 0, 0, 1], # P1 -> C00, C11
        [0, 0, 1, 1], # P2 -> C10, C11
        [0, 1, 0, 0], # P3 -> C01
        [-1, 1, 1, -1],# P4 -> -C00, C01, C10, -C11
        [1, 0, 0, 0],  # P5 -> C00
        [0, 0, 0, 1],  # P6 -> C11
        [-1, 0, 0, -1] # P7 -> -C00, -C11
    ]
    
    # Outer Strassen Linear Maps for A and B blocks
    # A blocks: A00 (0:4), A01 (4:8), A10 (8:12), A11 (12:16)
    # B blocks: B00 (0:4), B01 (4:8), B10 (8:12), B11 (12:16)
    
    # Define coefficients for outer products 1..7
    # Each outer product q defines A_q and B_q as linear combos of blocks
    # We represent the linear map as a 4x16 matrix (or sparse logic)
    
    def get_outer_maps(q):
        """
        Returns (map_A, map_B) as 4x16 numpy arrays
        map_A maps 16 A entries to 4 entries of A_q
        map_B maps 16 B entries to 4 entries of B_q
        """
        map_A = np.zeros((4, 16))
        map_B = np.zeros((4, 16))
        
        if q == 1:
            # A_q = A00 + A11, B_q = B00 + B11
            map_A[:, 0:4] = 1
            map_A[:, 12:16] = 1
            map_B[:, 0:4] = 1
            map_B[:, 12:16] = 1
        elif q == 2:
            # A_q = A10 + A11, B_q = B00
            map_A[:, 8:12] = 1
            map_A[:, 12:16] = 1
            map_B[:, 0:4] = 1
        elif q == 3:
            # A_q = A00, B_q = B01 - B11
            map_A[:, 0:4] = 1
            map_B[:, 4:8] = 1
            map_B[:, 12:16] = -1
        elif q == 4:
            # A_q = A11, B_q = B10 - B00
            map_A[:, 12:16] = 1
            map_B[:, 8:12] = 1
            map_B[:, 0:4] = -1
        elif q == 5:
            # A_q = A00 + A01, B_q = B11
            map_A[:, 0:4] = 1
            map_A[:, 4:8] = 1
            map_B[:, 12:16] = 1
        elif q == 6:
            # A_q = A10 - A00, B_q = B00 + B01
            map_A[:, 8:12] = 1
            map_A[:, 0:4] = -1
            map_B[:, 0:4] = 1
            map_B[:, 4:8] = 1
        elif q == 7:
            # A_q = A01 - A11, B_q = B10 + B11
            map_A[:, 4:8] = 1
            map_A[:, 12:16] = -1
            map_B[:, 8:12] = 1
            map_B[:, 12:16] = 1
            
        return map_A, map_B

    # Outer Coefficients for C blocks
    # C blocks: C00 (0:4), C01 (4:8), C10 (8:12), C11 (12:16)
    # Coeff[q][block] = coefficient
    outer_coeff = [
        [0, 0, 0, 0], # dummy
        [1, 0, 0, 1], # P1 -> C00, C11
        [0, 0, 1, 1], # P2 -> C10, C11
        [0, 1, 0, 0], # P3 -> C01
        [-1, 1, 1, -1],# P4 -> -C00, C01, C10, -C11
        [1, 0, 0, 0],  # P5 -> C00
        [0, 0, 0, 1],  # P6 -> C11
        [-1, 0, 0, -1] # P7 -> -C00, -C11
    ]

    U_list = []
    V_list = []
    W_list = []
    
    # We need to handle the 4x4x4 part first
    # Then duplicate for the 4x4x8 case
    # The 4x4x8 case is C_L = A B_L, C_R = A B_R
    # So we generate 49 factors for B_L/C_L and 49 for B_R/C_R
    
    # Indices for 4x4x4
    for q in range(1, 8):
        map_A, map_B = get_outer_maps(q)
        for r in range(7):
            # Global U for this factor (length 16)
            u_inner = np.array(inner_u[r])
            U_factor = map_A @ u_inner
            
            # Global V for this factor (length 16)
            v_inner = np.array(inner_v[r])
            V_factor = map_B @ v_inner
            
            # Global W for this factor (length 16)
            w_inner = np.array(inner_w[r])
            W_factor = np.zeros(16)
            
            coeff = outer_coeff[q]
            # Map the 4 inner output entries to the 16 global output entries
            # Inner indices 0,1,2,3 correspond to 2x2 block entries
            # Block 00: global 0,1,4,5
            # Block 01: global 2,3,6,7
            # Block 10: global 8,9,12,13
            # Block 11: global 10,11,14,15
            
            if coeff[0] != 0:
                # Block 00
                W_factor[0] += coeff[0] * w_inner[0]
                W_factor[1] += coeff[0] * w_inner[1]
                W_factor[4] += coeff[0] * w_inner[2]
                W_factor[5] += coeff[0] * w_inner[3]
                
            if coeff[1] != 0:
                # Block 01
                W_factor[2] += coeff[1] * w_inner[0]
                W_factor[3] += coeff[1] * w_inner[1]
                W_factor[6] += coeff[1] * w_inner[2]
                W_factor[7] += coeff[1] * w_inner[3]
                
            if coeff[2] != 0:
                # Block 10
                W_factor[8] += coeff[2] * w_inner[0]
                W_factor[9] += coeff[2] * w_inner[1]
                W_factor[12] += coeff[2] * w_inner[2]
                W_factor[13] += coeff[2] * w_inner[3]
                
            if coeff[3] != 0:
                # Block 11
                W_factor[10] += coeff[3] * w_inner[0]
                W_factor[11] += coeff[3] * w_inner[1]
                W_factor[14] += coeff[3] * w_inner[2]
                W_factor[15] += coeff[3] * w_inner[3]

            # Store for Left half (B_L, C_L)
            # U is same for both
            # V needs to be mapped to B_L (cols 0-3 of B, so rows 0-3 of B_L? No, B is 4x8. B_L is 4x4 (cols 0-3). B_R is 4x4 (cols 4-7).
            # Wait, B_L uses the same A.
            # The V factor for the left half should only have non-zeros in the first 4 columns of B (if B was 4x8 flattened row-major?).
            # B is 4x8. Flattened row-major:
            # Row 0: 0,1,2,3,4,5,6,7
            # Row 1: 8,9,10,11,12,13,14,15
            # ...
            # B_L is cols 0-3. Indices:
            # Row 0: 0-3
            # Row 1: 8-11
            # Row 2: 16-19
            # Row 3: 24-27
            # Total 32 entries.
            
            # My V_factor is length 16, corresponding to a 4x4 matrix (B_L or B_R).
            # For B_L, we place V_factor into the slots corresponding to cols 0-3.
            # For B_R, we place V_factor into the slots corresponding to cols 4-7.
            
            V_L = np.zeros(32)
            V_R = np.zeros(32)
            
            # Map 16 entries of V_factor (4x4) to 32 entries of B (4x8)
            # V_factor[i*4 + j] corresponds to B[i, j]
            for i in range(4):
                for j in range(4):
                    val = V_factor[i*4 + j]
                    V_L[i*8 + j] = val
                    V_R[i*8 + (j+4)] = val
            
            # W_L and W_R
            # C is 4x8. C_L is cols 0-3, C_R is cols 4-7.
            # W_factor is length 16 (4x4).
            W_L = np.zeros(32)
            W_R = np.zeros(32)
            
            for i in range(4):
                for j in range(4):
                    val = W_factor[i*4 + j]
                    W_L[i*8 + j] = val
                    W_R[i*8 + (j+4)] = val
            
            U_factor_list = U_factor.tolist()
            
            U_list.append(U_factor_list)
            V_list.append(V_L.tolist())
            W_list.append(W_L.tolist())
            
            U_list.append(U_factor_list)
            V_list.append(V_R.tolist())
            W_list.append(W_R.tolist())
            
    rank = len(U_list)
    U = U_list
    V = V_list
    W = W_list
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
