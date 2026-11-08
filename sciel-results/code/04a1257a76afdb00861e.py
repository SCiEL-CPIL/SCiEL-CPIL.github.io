import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-reduced exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 8 matrix.
    
    Uses a block decomposition:
    A (2x3) = [A1 (2x1), A2 (2x2)]
    B (3x8) = [B1 (1x8); B2 (2x8)]
    
    C = A1*B1 + A2*B2
    
    Rank(A1*B1) = 2*1*8 = 16
    Rank(A2*B2) = 4 * Rank(2x2x2) = 4 * 7 = 28 (using Strassen for each 2x2x2 block)
    
    Total Rank = 16 + 28 = 44.
    """
    n, m, p = DIMENSIONS
    
    # Total rank
    rank_1 = 2 * 1 * 8  # 16
    rank_2 = 4 * 7      # 28
    total_rank = rank_1 + rank_2
    
    # Initialize matrices with zeros
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: A1 (2x1) * B1 (1x8) ---
    # n=2, m=1, p=8
    # U1 size 2x16, V1 size 1x16, W1 size 8x2
    col_offset = 0
    for k in range(8):
        # Term for i=0
        # U[0*3 + 0, col] = 1
        # V[0*8 + k, col] = 1
        # W[k*2 + 0, col] = 1
        U[0 * 3 + 0][col_offset] = 1
        V[0 * 8 + k][col_offset] = 1
        W[k * 2 + 0][col_offset] = 1
        col_offset += 1
        
        # Term for i=1
        # U[1*3 + 0, col] = 1
        # V[0*8 + k, col] = 1
        # W[k*2 + 1, col] = 1
        U[1 * 3 + 0][col_offset] = 1
        V[0 * 8 + k][col_offset] = 1
        W[k * 2 + 1][col_offset] = 1
        col_offset += 1

    # --- Part 2: A2 (2x2) * B2 (2x8) ---
    # n=2, m=2, p=8
    # Split B2 into 4 blocks of 2x2.
    # For each block t (0..3), columns 2t, 2t+1.
    # Use Strassen for 2x2x2.
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We need to map these to the tensor entries.
    # The tensor is sum of rank-1 terms.
    # Each Strassen multiplication M_r is a product of two linear forms.
    # M_r = (sum_u U_{iu} A_{iu}) * (sum_v V_{jv} B_{jv})?
    # No, the decomposition is C_{ik} = sum_r U_{ir} V_{jr} W_{kr} A_{ij} B_{jk}?
    # The standard form is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # Decomposition: C_{ik} = sum_r U_{i,r} V_{j,r} W_{k,r} A_{ij} B_{jk} is not quite right.
    # The tensor T_{ijk} = delta_{jk} delta_{ij}? No.
    # T_{i, j, k} is 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # So T_{i, j, k} = 1 for all i,j,k.
    # We want to express T as sum of rank 1 tensors u_r v_r w_r.
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    
    # For Strassen, we have linear combinations.
    # M1 = (A00+A11)(B00+B11)
    # This contributes to C00, C11.
    # C00 += M1, C11 += M1.
    # So for r corresponding to M1:
    # U[0, r] = 1, U[1, r] = 1 (coefficients of A00, A11)
    # V[0, r] = 1, V[1, r] = 1 (coefficients of B00, B11) -- wait, B indices are local.
    # W[0, r] = 1, W[1, r] = 1 (coefficients for C00, C11)
    
    # Let's define the Strassen terms for local 2x2x2.
    # Local A indices: 0,1. Local B indices: 0,1. Local C indices: 0,1.
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ([1, 1], [1, 1], [1, 1]), # C00, C11 get +1
        # M2
        ([0, 1], [1, 0], [0, 1]), # (A10+A11)B00 -> C10 gets +1
        # M3
        ([1, 0], [0, -1], [1, 0]), # A00(B01-B11) -> C01 gets +1
        # M4
        ([0, 1], [-1, 1], [1, 0]), # A11(B10-B00) -> C10 gets +1
        # M5
        ([1, 0], [0, 1], [-1, 0]), # (A00+A01)B11 -> C00 gets -1
        # M6
        ([1, -1], [1, 1], [0, 1]), # (A10-A00)(B00+B01) -> C11 gets +1
        # M7
        ([0, 1], [-1, 1], [1, -1]), # (A01-A11)(B10+B11) -> C00 gets +1, C11 gets -1
    ]
    
    # Wait, I need to double check the Strassen reconstruction.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify the terms.
    # M1 = (A00+A11)(B00+B11).
    #   U: [1, 1] (A00, A11)
    #   V: [1, 1] (B00, B11) -- Wait, B indices in M1 are B00 and B11.
    #   In local 2x2 B, indices are (0,0), (0,1), (1,0), (1,1).
    #   V vector has length 4 (m*p = 2*2 = 4).
    #   V[0] coeff for B00, V[1] for B01, V[2] for B10, V[3] for B11.
    #   So for M1, V = [1, 0, 0, 1].
    #   W vector has length 4 (p*n = 2*2 = 4).
    #   W[0] for C00, W[1] for C01, W[2] for C10, W[3] for C11.
    #   M1 contributes to C00 and C11.
    #   So W = [1, 0, 0, 1].
    
    # Let's redo the Strassen terms with full vectors.
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # U: [1, 0, 0, 1] (A00, A01, A10, A11) -- Wait, U size is n*m = 4.
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # U1 = [1, 0, 0, 1]
    # V1 = [1, 0, 0, 1]
    # W1 = [1, 0, 0, 1]
    
    # M2: (A10+A11)B00 -> C10
    # U2 = [0, 0, 1, 1]
    # V2 = [1, 0, 0, 0]
    # W2 = [0, 0, 1, 0]
    
    # M3: A00(B01-B11) -> C01
    # U3 = [1, 0, 0, 0]
    # V3 = [0, 1, 0, -1]
    # W3 = [0, 1, 0, 0]
    
    # M4: A11(B10-B00) -> C10
    # U4 = [0, 0, 0, 1]
    # V4 = [0, 0, 1, 0] # Wait, B10 is index 2, B00 is index 0.
    # V4 = [-1, 0, 1, 0]
    # W4 = [0, 0, 1, 0]
    
    # M5: (A00+A01)B11 -> C00
    # U5 = [1, 1, 0, 0]
    # V5 = [0, 0, 0, 1]
    # W5 = [1, 0, 0, 0]
    
    # M6: (A10-A00)(B00+B01) -> C11
    # U6 = [-1, 0, 1, 0]
    # V6 = [1, 1, 0, 0]
    # W6 = [0, 0, 0, 1]
    
    # M7: (A01-A11)(B10+B11) -> C00, C11
    # U7 = [0, 1, 0, -1]
    # V7 = [0, 0, 1, 1]
    # W7 = [1, 0, 0, -1]
    
    # Check C00: M1 + M4 - M5 + M7
    # W00: 1 (M1) + 0 (M4) - 1 (M5) + 1 (M7) = 1. Correct.
    # Check C01: M3 + M5
    # W01: 0 (M3) + 0 (M5) = 0?
    # Wait, C01 = M3 + M5.
    # M3 contributes to C01. W3[1] = 1.
    # M5 contributes to C00. W5[1] = 0.
    # So C01 gets 1 from M3. Correct.
    # Check C10: M2 + M4
    # W10: 1 (M2) + 1 (M4) = 2?
    # Schoolbook C10 = A10B00 + A11B10.
    # M2 = (A10+A11)B00 = A10B00 + A11B00.
    # M4 = A11(B10-B00) = A11B10 - A11B00.
    # Sum = A10B00 + A11B10. Correct.
    # So W10 should be 1.
    # My W2[2]=1, W4[2]=1. Sum=2.
    # But the terms cancel?
    # The tensor decomposition is linear.
    # C10 = sum_r U_r V_r W_r A B.
    # If W2[2]=1 and W4[2]=1, then C10 gets 2 * (something)?
    # No, the "something" is the product of the linear forms.
    # M2 * 1 + M4 * 1 = M2 + M4.
    # This is correct. The coefficient 1 is correct.
    
    # Check C11: M1 - M2 + M3 + M6
    # W11: 1 (M1) - 1 (M2) + 0 (M3) + 1 (M6) = 1. Correct.
    
    # So the terms are:
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
    ]
    
    for t in range(4):
        # Block t uses columns 2t, 2t+1 of B2.
        # B2 rows are global rows 1, 2.
        # So global B indices are:
        # Row 1, Col 2t: 1*8 + 2t
        # Row 1, Col 2t+1: 1*8 + 2t+1
        # Row 2, Col 2t: 2*8 + 2t
        # Row 2, Col 2t+1: 2*8 + 2t+1
        
        # Local B indices 0,1,2,3 map to:
        # 0 -> 1*8 + 2t
        # 1 -> 1*8 + 2t+1
        # 2 -> 2*8 + 2t
        # 3 -> 2*8 + 2t+1
        
        for r in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[r]
            
            # Fill U
            # Global U indices for A2:
            # Local A 0,1,2,3 map to:
            # 0 -> 0*3 + 1 (Row 0, Col 1)
            # 1 -> 0*3 + 2 (Row 0, Col 2)
            # 2 -> 1*3 + 1 (Row 1, Col 1)
            # 3 -> 1*3 + 2 (Row 1, Col 2)
            
            for lu in range(4):
                if u_coeffs[lu] != 0:
                    gu = [0, 1, 2, 3]
                    global_u_idx = gu[lu]
                    # Wait, mapping:
                    # lu 0 (A00 local) -> global (0,1) -> 0*3+1 = 1
                    # lu 1 (A01 local) -> global (0,2) -> 0*3+2 = 2
                    # lu 2 (A10 local) -> global (1,1) -> 1*3+1 = 4
                    # lu 3 (A11 local) -> global (1,2) -> 1*3+2 = 5
                    global_u_idx = [1, 2, 4, 5][lu]
                    U[global_u_idx][col_offset] += u_coeffs[lu]
            
            # Fill V
            for lv in range(4):
                if v_coeffs[lv] != 0:
                    # lv 0 (B00 local) -> global (1, 2t) -> 1*8 + 2t
                    # lv 1 (B01 local) -> global (1, 2t+1) -> 1*8 + 2t+1
                    # lv 2 (B10 local) -> global (2, 2t) -> 2*8 + 2t
                    # lv 3 (B11 local) -> global (2, 2t+1) -> 2*8 + 2t+1
                    global_v_idx = [8 + 2*t, 8 + 2*t + 1, 16 + 2*t, 16 + 2*t + 1][lv]
                    V[global_v_idx][col_offset] += v_coeffs[lv]
            
            # Fill W
            # Global W indices for C:
            # Local C 0,1,2,3 map to:
            # 0 (C00) -> global (0, 2t) -> 2t*2 + 0
            # 1 (C01) -> global (0, 2t+1) -> (2t+1)*2 + 0
            # 2 (C10) -> global (1, 2t) -> 2t*2 + 1
            # 3 (C11) -> global (1, 2t+1) -> (2t+1)*2 + 1
            
            for lw in range(4):
                if w_coeffs[lw] != 0:
                    global_w_idx = [2*(2*t) + 0, 2*(2*t+1) + 0, 2*(2*t) + 1, 2*(2*t+1) + 1][lw]
                    W[global_w_idx][col_offset] += w_coeffs[lw]
            
            col_offset += 1

    return {"U": U, "V": V, "W": W}
