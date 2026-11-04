import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    
    # Base Strassen factors for 2x2x2 (Rank 7)
    # U, V, W shapes (4, 7)
    # Indices: row = i*2 + j
    U_base = np.zeros((4, 7))
    V_base = np.zeros((4, 7))
    W_base = np.zeros((4, 7))
    
    # Strassen Terms:
    # 0: M1 = (A11+A22)(B12+B22)
    # 1: M2 = (A21+A22)B11
    # 2: M3 = A11(B11-B21)
    # 3: M4 = A22(B22-B12)
    # 4: M5 = (A11+A12)B22
    # 5: M6 = (A21-A11)(B11+B12)
    # 6: M7 = (A12-A22)(B21+B22)
    
    # Coefficients for A blocks (11, 12, 21, 22) for each term
    # A11 is block (0,0), A12 is (0,1), A21 is (1,0), A22 is (1,1)
    A_coeffs = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 0, 1, 0],# M6: A21 - A11
        [0, 1, 0, -1] # M7: A12 - A22
    ]
    
    # Coefficients for B blocks (11, 12, 21, 22) for each term
    B_coeffs = [
        [0, 1, 0, 1], # M1: B12 + B22
        [1, 0, 0, 0], # M2: B11
        [1, 0, -1, 0],# M3: B11 - B21
        [0, -1, 0, 1],# M4: B22 - B12
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11 + B12
        [0, 0, 1, 1]  # M7: B21 + B22
    ]
    
    # Base U, V, W for 2x2x2
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: C11 += M1, C22 += M1
    # M2: C21 += M2, C22 -= M2
    # M3: C12 += M3, C22 += M3
    # M4: C11 += M4, C21 += M4
    # M5: C11 -= M5, C12 += M5
    # M6: C22 += M6
    # M7: C11 += M7
    
    # W_base[output_idx, term_idx]
    # C11 is idx 0, C12 is idx 1, C21 is idx 2, C22 is idx 3
    W_base[0, 0] = 1  # M1 to C11
    W_base[3, 0] = 1  # M1 to C22
    
    W_base[2, 1] = 1  # M2 to C21
    W_base[3, 1] = -1 # M2 to C22
    
    W_base[1, 2] = 1  # M3 to C12
    W_base[3, 2] = 1  # M3 to C22
    
    W_base[0, 3] = 1  # M4 to C11
    W_base[2, 3] = 1  # M4 to C21
    
    W_base[0, 4] = -1 # M5 to C11
    W_base[1, 4] = 1  # M5 to C12
    
    W_base[3, 5] = 1  # M6 to C22
    
    W_base[0, 6] = 1  # M7 to C11
    
    # U_base and V_base are determined by the linear combinations of blocks
    # For each term k (0..6), the A-side is sum(A_coeffs[k][b] * Block_b)
    # The B-side is sum(B_coeffs[k][b] * Block_b)
    # The sub-decomposition s (0..6) applies to the resulting 2x2 matrices.
    # U_base[local_a_idx, s] is the coefficient of A_local in sub-term s.
    # This is exactly the same as the standard 2x2 Strassen U factors.
    
    # Standard 2x2 Strassen U factors (A-side)
    # Term 0 (M1): A11 + A22 -> U[0,0]=1, U[3,0]=1
    # Term 1 (M2): A21 + A22 -> U[2,1]=1, U[3,1]=1
    # Term 2 (M3): A11 -> U[0,2]=1
    # Term 3 (M4): A22 -> U[3,3]=1
    # Term 4 (M5): A11 + A12 -> U[0,4]=1, U[1,4]=1
    # Term 5 (M6): A21 - A11 -> U[2,5]=1, U[0,5]=-1
    # Term 6 (M7): A12 - A22 -> U[1,6]=1, U[3,6]=-1
    
    U_base[0, 0] = 1; U_base[3, 0] = 1
    U_base[2, 1] = 1; U_base[3, 1] = 1
    U_base[0, 2] = 1
    U_base[3, 3] = 1
    U_base[0, 4] = 1; U_base[1, 4] = 1
    U_base[2, 5] = 1; U_base[0, 5] = -1
    U_base[1, 6] = 1; U_base[3, 6] = -1
    
    # Standard 2x2 Strassen V factors (B-side)
    # Term 0 (M1): B12 + B22 -> V[1,0]=1, V[3,0]=1
    # Term 1 (M2): B11 -> V[0,1]=1
    # Term 2 (M3): B11 - B21 -> V[0,2]=1, V[2,2]=-1
    # Term 3 (M4): B22 - B12 -> V[3,3]=1, V[1,3]=-1
    # Term 4 (M5): B22 -> V[3,4]=1
    # Term 5 (M6): B11 + B12 -> V[0,5]=1, V[1,5]=1
    # Term 6 (M7): B21 + B22 -> V[2,6]=1, V[3,6]=1
    
    V_base[1, 0] = 1; V_base[3, 0] = 1
    V_base[0, 1] = 1
    V_base[0, 2] = 1; V_base[2, 2] = -1
    V_base[3, 3] = 1; V_base[1, 3] = -1
    V_base[3, 4] = 1
    V_base[0, 5] = 1; V_base[1, 5] = 1
    V_base[2, 6] = 1; V_base[3, 6] = 1

    # Construct 4x4x4 factors (Rank 49)
    rank_strassen = 49
    U_s = np.zeros((16, rank_strassen))
    V_s = np.zeros((16, rank_strassen)) # Note: V for 4x4x4 part has m=4, p=4. Shape (16, 49)
    W_s = np.zeros((16, rank_strassen)) # Note: W for 4x4x4 part has p=4, n=4. Shape (16, 49)
    
    # Indices for 4x4:
    # U row: i*4 + l
    # V row: l*4 + j (for the 4x4 part, p=4)
    # W row: j*4 + i (for the 4x4 part, p=4, n=4)
    
    for r in range(rank_strassen):
        k = r // 7
        s = r % 7
        
        # A-side
        # U_s[global_i, global_l] = A_coeffs[k][block] * U_base[local_i, local_l, s]
        for i in range(4):
            for l in range(4):
                bi = i // 2
                bl = l // 2
                li = i % 2
                ll = l % 2
                
                coeff_block = A_coeffs[k][bi * 2 + bl]
                coeff_base = U_base[li * 2 + ll, s]
                
                U_s[i * 4 + l, r] = coeff_block * coeff_base
        
        # B-side
        # V_s[global_l, global_j] = B_coeffs[k][block] * V_base[local_l, local_j, s]
        for l in range(4):
            for j in range(4):
                bl = l // 2
                bj = j // 2
                ll = l % 2
                lj = j % 2
                
                coeff_block = B_coeffs[k][bl * 2 + bj]
                coeff_base = V_base[ll * 2 + lj, s]
                
                V_s[l * 4 + j, r] = coeff_block * coeff_base
        
        # W-side
        # W_s[global_j, global_i] = W_base[local_j, local_i, s]
        # Note: W_base indices are output (i,j) of 2x2.
        # Here output is 4x4.
        # Local output indices: li_out = i % 2, lj_out = j % 2.
        # W_base row index is li_out * 2 + lj_out.
        for j in range(4):
            for i in range(4):
                bj = j // 2
                bi = i // 2
                lj = j % 2
                li = i % 2
                
                # The Strassen term k contributes to the output blocks.
                # The coefficient for the block (bi, bj) in the output C is determined by the Strassen formula.
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                # We need a coefficient matrix for W blocks.
                
                # Let's define C_coeffs[k][block]
                # k=0 (M1): C11(1), C22(1)
                # k=1 (M2): C21(1), C22(-1)
                # k=2 (M3): C12(1), C22(1)
                # k=3 (M4): C11(1), C21(1)
                # k=4 (M5): C11(-1), C12(1)
                # k=5 (M6): C22(1)
                # k=6 (M7): C11(1)
                
                pass # We need to calculate this properly below

    # Define C_coeffs for the 7 terms
    # Blocks: 0:C11, 1:C12, 2:C21, 3:C22
    C_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Re-calculate W_s
    for r in range(rank_strassen):
        k = r // 7
        s = r % 7
        
        for j in range(4):
            for i in range(4):
                bj = j // 2
                bi = i // 2
                lj = j % 2
                li = i % 2
                
                coeff_block = C_coeffs[k][bi * 2 + bj]
                coeff_base = W_base[li * 2 + lj, s]
                
                # W row index for 4x4x4 part: j*4 + i
                W_s[j * 4 + i, r] = coeff_block * coeff_base

    # Construct Schoolbook factors for 4x4x1 (Rank 16)
    rank_school = 16
    U_sc = np.zeros((16, rank_school))
    V_sc = np.zeros((20, rank_school)) # m*p = 4*5. But we only use j=4.
    W_sc = np.zeros((20, rank_school)) # p*n = 5*4. We use j=4.
    
    for r in range(rank_school):
        l = r // 4
        i = r % 4
        
        # U: i*4 + l
        U_sc[i * 4 + l, r] = 1
        
        # V: l*5 + 4 (j=4)
        V_sc[l * 5 + 4, r] = 1
        
        # W: 4*4 + i (j=4, n=4)
        W_sc[4 * 4 + i, r] = 1

    # Combine
    U_final = np.hstack([U_s, U_sc])
    V_final = np.hstack([V_s, V_sc]) # V_s is (16, 49), V_sc is (20, 16). Need to pad V_s rows to 20.
    # V_s corresponds to j=0..3. V_sc corresponds to j=4.
    # V_final shape must be (20, 65).
    V_full = np.zeros((20, 65))
    V_full[0:16, 0:49] = V_s
    V_full[0:20, 49:65] = V_sc # V_sc is 20x16
    
    W_full = np.zeros((20, 65))
    W_full[0:16, 0:49] = W_s
    W_full[16:20, 49:65] = W_sc # W_sc is 20x16? No, W_sc is (20, 16) but only rows 16..19 are non-zero?
    # W_sc row index: 4*4 + i = 16 + i. So rows 16..19.
    # W_full rows 0..15 come from W_s.
    # W_full rows 16..19 come from W_sc.
    # W_sc shape is (20, 16). Rows 0..15 are 0. Rows 16..19 have values.
    # So W_full[:, 49:65] = W_sc is correct.

    return {'U': U_final, 'V': V_full, 'W': W_full}
