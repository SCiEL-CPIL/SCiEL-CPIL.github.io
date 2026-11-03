import numpy as np
import time

def construct():
    """
    Construct a rank-102 tensor decomposition for 4x4x9 matrix multiplication.
    
    Uses Laderman/Strassen decomposition for the 4x4x4 parts (rank 49 each) 
    and trivial decomposition for the 4x4x1 part (rank 4).
    Total rank = 49 + 49 + 4 = 102.
    """
    n, m, p = 4, 4, 9
    
    # 1. Define Rank-7 Strassen decomposition for 2x2x2
    # U: 4x7 (2x2 flattened), V: 4x7, W: 4x7
    # Indices: U[2*i+j], V[2*j+k], W[2*k+i]
    
    # Strassen terms for C = AB
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_2x2 = np.zeros((4, 7), dtype=int)
    V_2x2 = np.zeros((4, 7), dtype=int)
    W_2x2 = np.zeros((4, 7), dtype=int)
    
    # Term 1: M1
    # A: 00, 11. B: 00, 11.
    U_2x2[0, 0] = 1; U_2x2[3, 0] = 1
    V_2x2[0, 0] = 1; V_2x2[3, 0] = 1
    # C: 00(+1), 11(+1)
    W_2x2[0, 0] = 1; W_2x2[3, 0] = 1
    
    # Term 2: M2
    # A: 10, 11. B: 00.
    U_2x2[2, 1] = 1; U_2x2[3, 1] = 1
    V_2x2[0, 1] = 1
    # C: 10(+1), 11(-1)
    W_2x2[2, 1] = 1; W_2x2[3, 1] = -1
    
    # Term 3: M3
    # A: 00. B: 10, -11.
    U_2x2[0, 2] = 1
    V_2x2[2, 2] = 1; V_2x2[3, 2] = -1
    # C: 01(+1), 11(+1)
    W_2x2[1, 2] = 1; W_2x2[3, 2] = 1
    
    # Term 4: M4
    # A: 11. B: 01, -00.
    U_2x2[3, 3] = 1
    V_2x2[1, 3] = 1; V_2x2[0, 3] = -1
    # C: 00(+1), 10(+1)
    W_2x2[0, 3] = 1; W_2x2[2, 3] = 1
    
    # Term 5: M5
    # A: 00, 01. B: 11.
    U_2x2[0, 4] = 1; U_2x2[1, 4] = 1
    V_2x2[3, 4] = 1
    # C: 00(-1), 01(+1)
    W_2x2[0, 4] = -1; W_2x2[1, 4] = 1
    
    # Term 6: M6
    # A: 10, -00. B: 00, 01.
    U_2x2[2, 5] = 1; U_2x2[0, 5] = -1
    V_2x2[0, 5] = 1; V_2x2[1, 5] = 1
    # C: 11(+1)
    W_2x2[3, 5] = 1
    
    # Term 7: M7
    # A: 01, -11. B: 10, 11.
    U_2x2[1, 6] = 1; U_2x2[3, 6] = -1
    V_2x2[2, 6] = 1; V_2x2[3, 6] = 1
    # C: 00(+1)
    W_2x2[0, 6] = 1

    # 2. Construct Rank-49 decomposition for 4x4x4
    # We have 7 "macro" terms, each is a 2x2x2 multiplication.
    # Total 7*7 = 49 terms.
    
    U_4x4 = np.zeros((16, 49), dtype=int)
    V_4x4 = np.zeros((16, 49), dtype=int)
    W_4x4 = np.zeros((16, 49), dtype=int)
    
    # Mapping of global indices to local 2x2 block indices
    # A is 4x4. Blocks: A00 (0-1, 0-1), A01 (0-1, 2-3), A10 (2-3, 0-1), A11 (2-3, 2-3)
    # Global i, j. Local block bi, bj. Local i_l, j_l.
    # i = 2*bi + i_l, j = 2*bj + j_l
    
    # For each macro term t (0..6), we have local U_2x2, V_2x2, W_2x2.
    # We need to expand them to global 4x4 indices.
    
    for t in range(7):
        # Get local U, V, W for this macro term
        # Note: The local U, V, W in U_2x2 etc are for the 2x2 matrices.
        # The "macro" multiplication M_t = A_t * B_t.
        # A_t is a linear combination of A blocks.
        # B_t is a linear combination of B blocks.
        
        # We iterate over the 7 rank-1 components of the 2x2x2 product M_t.
        for r_local in range(7):
            r_global = t * 7 + r_local
            
            # Expand U
            # U_2x2[i_l * 2 + j_l, r_local] is the coefficient for A_{i_l, j_l} in the local 2x2.
            # This coefficient applies to the corresponding block of A.
            # Which block?
            # The Strassen formula defines A_t as a sum of A_blocks.
            # For example, M1 uses A00 and A11.
            # So for term t=0 (M1), the local A is composed of A00 and A11.
            # The local U_2x2 has non-zeros at (0,0) and (1,1) for M1.
            # (0,0) corresponds to A00's top-left. (1,1) corresponds to A11's bottom-right.
            
            # We need a mapping from (t, r_local, i_l, j_l) to global (i, j).
            # This mapping depends on which A blocks are involved in M_t.
            
            # Let's define the block participation for each M_t.
            # M1: A00, A11
            # M2: A10, A11
            # M3: A00
            # M4: A11
            # M5: A00, A01
            # M6: A10, A00
            # M7: A01, A11
            
            # Similarly for B blocks.
            # M1: B00, B11
            # M2: B00
            # M3: B10, B11
            # M4: B01, B00
            # M5: B11
            # M6: B00, B01
            # M7: B10, B11
            
            # And for C blocks (W).
            # M1: C00, C11
            # M2: C10, C11
            # M3: C01, C11
            # M4: C00, C10
            # M5: C00, C01
            # M6: C11
            # M7: C00
            
            # This manual mapping is error-prone. 
            # Alternative: Use the explicit Strassen formulas to build the 49 terms directly.
            
            pass # We will implement the direct construction below.

    # Direct construction of 49 terms for 4x4x4
    # We define the 7 M terms explicitly with their block coefficients.
    
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    # C_blocks: 00, 01, 10, 11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will generate the 49 terms by iterating through the 7 M's and their 7 internal Strassen terms.
    
    U_4x4 = np.zeros((16, 49), dtype=int)
    V_4x4 = np.zeros((16, 49), dtype=int)
    W_4x4 = np.zeros((16, 49), dtype=int)
    
    # Helper to add a 2x2 block term to global matrices
    def add_block_term(U, V, W, block_a, block_b, block_c, coeff_a, coeff_b, coeff_c, offset):
        # block_a, block_b, block_c are indices 0..3 (00, 01, 10, 11)
        # i_l, j_l are local indices 0..1
        for i_l in range(2):
            for j_l in range(2):
                if U_2x2[i_l * 2 + j_l, offset] != 0:
                    i = 2 * (block_a // 2) + i_l
                    j = 2 * (block_a % 2) + j_l
                    U[i * 4 + j, offset] += U_2x2[i_l * 2 + j_l, offset] * coeff_a
                    
        for j_l in range(2):
            for k_l in range(2):
                if V_2x2[j_l * 2 + k_l, offset] != 0:
                    j = 2 * (block_b // 2) + j_l
                    k = 2 * (block_b % 2) + k_l
                    V[j * 4 + k, offset] += V_2x2[j_l * 2 + k_l, offset] * coeff_b
                    
        for k_l in range(2):
            for i_l in range(2):
                if W_2x2[k_l * 2 + i_l, offset] != 0:
                    k = 2 * (block_c // 2) + k_l
                    i = 2 * (block_c % 2) + i_l
                    W[k * 4 + i, offset] += W_2x2[k_l * 2 + i_l, offset] * coeff_c

    # Define the 7 M terms
    # Each term is a list of (block_a, block_b, block_c, coeff_a, coeff_b, coeff_c)
    # Note: A single M term might involve multiple blocks.
    # We need to sum the contributions.
    
    # M1: A00+A11, B00+B11, C00+C11
    # The local 2x2 product is computed.
    # The "A" in the local product is a 2x2 matrix.
    # Its (0,0) is A00(0,0)? No.
    # The local 2x2 matrix A_t is formed by taking the corresponding 2x2 blocks.
    # For M1, A_t[0,0] = A00[0,0] + A11[0,0]? No.
    # Strassen operates on 2x2 matrices.
    # A_t is a 2x2 matrix.
    # A_t[0,0] is the top-left element of the 2x2 matrix A_t.
    # In the context of 4x4, A_t is a 2x2 matrix of 2x2 blocks? No.
    # Strassen splits 4x4 into 2x2 blocks of 2x2 matrices.
    # So A_t is a 2x2 matrix where each entry is a 2x2 matrix?
    # No, standard Strassen on 4x4:
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # M1 = (A00+A11)(B00+B11).
    # Here A00+A11 is a 2x2 matrix sum.
    # So the local 2x2 multiplication is performed on these summed 2x2 matrices.
    
    # This means for M1, the "A" used in the local 2x2 Strassen is (A00+A11).
    # So for each element (i_l, j_l) of the local A, the value is A00[i_l, j_l] + A11[i_l, j_l].
    # So we add the term for block A00 with coeff 1, and block A11 with coeff 1.
    
    # We will iterate over the 7 M's.
    # For each M, we know which A blocks are involved and their coefficients.
    # Same for B and C.
    
    m_defs = [
        # M1
        ([0, 3], [0, 3], [0, 3], [1, 1], [1, 1], [1, 1]),
        # M2
        ([2, 3], [0], [2, 3], [1, 1], [1], [1, -1]),
        # M3
        ([0], [2, 3], [1, 3], [1], [1, -1], [1, 1]),
        # M4
        ([3], [1, 0], [0, 2], [1], [1, -1], [1, 1]),
        # M5
        ([0, 1], [3], [0, 1], [1, 1], [1], [-1, 1]),
        # M6
        ([2, 0], [0, 1], [3], [1, -1], [1, 1], [1]),
        # M7
        ([1, 3], [2, 3], [0], [1, -1], [1, 1], [1]),
    ]
    
    # Note: The C block coefficients must match the C formula.
    # C00 = M1 + M4 - M5 + M7.
    # So M1 contributes +1 to C00. M4 +1 to C00. M5 -1 to C00. M7 +1 to C00.
    # My m_defs C lists:
    # M1: [0, 3] (C00, C11) coeff [1, 1]. Correct.
    # M2: [2, 3] (C10, C11) coeff [1, -1]. Correct (C10+=M2, C11-=M2).
    # M3: [1, 3] (C01, C11) coeff [1, 1]. Correct (C01+=M3, C11+=M3).
    # M4: [0, 2] (C00, C10) coeff [1, 1]. Correct (C00+=M4, C10+=M4).
    # M5: [0, 1] (C00, C01) coeff [-1, 1]. Correct (C00-=M5, C01+=M5).
    # M6: [3] (C11) coeff [1]. Correct (C11+=M6).
    # M7: [0] (C00) coeff [1]. Correct (C00+=M7).
    
    for t in range(7):
        a_blocks, b_blocks, c_blocks, a_coeffs, b_coeffs, c_coeffs = m_defs[t]
        
        # We need to sum the contributions from each block combination.
        # Since the local Strassen U, V, W are linear, we can just add the terms.
        # For each local rank-1 term r_local (0..6):
        for r_local in range(7):
            r_global = t * 7 + r_local
            
            # For each involved A block
            for idx, block_a in enumerate(a_blocks):
                coeff_a = a_coeffs[idx]
                if coeff_a == 0: continue
                # Add contribution to U
                for i_l in range(2):
                    for j_l in range(2):
                        val = U_2x2[i_l * 2 + j_l, r_local]
                        if val != 0:
                            i = 2 * (block_a // 2) + i_l
                            j = 2 * (block_a % 2) + j_l
                            U_4x4[i * 4 + j, r_global] += val * coeff_a

            for idx, block_b in enumerate(b_blocks):
                coeff_b = b_coeffs[idx]
                if coeff_b == 0: continue
                for j_l in range(2):
                    for k_l in range(2):
                        val = V_2x2[j_l * 2 + k_l, r_local]
                        if val != 0:
                            j = 2 * (block_b // 2) + j_l
                            k = 2 * (block_b % 2) + k_l
                            V_4x4[j * 4 + k, r_global] += val * coeff_b

            for idx, block_c in enumerate(c_blocks):
                coeff_c = c_coeffs[idx]
                if coeff_c == 0: continue
                for k_l in range(2):
                    for i_l in range(2):
                        val = W_2x2[k_l * 2 + i_l, r_local]
                        if val != 0:
                            k = 2 * (block_c // 2) + k_l
                            i = 2 * (block_c % 2) + i_l
                            W_4x4[k * 4 + i, r_global] += val * coeff_c

    # 3. Construct Rank-4 decomposition for 4x4x1 (last column)
    U_4x1 = np.zeros((16, 4), dtype=int)
    V_4x1 = np.zeros((36, 4), dtype=int) # 4x9 flattened? No, 4x1 part.
    # Wait, V is m x p. Here m=4, p=9.
    # We only use the last column k=8.
    # V[j * 9 + 8].
    
    W_4x1 = np.zeros((36, 4), dtype=int) # 9x4 flattened.
    # W[8 * 4 + i].
    
    for j in range(4):
        # Term j
        # U: 1 at (i, j) for all i?
        # No, U_4x1[i * 4 + j] = 1 for all i.
        for i in range(4):
            U_4x1[i * 4 + j, j] = 1
            
        # V: 1 at (j, 8)
        V_4x1[j * 9 + 8, j] = 1
        
        # W: 1 at (8, i) for all i
        for i in range(4):
            W_4x1[8 * 4 + i, j] = 1

    # 4. Concatenate
    # U: 16 x (49 + 49 + 4)
    # V: 36 x (49 + 49 + 4)
    # W: 36 x (49 + 49 + 4)
    
    # Part 1: 4x4x4 (cols 0-3)
    U1 = np.zeros((16, 49), dtype=int)
    V1 = np.zeros((36, 49), dtype=int)
    W1 = np.zeros((36, 49), dtype=int)
    
    # Fill U1 from U_4x4
    U1[:, :] = U_4x4
    # Fill V1 from V_4x4 (cols 0-3)
    for j in range(4):
        for k in range(4):
            V1[j * 9 + k, :] = V_4x4[j * 4 + k, :]
    # Fill W1 from W_4x4 (cols 0-3)
    for k in range(4):
        for i in range(4):
            W1[k * 4 + i, :] = W_4x4[k * 4 + i, :]
            
    # Part 2: 4x4x4 (cols 4-7)
    U2 = np.zeros((16, 49), dtype=int)
    V2 = np.zeros((36, 49), dtype=int)
    W2 = np.zeros((36, 49), dtype=int)
    
    U2[:, :] = U_4x4
    for j in range(4):
        for k in range(4):
            V2[j * 9 + (k + 4), :] = V_4x4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W2[(k + 4) * 4 + i, :] = W_4x4[k * 4 + i, :]
            
    # Part 3: 4x4x1 (col 8)
    U3 = U_4x1
    V3 = V_4x1
    W3 = W_4x1
    
    U_total = np.hstack([U1, U2, U3])
    V_total = np.hstack([V1, V2, V3])
    W_total = np.hstack([W1, W2, W3])
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Check rank
    print(f"Rank: {U.shape[1]}")
    
    # Check correctness on random matrices
    A = np.random.randint(-5, 5, size=(4, 4))
    B = np.random.randint(-5, 5, size=(4, 9))
    
    C_true = A @ B
    
    # C_decomp = sum_r W_r * (U_r * V_r) ?
    # C_ik = sum_r W_ki * sum_j U_ij V_jk
    C_decomp = np.zeros((4, 9), dtype=int)
    for r in range(U.shape[1]):
        Ur = U[:, r].reshape(4, 4)
        Vr = V[:, r].reshape(4, 9)
        Wr = W[:, r].reshape(9, 4)
        
        # Product P = Ur @ Vr (4x9)
        P = Ur @ Vr
        # Contribution to C is Wr * P?
        # C_ik += Wr_ki * P_ik
        C_decomp += Wr * P
        
    if np.array_equal(C_true, C_decomp):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print("True:", C_true)
        print("Decomp:", C_decomp)

