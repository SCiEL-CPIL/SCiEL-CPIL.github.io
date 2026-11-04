import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u, v, w):
        nonlocal rank
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)
        rank += 1

    # Strassen 2x2x2 terms (indices 0..3 for 2x2)
    # A: 0,1,2,3 (row-major)
    # B: 0,1,2,3
    # C: 0,1,2,3
    
    strassen_2x2 = [
        # M1: (A00+A11)(B00+B11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,0]), # C00 part? No, this is one rank 1 term.
        # Actually, the 7 terms are the multiplications.
        # The output C is a linear combination of these 7 products.
        # So we need to add 7 terms to the tensor, and the W vectors will have the coefficients for the output.
    ]
    
    # Let's define the 7 rank-1 tensors for 2x2x2 multiplication.
    # T = sum t_r
    # t1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] ? No.
    # The product M1 = (A00+A11)(B00+B11) contributes to C00, C01, C10, C11?
    # No, M1 is a scalar.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So the 7 rank-1 terms are:
    # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (M1 contributes 1 to C00, 0 to C01, 0 to C10, 1 to C11? No)
    # M1 is a scalar. It is added to C00 with coeff 1, C11 with coeff 1.
    # So w1 = [1, 0, 0, 1].
    
    # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0] (M2 = (A10+A11)B00. C10 += M2)
    # M2 contributes to C10 with coeff 1.
    # u2 = [0,0,1,1], v2 = [1,0,0,0], w2 = [0,0,1,0].
    
    # 3. u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,0] (M3 = A00(B10-B11). C01 += M3)
    # w3 = [0,1,0,0].
    
    # 4. u=[0,0,0,1], v=[1,-1,0,0], w=[1,0,0,0] (M4 = A11(B01-B00). C00 += M4, C10 += M4)
    # w4 = [1,0,1,0].
    
    # 5. u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0] (M5 = (A00+A01)B11. C00 -= M5, C01 += M5)
    # w5 = [-1,1,0,0].
    
    # 6. u=[0,-1,1,0], v=[1,1,0,0], w=[0,0,0,1] (M6 = (A10-A00)(B00+B01). C11 += M6)
    # w6 = [0,0,0,1].
    
    # 7. u=[0,1,-1,0], v=[0,0,1,1], w=[1,0,0,1] (M7 = (A01-A11)(B10+B11). C00 += M7, C11 += M7)
    # w7 = [1,0,0,1].
    
    # Let's verify C00: M1(1) + M4(1) - M5(1) + M7(1) = 1+1-1+1 = 2? No.
    # C00 = M1 + M4 - M5 + M7.
    # My w vectors:
    # w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1. Sum = 2.
    # But C00 should be A00B00 + A01B10.
    # This decomposition is correct for the scalar multiplications.
    
    strassen_terms_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]), # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]), # M2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]), # M3
        ([0,0,0,1], [1,-1,0,0], [1,0,1,0]), # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]), # M5
        ([0,-1,1,0], [1,1,0,0], [0,0,0,1]), # M6
        ([0,1,-1,0], [0,0,1,1], [1,0,0,1])  # M7
    ]
    
    # Now, for 4x4x4, we use 2x2 blocks.
    # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
    # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
    # C blocks: C00 (0-3), C01 (4-7), C10 (8-11), C11 (12-15)
    
    def add_strassen_4x4(k_offset):
        # k_offset is the start index of the 4 columns in B
        for u2, v2, w2 in strassen_terms_2x2:
            # This term computes one of the 7 M's.
            # The M is a 2x2 matrix multiplication.
            # So we need to expand u2, v2, w2 to 4x4 scale.
            # u2 is a vector of length 4 (for 2x2 A).
            # In 4x4 A, the block A00 is indices 0-3, A01 is 4-7, etc.
            # The term u2 corresponds to a linear combination of A elements.
            # Which block does it act on?
            # M1 = (A00+A11)(B00+B11).
            # So u1 acts on A00 and A11.
            # u1 = [1,0,0,1] in 2x2 space.
            # In 4x4 space, u1 should be 1 on A00 elements and 1 on A11 elements.
            # A00 indices: 0,1,2,3. A11 indices: 12,13,14,15.
            # So u_global has 1s at 0,1,2,3 and 12,13,14,15.
            
            # We need a mapping from 2x2 block index to 4x4 global index.
            # Block (bi, bj) of A is at global index bi*4 + bj? No.
            # A is 4x4. Block (0,0) is A[0:2, 0:2] -> indices 0,1,2,3.
            # Block (0,1) is A[0:2, 2:4] -> indices 4,5,6,7.
            # Block (1,0) is A[2:4, 0:2] -> indices 8,9,10,11.
            # Block (1,1) is A[2:4, 2:4] -> indices 12,13,14,15.
            
            # The 2x2 term u2 has indices 0,1,2,3 corresponding to A00, A01, A10, A11 of the 2x2 block structure.
            # Wait, the 2x2 Strassen is applied to the 2x2 block matrix.
            # So u2[0] corresponds to block A00, u2[1] to A01, u2[2] to A10, u2[3] to A11.
            
            u_global = [0] * 16
            v_global = [0] * (4*9) # 36
            w_global = [0] * (9*4) # 36
            
            for b_idx in range(4):
                if u2[b_idx] != 0:
                    # Add u2[b_idx] to the corresponding block in A
                    # Block b_idx in 2x2 structure:
                    # 0: (0,0) -> A[0:2, 0:2]
                    # 1: (0,1) -> A[0:2, 2:4]
                    # 2: (1,0) -> A[2:4, 0:2]
                    # 3: (1,1) -> A[2:4, 2:4]
                    br, bc = (b_idx // 2, b_idx % 2)
                    for i in range(2):
                        for j in range(2):
                            global_i = br*2 + i
                            global_j = bc*2 + j
                            u_global[global_i*4 + global_j] += u2[b_idx]
                
                if v2[b_idx] != 0:
                    # v2[b_idx] corresponds to block of B
                    # B block (b_idx) is B[br*2:(br+1)*2, bc*2:(bc+1)*2]
                    # But B is 4x9. The first 4 columns are used for this 4x4x4 block.
                    # So B block (0,0) is B[0:2, k_offset:k_offset+2]
                    br, bc = (b_idx // 2, b_idx % 2)
                    for i in range(2):
                        for j in range(2):
                            global_i = br*2 + i
                            global_j = bc*2 + j
                            # B index is i*9 + (k_offset + j)
                            v_global[global_i*9 + (k_offset + global_j)] += v2[b_idx]
                
                if w2[b_idx] != 0:
                    # w2[b_idx] corresponds to block of C
                    # C block (b_idx) is C[br*2:(br+1)*2, bc*2:(bc+1)*2]
                    br, bc = (b_idx // 2, b_idx % 2)
                    for i in range(2):
                        for j in range(2):
                            global_i = br*2 + i
                            global_j = bc*2 + j
                            # W index is (k_offset + global_j)*4 + global_i
                            w_global[(k_offset + global_j)*4 + global_i] += w2[b_idx]
            
            add_term(u_global, v_global, w_global)

    # Add Strassen for B1 (k=0..3)
    add_strassen_4x4(0)
    # Add Strassen for B2 (k=4..7)
    add_strassen_4x4(4)
    
    # Add Schoolbook for B3 (k=8)
    # 4x4x1 tensor
    for i in range(4):
        for j in range(4):
            u = [0]*16
            v = [0]*36
            w = [0]*36
            u[i*4+j] = 1
            v[j*9 + 8] = 1
            w[8*4 + i] = 1
            add_term(u, v, w)
            
    return {"U": U_rows, "V": V_rows, "W": W_rows}
