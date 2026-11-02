import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 via Strassen 2x2 on 3x3 blocks
    # M3 schoolbook rank is 27
    # 7 terms * 27 = 189
    
    # 1. Define M3 schoolbook decomposition
    # T3 = sum_{i,j,k} A[i,k] B[k,j] C[j,i] (Trace convention)
    # u_{i,k}, v_{k,j}, w_{j,i}
    # Note: W index in task is k*n+i? 
    # Task: W[k*n+i, r]. 
    # My derivation: W corresponds to C.
    # If C is 3x3, local index c_r, c_c.
    # Task global W index for 6x6: k*6+i.
    
    # Let's generate the 27 components for 3x3
    # u: row i, col k
    # v: row k, col j
    # w: row j, col i? 
    # Trace: sum A[i,k] B[k,j] C[j,i]
    # So W must pick C[j,i].
    # In 3x3, W index j*3+i.
    
    m3_rank = 27
    u3 = np.zeros((9, m3_rank))
    v3 = np.zeros((9, m3_rank))
    w3 = np.zeros((9, m3_rank))
    
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                # A[i,k]
                u3[i*3+k, col] = 1
                # B[k,j]
                v3[k*3+j, col] = 1
                # C[j,i]
                w3[j*3+i, col] = 1
                col += 1
                
    # 2. Define Strassen Linear Maps
    # Blocks 00, 01, 10, 11
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # Map L_A: A (36x1) -> X (9x1)
    # X is 3x3. 
    # We need (L_A)^* : u (9x1) -> A_vec (36x1)
    # L_A(A) = sum_{ab} x_ab A_ab
    # (L_A)^*(u) = sum_{ab} x_ab u_ab (where u_ab is the block of u)
    
    # Strassen Terms:
    # M1 = (A10 - A11)(B11 - B01)
    # M2 = (A00 + A10)B11
    # M3 = A00(B01 - B11)
    # M4 = A11(B00 - B01)
    # M5 = (A00 + A11)B00
    # M6 = (A01 - A00)(B00 + B10)
    # M7 = (A10 - A00)(B01 + B11)
    
    # Coefficients for A blocks [00, 01, 10, 11]
    # Coefficients for B blocks [00, 01, 10, 11]
    
    strassen_coeffs = [
        # t, a_coeffs [00,01,10,11], b_coeffs [00,01,10,11]
        (0, [0, 0, 1, -1], [0, -1, 0, 1]), # M1
        (1, [1, 0, 1, 0], [0, 0, 0, 1]),   # M2
        (2, [1, 0, 0, 0], [0, 1, 0, -1]),  # M3
        (3, [0, 0, 0, 1], [1, -1, 0, 0]),  # M4
        (4, [1, 0, 0, 1], [1, 0, 0, 0]),   # M5
        (5, [-1, 1, 0, 0], [1, 0, 1, 0]),  # M6
        (6, [-1, 0, 1, 0], [0, 1, 0, 1]),  # M7
    ]
    
    total_rank = 7 * m3_rank
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Helper to map 3x3 block coeffs to 6x6 vector
    # Block ab is at rows 3a..3a+2, cols 3b..3b+2
    def expand_block_coeffs(coeffs, vec3x3):
        # coeffs: [c00, c01, c10, c11]
        # vec3x3: 9x1
        # returns 36x1
        out = np.zeros(36)
        for idx, (a, b) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            c = coeffs[idx]
            if c != 0:
                # Block ab rows 3a to 3a+2, cols 3b to 3b+2
                for i in range(3):
                    for j in range(3):
                        row = 3*a + i
                        col = 3*b + j
                        out[row*6 + col] = c * vec3x3[i*3 + j]
        return out
    
    col_offset = 0
    for t, a_coeffs, b_coeffs in strassen_coeffs:
        for r in range(m3_rank):
            col = col_offset + r
            
            u_r = u3[:, r]
            v_r = v3[:, r]
            w_r = w3[:, r]
            
            U[:, col] = expand_block_coeffs(a_coeffs, u_r)
            V[:, col] = expand_block_coeffs(b_coeffs, v_r)
            W[:, col] = expand_block_coeffs([1,1,1,1], w_r) # W is same for all blocks?
            # Wait. W is C.
            # In Strassen, C blocks are linear combos of M_t.
            # But here we are decomposing the TRACE.
            # Trace term for M_t is Tr(C_block M_t).
            # C_block depends on t.
            # My W expansion above assumed W picks C directly.
            # But C is the OUTPUT.
            # The tensor is T(A,B,C).
            # For term t, we have Tr(C M_t).
            # C is linear in C_blocks.
            # We need to map w_r (which is in C 3x3 space) to global C space.
            # But C_blocks are determined by C.
            # Actually, W in the decomposition should correspond to the C part of the trace.
            # Tr(C M_t) = sum_{pq} Tr(C_pq M_t_pq)?
            # No. M_t is 3x3. C is 6x6.
            # The Strassen algorithm computes C_blocks from M_t.
            # So C_pq = sum_t w_pq^t M_t.
            # Tr(C M^T) = sum_pq Tr(C_pq (M^T)_qp) = sum_pq Tr(C_pq (M_qp)^T).
            # This couples C_pq with M_qp.
            # This means W is NOT just expanding w_r to all blocks.
            # W must be expanded according to the Strassen reconstruction coefficients.
            
            # Let's fix W.
            # Reconstruction:
            # C00 = M5 + M4 - M2 + M6
            # C01 = M1 + M2
            # C10 = M3 + M5
            # C11 = M1 - M3 + M4 + M7
            
            # For term t, M_t contributes to C_pq with coefficient w_pq^t.
            # Trace term: Tr(C M_t) ? No.
            # The tensor is sum_t Tr(C M_t^T) ?
            # No, the decomposition is for the specific M_t.
            # T = sum_t T_t.
            # T_t(A,B,C) = Tr(C M_t^T) ?
            # No. T(A,B,C) = Tr(C (AB)^T)?
            # Standard: Tr(C AB).
            # AB = sum_t w_qp^t M_t.
            # Tr(C AB) = sum_qp Tr(C_qp (AB)_pq) = sum_qp Tr(C_qp sum_t w_pq^t M_t).
            # = sum_t sum_qp w_pq^t Tr(C_qp M_t).
            # So for term t, the C-part is sum_qp w_pq^t C_qp.
            # So W vector for term t is sum_qp w_pq^t (expansion of w_r to block qp).
            
            w_global = np.zeros(36)
            # Coeffs for C blocks [00, 01, 10, 11] for this term t
            c_coeffs = [0, 0, 0, 0]
            if t == 0: # M1
                c_coeffs = [0, 1, 0, 1] # C01 += M1, C11 += M1
            elif t == 1: # M2
                c_coeffs = [1, 1, 0, 0] # C00 -= M2? No C00 = M5+M4-M2+M6.
                # Wait, C00 = M5 + M4 - M2 + M6.
                # So coeff for M2 in C00 is -1.
                # Coeff for M2 in C01 is 1.
                c_coeffs = [-1, 1, 0, 0]
            elif t == 2: # M3
                c_coeffs = [0, 0, 1, -1] # C10 += M3, C11 -= M3
            elif t == 3: # M4
                c_coeffs = [1, 0, 0, 1] # C00 += M4, C11 += M4
            elif t == 4: # M5
                c_coeffs = [1, 0, 1, 0] # C00 += M5, C10 += M5
            elif t == 5: # M6
                c_coeffs = [1, 0, 0, 0] # C00 += M6
            elif t == 6: # M7
                c_coeffs = [0, 0, 0, 1] # C11 += M7
            
            # Now expand w_r (3x3) to global W using c_coeffs
            for idx, (p, q) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
                c = c_coeffs[idx]
                if c != 0:
                    for i in range(3):
                        for j in range(3):
                            row = 3*p + i
                            col = 3*q + j
                            w_global[row*6 + col] += c * w_r[i*3 + j]
            
            W[:, col] = w_global
            
            col_offset += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
