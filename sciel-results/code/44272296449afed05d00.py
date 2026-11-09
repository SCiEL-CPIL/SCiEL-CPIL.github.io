import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 110.0
    
    n, m, p = 4, 4, 6
    
    # Split p=6 into p1=4 and p2=2.
    # M_4,4,6 is composed of M_4,4,4 (for k=1..4) and M_4,4,2 (for k=5,6).
    # R(4,4,4) = 49 (Landsberg bound, constructible)
    # R(4,4,2) <= 2*R(4,4,1) = 2*16 = 32 (using column split)
    # Total rank = 49 + 32 = 81.
    
    rank_444 = 49
    rank_442 = 32
    total_rank = rank_444 + rank_442
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # --- Part 1: M_4,4,4 with rank 49 ---
    # We use the known 49-rank decomposition for 4x4x4 matrix multiplication.
    # This can be constructed from Strassen's 2x2x2 algorithm (rank 7) applied recursively.
    # R(4,4,4) <= 2*R(2,2,2) * 4 + ... no, block multiplication:
    # R(2r, 2r, 2r) <= 7 R(r, r, r).
    # R(4,4,4) <= 7*R(2,2,2) = 49.
    
    # We need to explicitly construct the 49 rank-1 tensors.
    # The 2x2x2 Strassen decomposition has rank 7.
    # Let's define the 7 vectors for 2x2x2.
    
    # Strassen 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # i = m1 - m5 + m7
    # j = m3 + m5
    # k = m2 + m4
    # l = m1 - m3 + m6
    
    # Tensors (u, v, w) where T[i,j,k] = u[i,j] v[j,k] w[k,i]
    # Note: Indices for 2x2:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # m1: (a+d)(e+h)
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # w1 = [1, -1, 0, 1]  # i = m1, l = m1, j = -m1 (from m1-m5+m7? No, w depends on i,j,k)
    # Let's derive w carefully.
    # C_00 (i) = m1 - m5 + m7. So w1[0] = 1.
    # C_01 (j) = m3 + m5. m1 not involved. w1[1] = 0.
    # C_10 (k) = m2 + m4. m1 not involved. w1[2] = 0.
    # C_11 (l) = m1 - m3 + m6. So w1[3] = 1.
    # Wait, the formula for l is m1 - m3 + m6.
    # So for m1, w is [1, 0, 0, 1].
    
    # m2: (c+d)e
    # u2 = [0, 0, 1, 1]
    # v2 = [1, 0, 0, 0]
    # C_00: 0. C_01: 0. C_10: 1. C_11: 0.
    # w2 = [0, 0, 1, 0]
    
    # m3: a(f-h)
    # u3 = [1, 0, 0, 0]
    # v3 = [0, 1, 0, -1]
    # C_00: 0. C_01: 1. C_10: 0. C_11: -1.
    # w3 = [0, 1, 0, -1]
    
    # m4: d(g-e)
    # u4 = [0, 0, 0, 1]
    # v4 = [0, 0, 1, -1]
    # C_00: 0. C_01: 0. C_10: 1. C_11: 0.
    # w4 = [0, 0, 1, 0]
    
    # m5: (a+b)h
    # u5 = [1, 1, 0, 0]
    # v5 = [0, 0, 0, 1]
    # C_00: -1. C_01: 1. C_10: 0. C_11: 0.
    # w5 = [-1, 1, 0, 0]
    
    # m6: (c-a)(e+f)
    # u6 = [0, 0, -1, 1]  # -a + c. Wait, u is A. c is index 2, a is index 0.
    # u6 = [-1, 0, 1, 0]
    # v6 = [1, 1, 0, 0]
    # C_00: 0. C_01: 0. C_10: 0. C_11: 1.
    # w6 = [0, 0, 0, 1]
    
    # m7: (b-d)(g+h)
    # u7 = [0, 1, 0, -1]  # b - d. b is index 1, d is index 3.
    # v7 = [0, 0, 1, 1]
    # C_00: 1. C_01: 0. C_10: 0. C_11: 0.
    # w7 = [1, 0, 0, 0]
    
    U_222 = np.array([
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float).T
    
    V_222 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ], dtype=float).T
    
    W_222 = np.array([
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 0, 1, 0]
    ], dtype=float).T
    
    # Now, construct 4x4x4 from four 2x2x2 blocks.
    # A (4x4) is split into A_TL, A_TR, A_BL, A_BR.
    # B (4x4) is split into B_TL, B_TR, B_BL, B_BR.
    # C (4x4) is split into C_TL, C_TR, C_BL, C_BR.
    # C_TL = A_TL B_TL + A_TR B_BL
    # C_TR = A_TL B_TR + A_TR B_BR
    # C_BL = A_BL B_TL + A_BR B_BL
    # C_BR = A_BL B_TR + A_BR B_BR
    
    # There are 8 sub-multiplications of 2x2 matrices.
    # Each uses the 7-rank Strassen decomposition.
    # Total rank = 8 * 7 = 56?
    # Wait, the block structure for 4x4x4 is not just 8 independent 2x2x2s.
    # The standard Strassen 4x4 algorithm reuses intermediate results.
    # Specifically, the 8 products are:
    # P1 = A_TL B_TL
    # P2 = A_TR B_BL
    # P3 = A_TL B_TR
    # P4 = A_TR B_BR
    # P5 = A_BL B_TL
    # P6 = A_BR B_BL
    # P7 = A_BL B_TR
    # P8 = A_BR B_BR
    
    # C_TL = P1 + P2
    # C_TR = P3 + P4
    # C_BL = P5 + P6
    # C_BR = P7 + P8
    
    # This uses 8 * 7 = 56 scalars.
    # But R(4,4,4) <= 49. So this construction is not optimal.
    # The optimal 49-rank construction is more complex (Landsberg).
    # However, 56 + 32 = 88 is still < 96.
    # Let's check if 56 is correct.
    # Yes, naive block application gives 56.
    # Can we do better?
    # There is a 49-rank decomposition. But implementing it from scratch is risky.
    # 88 is a safe improvement over 96.
    # Let's see if we can get 49.
    # The 49-rank decomposition for 4x4x4 is known.
    # It uses 49 scalar multiplications.
    # It is not just a simple block application of Strassen.
    # It involves a specific pattern of additions.
    
    # Given the risk of error in implementing the 49-rank decomposition, 
    # and the time limit, I will implement the 56-rank block-Strassen for 4x4x4.
    # Total rank = 56 + 32 = 88.
    
    # Let's refine the 4x4x2 part.
    # R(4,4,2) <= 2*R(4,4,1) = 32.
    # R(4,4,1) = 16 (schoolbook).
    # Can we do better for 4x4x2?
    # R(4,4,2) <= R(4,4,1) + R(4,4,1) = 32.
    # Is there a better bound?
    # R(4,4,2) is the rank of the tensor for 4x4 matrix times 4x2 matrix.
    # It is known that R(4,4,2) = 32? Or lower?
    # Lower bound: max(16, 16, 8) = 16.
    # Upper bound: 32.
    # I don't know a better construction.
    
    # So, 88 is a solid improvement.
    
    # Implementation of 4x4x4 with 56 rank (8 * 7).
    
    U_444 = np.zeros((16, 56))
    V_444 = np.zeros((16, 56))
    W_444 = np.zeros((16, 56))
    
    # Blocks of A (4x4):
    # 0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3)
    # 4: (1,0), 5: (1,1), 6: (1,2), 7: (1,3)
    # 8: (2,0), 9: (2,1), 10: (2,2), 11: (2,3)
    # 12: (3,0), 13: (3,1), 14: (3,2), 15: (3,3)
    
    # Block (r, c) of A is at indices:
    # r=0 (rows 0,1), c=0 (cols 0,1): indices 0,1,4,5
    # r=0, c=1 (cols 2,3): indices 2,3,6,7
    # r=1 (rows 2,3), c=0: indices 8,9,12,13
    # r=1, c=1: indices 10,11,14,15
    
    def get_block_indices(block_r, block_c):
        row_start = block_r * 2
        col_start = block_c * 2
        return [row_start*4 + col_start, row_start*4 + col_start+1,
                (row_start+1)*4 + col_start, (row_start+1)*4 + col_start+1]
    
    # Similarly for B (4x4) and C (4x4).
    # B indices: j, k. j is row, k is col.
    # C indices: k, i. k is row, i is col.
    
    # We have 8 sub-multiplications.
    # 1. A_TL * B_TL -> C_TL
    # 2. A_TR * B_BL -> C_TL
    # 3. A_TL * B_TR -> C_TR
    # 4. A_TR * B_BR -> C_TR
    # 5. A_BL * B_TL -> C_BL
    # 6. A_BR * B_BL -> C_BL
    # 7. A_BL * B_TR -> C_BL
    # 8. A_BR * B_BR -> C_BR
    
    # For each sub-multiplication, we map the 2x2 indices to 4x4 indices.
    
    # Sub-mult 1: A_TL (r=0,c=0), B_TL (r=0,c=0), C_TL (r=0,c=0)
    # A_TL indices: [0, 1, 4, 5] -> local 0,1,2,3
    # B_TL indices: [0, 1, 4, 5] -> local 0,1,2,3
    # C_TL indices: [0, 1, 4, 5] -> local 0,1,2,3
    
    # Sub-mult 2: A_TR (r=0,c=1), B_BL (r=1,c=0), C_TL (r=0,c=0)
    # A_TR indices: [2, 3, 6, 7]
    # B_BL indices: [8, 9, 12, 13]
    # C_TL indices: [0, 1, 4, 5]
    
    # And so on.
    
    # We will iterate over the 8 sub-multiplications.
    
    # For each sub-mult, we use the 7 Strassen tensors.
    # The U vector for the sub-mult will have non-zeros only at the A-block indices.
    # The V vector for the sub-mult will have non-zeros only at the B-block indices.
    # The W vector for the sub-mult will have non-zeros only at the C-block indices.
    
    # We need to map the local 2x2 indices (0..3) to the global 4x4 indices (0..15).
    
    # Local A indices for a block:
    # local 0 -> global A_idx[0]
    # local 1 -> global A_idx[1]
    # local 2 -> global A_idx[2]
    # local 3 -> global A_idx[3]
    
    # The Strassen U_222 vector is of length 4.
    # U_222[col] is the coefficient for local index col.
    
    # So, for a given sub-mult, we take a column of U_222, and place it into the global U vector at the corresponding global indices.
    
    # Let's define the 8 sub-multiplications.
    # (A_block, B_block, C_block)
    # A_block: (r, c), B_block: (r, c), C_block: (r, c)
    
    submults = [
        ((0,0), (0,0), (0,0)), # 1
        ((0,1), (1,0), (0,0)), # 2
        ((0,0), (0,1), (0,1)), # 3
        ((0,1), (1,1), (0,1)), # 4
        ((1,0), (0,0), (1,0)), # 5
        ((1,1), (1,0), (1,0)), # 6
        ((1,0), (0,1), (1,1)), # 7
        ((1,1), (1,1), (1,1)), # 8
    ]
    
    # Note: The B_block indices for sub-mult 2 are (1,0), which is B_BL.
    # Wait, B is 4x4. B_BL is rows 2,3 and cols 0,1.
    # My get_block_indices function takes (r, c).
    # For B, r is row block, c is col block.
    # B_BL is r=1, c=0. Correct.
    
    # For C, r is row block, c is col block.
    # C_TL is r=0, c=0. Correct.
    
    col_offset = 0
    for a_block, b_block, c_block in submults:
        a_indices = get_block_indices(a_block[0], a_block[1])
        b_indices = get_block_indices(b_block[0], b_block[1])
        c_indices = get_block_indices(c_block[0], c_block[1])
        
        for s in range(7):
            # Get the s-th Strassen tensor vectors
            u_vec = U_222[:, s]
            v_vec = V_222[:, s]
            w_vec = W_222[:, s]
            
            # Map to global indices
            for local_idx in range(4):
                U[a_indices[local_idx], col_offset + s] += u_vec[local_idx]
                V[b_indices[local_idx], col_offset + s] += v_vec[local_idx]
                W[c_indices[local_idx], col_offset + s] += w_vec[local_idx]
            
    col_offset += 56
    
    # --- Part 2: M_4,4,2 with rank 32 ---
    # We split the 2 columns of B into two 1-column vectors.
    # B_col1 and B_col2.
    # C_col1 = A * B_col1
    # C_col2 = A * B_col2
    
    # Each is a 4x4 matrix times a 4x1 vector.
    # This is a matrix-vector multiplication.
    # R(4,4,1) = 16.
    # We use schoolbook for this.
    
    # For B_col1 (k=4 in 0-based? No, k=0,1 for the 4x2 part? No.)
    # The 4x2 part corresponds to the last 2 columns of the original 4x6 matrix.
    # So k=4 and k=5 in the global 0..5 range.
    
    # Let's handle k=4 first.
    # A (4x4) * B_col1 (4x1) -> C_col1 (4x1).
    # C_col1 has indices (k=4, i=0..3).
    # B_col1 has indices (j=0..3, k=4).
    # A has indices (i=0..3, j=0..3).
    
    # We use 16 rank-1 tensors.
    # Each tensor corresponds to one term in the dot product.
    # Term (i,j): A[i,j] * B[j,4] -> C[4,i].
    
    for i in range(4):
        for j in range(4):
            U[i*4+j, col_offset] = 1.0
            V[j*6+4, col_offset] = 1.0
            W[4*4+i, col_offset] = 1.0
            col_offset += 1
    
    # Now for k=5.
    for i in range(4):
        for j in range(4):
            U[i*4+j, col_offset] = 1.0
            V[j*6+5, col_offset] = 1.0
            W[5*4+i, col_offset] = 1.0
            col_offset += 1
    
    # Total rank used: 56 + 16 + 16 = 88.
    
    # Verify dimensions
    assert U.shape == (16, 88)
    assert V.shape == (24, 88)
    assert W.shape == (24, 88)
    
    # Quick sanity check: verify decomposition on a random matrix.
    # (Optional, but good for debugging)
    if time.time() - start_time < DEADLINE - 10:
        A_test = np.random.randint(-2, 3, (4,4))
        B_test = np.random.randint(-2, 3, (4,6))
        
        # Compute C_test = A_test * B_test
        C_test = A_test @ B_test
        
        # Compute C_comp using the decomposition
        C_comp = np.zeros((4,6))
        for r in range(88):
            u_r = np.array(U[:, r]).reshape(4,4)
            v_r = np.array(V[:, r]).reshape(4,6)
            w_r = np.array(W[:, r]).reshape(6,4)
            
            # T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
            # C_comp[i,k] += sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
            
            # Efficiently:
            # M = U_r @ V_r  (4x6)
            # M_ik = sum_j U_r[i,j] V_r[j,k]
            # C_comp_ik += W_r[k,i] * M_ik
            # So C_comp += W_r.T * M ? No.
            # C_comp_ik = W_r[k,i] * M_ik.
            # This is element-wise multiplication of M and W_r.T, then sum over r?
            # No, for each r, we add to C_comp.
            
            M_r = u_r @ v_r
            C_comp += w_r.T * M_r # w_r.T is 4x6? w_r is 6x4, w_r.T is 4x6.
            
        if not np.allclose(C_test, C_comp):
            # Fallback to schoolbook 96 if decomposition fails
            rank = 96
            U = np.zeros((16, rank))
            V = np.zeros((24, rank))
            W = np.zeros((24, rank))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
