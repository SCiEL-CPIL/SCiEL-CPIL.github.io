import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We want to decompose the multiplication of A (2x4) and B (4x8) into C (2x8).
    # We can view A as a 2x2 block matrix where each block is 1x2? No, that's awkward for Strassen.
    # Let's view A as 2 blocks of size 2x2? No, A is 2x4.
    # Let's view A as a 1x2 block matrix where each block is 2x2.
    # Let A = [A0, A1] where A0, A1 are 2x2.
    # Let B be a 2x4 block matrix where each block is 2x2.
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2.
    # Then C = A @ B = [A0 @ [B00, B01] + A1 @ [B10, B11]]
    # C is 2x8. We can view C as a 1x4 block matrix of 2x2 blocks.
    # C = [C0, C1, C2, C3]
    # C0 = A0 @ B00 + A1 @ B10
    # C1 = A0 @ B01 + A1 @ B11
    # C2 = ... wait, A is 2x4, B is 4x8.
    # A = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    # B = [B_top_left, B_top_right, B_bottom_left, B_bottom_right] is not right.
    # B is 4x8. Let's partition B into 2x2 blocks.
    # B = [[B00, B01], [B10, B11]] where each is 2x2.
    # A = [A0, A1] where A0, A1 are 2x2.
    # C = A @ B = [A0, A1] * [[B00, B01], [B10, B11]]
    # C = [A0*B00 + A1*B10, A0*B01 + A1*B11]
    # So C consists of two 2x2 blocks? No, C is 2x8.
    # C_col_0_1 = A0*B00 + A1*B10 (2x2)
    # C_col_2_3 = A0*B01 + A1*B11 (2x2)
    # Wait, B is 4x8.
    # Let's partition B into 4 columns of width 2.
    # B = [B0, B1, B2, B3] where each Bj is 4x2.
    # This doesn't align with 2x2 Strassen blocks directly unless we partition B into 2x2 blocks.
    # Let B = [[B00, B01], [B10, B11]] where each is 2x2.
    # A = [A0, A1] where A0, A1 are 2x2.
    # C = A @ B.
    # C is 2x8.
    # Let's partition C into 4 blocks of 2x2: C = [C0, C1, C2, C3].
    # C0 = A0 @ B00 + A1 @ B10 ?
    # Let's check indices.
    # A is 2x4. A[i, j]. i in 0..1, j in 0..3.
    # A0 = A[:, 0:2], A1 = A[:, 2:4].
    # B is 4x8. B[j, k]. j in 0..3, k in 0..7.
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # Wait, B has 8 columns. So B00, B01, B02, B03?
    # Let's partition B into 2 rows of 2x4 blocks?
    # B_top = B[0:2, :], B_bottom = B[2:4, :].
    # B_top = [B00, B01, B02, B03] where each is 2x2.
    # B_bottom = [B10, B11, B12, B13] where each is 2x2.
    # A = [A0, A1] where A0, A1 are 2x2.
    # C = A @ B = A0 @ B_top + A1 @ B_bottom.
    # C is 2x8.
    # C = [C0, C1, C2, C3] where each is 2x2.
    # C0 = A0 @ B00 + A1 @ B10
    # C1 = A0 @ B01 + A1 @ B11
    # C2 = A0 @ B02 + A1 @ B12
    # C3 = A0 @ B03 + A1 @ B13
    
    # So we have 4 independent 2x2 matrix multiplications?
    # No, each Ci is a sum of two 2x2 multiplications.
    # Ci = A0 @ B_i_top + A1 @ B_i_bottom.
    # This is not a single 2x2 multiplication. It's a linear combination of two.
    # However, we can treat the whole thing as a tensor product.
    # The operation is (A0, A1) x (B_top, B_bottom) -> (C0, C1, C2, C3).
    # This can be viewed as 8 independent 2x2 multiplications if we expand the sums?
    # No, the sum is bilinear.
    # Ci = A0 @ B_i_top + A1 @ B_i_bottom.
    # We can compute M1 = A0 @ B_i_top and M2 = A1 @ B_i_bottom using Strassen (rank 7 each).
    # Then Ci = M1 + M2.
    # This would take 4 * 2 * 7 = 56 scalar multiplications.
    # Rank 56.
    
    # Let's construct the factors U, V, W for this structure.
    # The tensor T has indices corresponding to:
    # U index: (A_block, A_row, A_col) -> 2 * 2 * 2 = 8 elements?
    # A is 2x4. Flattened index u in 0..7.
    # A0 corresponds to u in 0..3. A1 corresponds to u in 4..7.
    # Within A0 (2x2), indices are 0,1,2,3.
    # Within A1 (2x2), indices are 4,5,6,7.
    
    # V index: (B_block, B_row, B_col) -> B is 4x8. Flattened index v in 0..31.
    # B_i_top corresponds to B[0:2, 2*i:2*i+2].
    # B_i_bottom corresponds to B[2:4, 2*i:2*i+2].
    # For a fixed i (0..3):
    # B_i_top indices:
    #   (0, 0) -> 0*8 + 2*i + 0 = 2i
    #   (0, 1) -> 0*8 + 2*i + 1 = 2i+1
    #   (1, 0) -> 1*8 + 2*i + 0 = 8+2i
    #   (1, 1) -> 1*8 + 2*i + 1 = 9+2i
    # B_i_bottom indices:
    #   (0, 0) -> 2*8 + 2*i + 0 = 16+2i
    #   (0, 1) -> 2*8 + 2*i + 1 = 17+2i
    #   (1, 0) -> 3*8 + 2*i + 0 = 24+2i
    #   (1, 1) -> 3*8 + 2*i + 1 = 25+2i
    
    # W index: (C_block, C_row, C_col) -> C is 2x8. Flattened index w in 0..15.
    # Convention: W[k*n+i, r] where k is col of B/C, i is row of A/C.
    # C is 2x8. i in 0..1, k in 0..7.
    # W index = k*2 + i.
    # C_i corresponds to columns 2*i and 2*i+1.
    # C_i[0, 0] -> k=2i, i_row=0 -> w = 2i*2 + 0 = 4i
    # C_i[0, 1] -> k=2i+1, i_row=0 -> w = (2i+1)*2 + 0 = 4i+2
    # C_i[1, 0] -> k=2i, i_row=1 -> w = 2i*2 + 1 = 4i+1
    # C_i[1, 1] -> k=2i+1, i_row=1 -> w = (2i+1)*2 + 1 = 4i+3
    
    # We will use the standard Strassen decomposition for 2x2x2.
    # Strassen factors (rank 7) for 2x2:
    # U_s (4x7), V_s (4x7), W_s (4x7).
    # Indices for 2x2:
    # U_s: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V_s: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W_s: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  <-- Note: W index k*2+i.
    #       k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
    #       So W_s[0] is C00, W_s[1] is C10, W_s[2] is C01, W_s[3] is C11.
    
    U_s = np.array([
        [1, 1, 0, 0, 1, 1, 0], # A00
        [0, 0, 1, 0, 1, 0, 1], # A01
        [0, 1, 0, 1, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, 1]  # A11
    ])
    # Let's re-verify Strassen U, V, W.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U columns (coeffs for A):
    # M1: A00=1, A11=1. Others 0.
    # M2: A10=1, A11=1. Others 0.
    # M3: A00=1. Others 0.
    # M4: A11=1. Others 0.
    # M5: A00=1, A01=1. Others 0.
    # M6: A10=1, A00=-1. Others 0.
    # M7: A01=1, A11=-1. Others 0.
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    
    # V columns (coeffs for B):
    # M1: B00=1, B11=1.
    # M2: B00=1.
    # M3: B01=1, B11=-1.
    # M4: B10=1, B00=-1.
    # M5: B11=1.
    # M6: B00=1, B01=1.
    # M7: B10=1, B11=1.
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    
    # W columns (coeffs for C):
    # C00: M1=1, M4=1, M5=-1, M7=-1.
    # C01: M3=1, M5=1.
    # C10: M2=1, M4=1.
    # C11: M1=1, M2=-1, M3=1, M7=1.
    
    # W index order: 0:C00, 1:C10, 2:C01, 3:C11.
    W_strassen = np.array([
        [1, 0, 0, 1],  # C00
        [0, 1, 0, 0],  # C10
        [0, 0, 1, 1],  # C01
        [1, -1, 1, 1]  # C11
    ])
    
    # Now build the full U, V, W for 2x4x8.
    # Total rank = 4 (blocks i) * 2 (A0/A1) * 7 (Strassen) = 56.
    rank = 56
    U = np.zeros((n*m, rank), dtype=int) # 8 x 56
    V = np.zeros((m*p, rank), dtype=int) # 32 x 56
    W = np.zeros((p*n, rank), dtype=int) # 16 x 56
    
    col = 0
    for i_block in range(4):
        # Process C_i = A0 @ B_i_top + A1 @ B_i_bottom
        
        # Part 1: A0 @ B_i_top
        # A0 indices in U: 0, 1, 2, 3.
        # B_i_top indices in V: 2i, 2i+1, 8+2i, 9+2i.
        # C_i indices in W: 4i, 4i+1, 4i+2, 4i+3.
        
        for r_s in range(7):
            # U part
            U[0, col] = U_strassen[0, r_s]
            U[1, col] = U_strassen[1, r_s]
            U[2, col] = U_strassen[2, r_s]
            U[3, col] = U_strassen[3, r_s]
            
            # V part
            V[2*i_block, col] = V_strassen[0, r_s]
            V[2*i_block + 1, col] = V_strassen[1, r_s]
            V[8 + 2*i_block, col] = V_strassen[2, r_s]
            V[9 + 2*i_block, col] = V_strassen[3, r_s]
            
            # W part
            W[4*i_block, col] = W_strassen[0, r_s]
            W[4*i_block + 1, col] = W_strassen[1, r_s]
            W[4*i_block + 2, col] = W_strassen[2, r_s]
            W[4*i_block + 3, col] = W_strassen[3, r_s]
            
            col += 1
        
        # Part 2: A1 @ B_i_bottom
        # A1 indices in U: 4, 5, 6, 7.
        # B_i_bottom indices in V: 16+2i, 17+2i, 24+2i, 25+2i.
        # C_i indices in W: 4i, 4i+1, 4i+2, 4i+3. (Same C block, so add to W)
        
        for r_s in range(7):
            # U part
            U[4, col] = U_strassen[0, r_s]
            U[5, col] = U_strassen[1, r_s]
            U[6, col] = U_strassen[2, r_s]
            U[7, col] = U_strassen[3, r_s]
            
            # V part
            V[16 + 2*i_block, col] = V_strassen[0, r_s]
            V[17 + 2*i_block, col] = V_strassen[1, r_s]
            V[24 + 2*i_block, col] = V_strassen[2, r_s]
            V[25 + 2*i_block, col] = V_strassen[3, r_s]
            
            # W part (Add to existing W entries for this C block)
            W[4*i_block, col] = W_strassen[0, r_s]
            W[4*i_block + 1, col] = W_strassen[1, r_s]
            W[4*i_block + 2, col] = W_strassen[2, r_s]
            W[4*i_block + 3, col] = W_strassen[3, r_s]
            
            col += 1
            
    # Verification (optional but good for sanity check in dev)
    # A = np.random.randint(-5, 6, (n, m))
    # B = np.random.randint(-5, 6, (m, p))
    # C_expected = A @ B
    # C_computed = np.zeros((n, p), dtype=int)
    # for r in range(rank):
    #     # Sum over j for U and V
    #     # C[i, k] += W[k*n+i, r] * (sum_j U[i*m+j, r] A[i, j]) * (sum_j V[j*p+k, r] B[j, k])
    #     # This is O(n*m*p*rank) which is small.
    #     for i in range(n):
    #         for k in range(p):
    #             w_val = W[k*n+i, r]
    #             if w_val != 0:
    #                 u_sum = 0
    #                 v_sum = 0
    #                 for j in range(m):
    #                     u_sum += U[i*m+j, r] * A[i, j]
    #                     v_sum += V[j*p+k, r] * B[j, k]
    #                 C_computed[i, k] += w_val * u_sum * v_sum
    # if not np.allclose(C_computed, C_expected):
    #     raise ValueError("Decomposition failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
