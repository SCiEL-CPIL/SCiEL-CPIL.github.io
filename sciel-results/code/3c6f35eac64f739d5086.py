import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36

    start_time = time.time()

    # Build the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Strategy: Use Strassen decomposition for the 4x4 * 4x4 part (first 4 columns of B)
    # and schoolbook for the remaining 5 columns.
    # 4x4 * 4x4 via Strassen: 7 * 7 = 49 multiplications
    # 4x4 * 4x5 via schoolbook: 4 * 4 * 5 = 80 multiplications
    # Total: 49 + 80 = 129

    # Actually, let me reconsider. For 4x4 * 4x5:
    # We can partition B_right (4x5) into [B_right1 (4x2) | B_right2 (4x3)]
    # But that doesn't help directly.
    # 
    # Better approach: Use the fact that 4x4 * 4x1 can be done with 7 mults using Strassen-like
    # But actually for 4x4 * 4x1, the standard approach is 16 mults schoolbook.
    # However, we can use a block approach:
    # A (4x4) * b (4x1) = [A11 A12; A21 A22] * [b1; b2] where Aij are 2x2, bi are 2x1
    # = [A11*b1 + A12*b2; A21*b1 + A22*b2]
    # Each 2x2*2x1 needs 4 mults, so total 8 mults per column.
    # For 5 columns: 5 * 8 = 40 mults.
    # Total: 49 + 40 = 89.

    # Let me build the 4x4*4x4 Strassen part explicitly.
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2 blocks
    # B_left = [[B11, B12], [B21, B22]] where Bij are 2x2 blocks (first 4 columns of B)
    
    # Strassen at 4x4 level:
    # S1 = (A11+A22) * (B11+B22)
    # S2 = (A21+A22) * B11
    # S3 = A11 * (B12-B22)
    # S4 = A22 * (B21-B11)
    # S5 = (A11+A12) * B22
    # S6 = (A21-A11) * (B11+B12)
    # S7 = (A12-A22) * (B21+B22)
    
    # Each S is a 2x2 * 2x2 product, which uses 7 Strassen mults.
    # So 7 * 7 = 49 mults for the 4x4*4x4 part.

    # For the right part (last 5 columns of B):
    # B_right is 4x5. Partition into rows: BR1 = B_right[0:2, :] (2x5), BR2 = B_right[2:4, :] (2x5)
    # C_right_top = A11*BR1 + A12*BR2 (2x5)
    # C_right_bot = A21*BR1 + A22*BR2 (2x5)
    # Each 2x2*2x5 product: for each of 5 columns, 2x2*2x1 needs 4 mults.
    # So 5*4 = 20 mults per 2x2*2x5.
    # Two such products: 40 mults.
    # Total: 49 + 40 = 89.

    R = 89
    U = np.zeros((dim_u, R), dtype=float)
    V = np.zeros((dim_v, R), dtype=float)
    W = np.zeros((dim_w, R), dtype=float)
    
    col = 0

    # Helper function to add a Strassen 2x2*2x2 decomposition
    # a_block: 2x2 array of u-indices for A block
    # b_block: 2x2 array of v-indices for B block  
    # c_block: 2x2 array of w-indices for C block
    # a_signs: 2x2 signs for A combination
    # b_signs: 2x2 signs for B combination
    # c_signs: 2x2 signs for C combination (which C entries get which sign)
    
    def add_strassen_2x2(a_idx, b_idx, c_idx, a_s, b_s, c_s):
        nonlocal col
        # Strassen 2x2*2x2:
        # M1 = (a00+a11)*(b00+b11) -> c00 += M1, c11 += M1
        # M2 = (a10+a11)*b00 -> c10 += M2
        # M3 = a00*(b01-b11) -> c01 += M3
        # M4 = a11*(b10-b00) -> c00 += M4, c10 += M4
        # M5 = (a00+a01)*b11 -> c01 += M5, c00 -= M5
        # M6 = (a10-a00)*(b00+b01) -> c11 += M6
        # M7 = (a01-a11)*(b10+b11) -> c00 += M7, c11 += M7
        
        # M1: (a00+a11)*(b00+b11)
        # U: a00 with sign a_s[0,0], a11 with sign a_s[1,1]
        # V: b00 with sign b_s[0,0], b11 with sign b_s[1,1]
        # W: c00 with sign c_s[0,0], c11 with sign c_s[1,1]
        # But we need separate rank-1 terms for each combination...
        
        # Actually, each Strassen mult is a product of two linear combinations.
        # For M1 = (a00+a11)*(b00+b11), this is actually 4 rank-1 terms if expanded,
        # but in tensor decomposition, it's ONE rank-1 term where U has a00 and a11,
        # V has b00 and b11, and W has c00 and c11.
        # Wait no - a rank-1 term is u * v * w where u, v, w are vectors.
        # The product (a00+a11)*(b00+b11) contributes to c00 and c11.
        # So U has nonzeros at a00 and a11 positions, V has nonzeros at b00 and b11,
        # and W has nonzeros at c00 and c11.
        
        # Let me define the 7 Strassen patterns more carefully.
        # Each pattern specifies which A entries (with signs) form the U vector,
        # which B entries (with signs) form the V vector, and which C entries (with signs) form the W vector.
        
        # M1: U = {a00: a_s[0,0], a11: a_s[1,1]}, V = {b00: b_s[0,0], b11: b_s[1,1]}, W = {c00: c_s[0,0], c11: c_s[1,1]}
        U[a_idx[0,0], col] += a_s[0,0]
        U[a_idx[1,1], col] += a_s[1,1]
        V[b_idx[0,0], col] += b_s[0,0]
        V[b_idx[1,1], col] += b_s[1,1]
        W[c_idx[0,0], col] += c_s[0,0]
        W[c_idx[1,1], col] += c_s[1,1]
        col += 1
        
        # M2: U = {a10: a_s[1,0], a11: a_s[1,1]}, V = {b00: b_s[0,0]}, W = {c10: c_s[1,0]}
        U[a_idx[1,0], col] += a_s[1,0]
        U[a_idx[1,1], col] += a_s[1,1]
        V[b_idx[0,0], col] += b_s[0,0]
        W[c_idx[1,0], col] += c_s[1,0]
        col += 1
        
        # M3: U = {a00: a_s[0,0]}, V = {b01: b_s[0,1], b11: -b_s[1,1]}, W = {c01: c_s[0,1]}
        U[a_idx[0,0], col] += a_s[0,0]
        V[b_idx[0,1], col] += b_s[0,1]
        V[b_idx[1,1], col] -= b_s[1,1]
        W[c_idx[0,1], col] += c_s[0,1]
        col += 1
        
        # M4: U = {a11: a_s[1,1]}, V = {b10: b_s[1,0], b00: -b_s[0,0]}, W = {c00: c_s[0,0], c10: c_s[1,0]}
        U[a_idx[1,1], col] += a_s[1,1]
        V[b_idx[1,0], col] += b_s[1,0]
        V[b_idx[0,0], col] -= b_s[0,0]
        W[c_idx[0,0], col] += c_s[0,0]
        W[c_idx[1,0], col] += c_s[1,0]
        col += 1
        
        # M5: U = {a00: a_s[0,0], a01: a_s[0,1]}, V = {b11: b_s[1,1]}, W = {c01: c_s[0,1], c00: -c_s[0,0]}
        U[a_idx[0,0], col] += a_s[0,0]
        U[a_idx[0,1], col] += a_s[0,1]
        V[b_idx[1,1], col] += b_s[1,1]
        W[c_idx[0,1], col] += c_s[0,1]
        W[c_idx[0,0], col] -= c_s[0,0]
        col += 1
        
        # M6: U = {a10: a_s[1,0], a00: -a_s[0,0]}, V = {b00: b_s[0,0], b01: b_s[0,1]}, W = {c11: c_s[1,1]}
        U[a_idx[1,0], col] += a_s[1,0]
        U[a_idx[0,0], col] -= a_s[0,0]
        V[b_idx[0,0], col] += b_s[0,0]
        V[b_idx[0,1], col] += b_s[0,1]
        W[c_idx[1,1], col] += c_s[1,1]
        col += 1
        
        # M7: U = {a01: a_s[0,1], a11: -a_s[1,1]}, V = {b10: b_s[1,0], b11: b_s[1,1]}, W = {c00: c_s[0,0], c11: c_s[1,1]}
        U[a_idx[0,1], col] += a_s[0,1]
        U[a_idx[1,1], col] -= a_s[1,1]
        V[b_idx[1,0], col] += b_s[1,0]
        V[b_idx[1,1], col] += b_s[1,1]
        W[c_idx[0,0], col] += c_s[0,0]
        W[c_idx[1,1], col] += c_s[1,1]
        col += 1

    # Now build the 4x4*4x4 part using Strassen.
    # A block indices (u-index = i*4+j):
    # A11: rows 0-1, cols 0-1 -> u = [0,1,4,5]
    # A12: rows 0-1, cols 2-3 -> u = [2,3,6,7]
    # A21: rows 2-3, cols 0-1 -> u = [8,9,12,13]
    # A22: rows 2-3, cols 2-3 -> u = [10,11,14,15]
    
    # B_left block indices (v-index = j*9+k, k=0..3):
    # B11: rows 0-1, cols 0-1 -> v = [0,1,9,10]
    # B12: rows 0-1, cols 2-3 -> v = [2,3,11,12]
    # B21: rows 2-3, cols 0-1 -> v = [18,19,27,28]
    # B22: rows 2-3, cols 2-3 -> v = [20,21,29,30]
    
    # C block indices (w-index = k*4+i, k=0..3):
    # C11: rows 0-1, cols 0-1 -> w = [0,1,4,5]
    # C12: rows 0-1, cols 2-3 -> w = [2,3,6,7]
    # C21: rows 2-3, cols 0-1 -> w = [8,9,12,13]
    # C22: rows 2-3, cols 2-3 -> w = [10,11,14,15]
    
    # Define 2x2 index arrays for each block
    A11_idx = np.array([[0, 1], [4, 5]])
    A12_idx = np.array([[2, 3], [6, 7]])
    A21_idx = np.array([[8, 9], [12, 13]])
    A22_idx = np.array([[10, 11], [14, 15]])
    
    B11_idx = np.array([[0, 1], [9, 10]])
    B12_idx = np.array([[2, 3], [11, 12]])
    B21_idx = np.array([[18, 19], [27, 28]])
    B22_idx = np.array([[20, 21], [29, 30]])
    
    C11_idx = np.array([[0, 1], [4, 5]])
    C12_idx = np.array([[2, 3], [6, 7]])
    C21_idx = np.array([[8, 9], [12, 13]])
    C22_idx = np.array([[10, 11], [14, 15]])
    
    # Identity signs
    I2 = np.array([[1, 0], [0, 1]])
    
    # S1 = (A11+A22) * (B11+B22) -> C11 += S1, C11 -= S5 later, etc.
    # Actually, the Strassen formula gives:
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # For S1 = (A11+A22)*(B11+B22):
    # a_s for A11 is I2, for A22 is I2 (both +1)
    # b_s for B11 is I2, for B22 is I2 (both +1)
    # But wait, S1 contributes to both C11 and C22 with +1 sign.
    # In my add_strassen_2x2, I need to call it for each S separately,
    # but the c_signs determine how it contributes to C.
    
    # Actually, I realize my approach is flawed. The Strassen decomposition at the 4x4 level
    # has 7 "super-multiplications", each of which is a 2x2*2x2 product.
    # Each super-multiplication contributes to specific C blocks with specific signs.
    
    # Let me restructure: for each of the 7 Strassen steps, I call add_strassen_2x2
    # with the appropriate A combination, B combination, and C contribution.
    
    # But the issue is that in standard Strassen, each M_i contributes to multiple C entries.
    # My add_strassen_2x2 function handles this by having W with multiple nonzeros.
    
    # Let me redefine: I'll call add_strassen_2x2 for each of the 7 Strassen mults,
    # where the A and B "blocks" are linear combinations of the actual blocks.
    
    # This is getting complex. Let me just use the schoolbook approach for simplicity
    # and correctness, which gives rank 144.
    
    # Reset
    U = np.zeros((dim_u, 144), dtype=float)
    V = np.zeros((dim_v, 144), dtype=float)
    W = np.zeros((dim_w, 144), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1.0
                V[v, col] = 1.0
                W[w, col] = 1.0
                col += 1
    
    # Verify
    T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
    error = np.linalg.norm(T_recon - T)
    
    if error < 1e-9:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    raise Exception("Decomposition verification failed")
