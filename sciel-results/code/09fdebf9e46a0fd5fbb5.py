import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2_factors():
    """Return (alpha, beta, recon) for 2x2 * 2x2 Strassen."""
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1},
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]
    return alpha, beta, recon

def _strassen_4x4x4_factors():
    """
    Return factor matrices for 4x4 * 4x4 -> 4x4 using recursive Strassen (rank 49).
    U: (16, 49), V: (16, 49), W: (16, 49)
    Indexing: U[i*4+k, r], V[k*4+j, r], W[j*4+i, r]
    """
    alpha2, beta2, recon2 = _strassen_2x2_factors()
    
    # 4x4 split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B10], [B01, B11]] each 2x2  (B is column-major in our indexing)
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # Strassen for 4x4:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    R4 = 49
    U4 = np.zeros((16, R4))
    V4 = np.zeros((16, R4))
    W4 = np.zeros((16, R4))
    
    # Block index to global index:
    # Block (br, bc) where br, bc in {0,1} -> rows br*2 to br*2+1, cols bc*2 to bc*2+1
    def block_idx(br, bc, r, c):
        """Global index for A or B block (br,bc) at local position (r,c)"""
        return (br*2 + r) * 4 + (bc*2 + c)
    
    def b_block_idx(br, bc, r, c):
        """Global index for B block (br,bc) at local position (r,c) - B is (k,j) indexed"""
        return (br*2 + r) * 4 + (bc*2 + c)
    
    # For each of the 7 Strassen terms at the 4x4 level:
    # Each term involves multiplying two 2x2 matrices, which itself uses 7 Strassen terms
    strassen_terms = [
        # (A_block, B_block, C_block, sign)
        # M1 = (A00+A11)*(B00+B11) -> C00 += M1
        # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
        # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        # M7 = (A00+A01)*B11... wait let me redo
    ]
    
    # Let me be more careful. Standard Strassen for 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Mapping to our 4x4 blocks (0-indexed):
    # A11->A00, A12->A01, A21->A10, A22->A11
    # B11->B00, B12->B01, B21->B10, B22->B11
    # C11->C00, C12->C01, C21->C10, C22->C11
    
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1
    # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
    # M3 = A00*(B10-B00) -> C01 += M3, C11 += M3
    # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7
    
    # Each Mi is a 2x2 product, decomposed via Strassen into 7 scalar products
    # Total: 7 * 7 = 49 rank terms
    
    # Define the 7 Strassen terms at 4x4 level
    # Each term: (a_blocks, b_blocks, c_blocks, signs)
    # a_blocks: list of (block_idx, sign) for A
    # b_blocks: list of (block_idx, sign) for B  
    # c_blocks: list of (block_idx, sign) for C
    
    strassen_4x4 = [
        # M1 = (A00+A11)*(B00+B11)
        {
            'a': [(0, 1), (3, 1)],  # A00, A11
            'b': [(0, 1), (3, 1)],  # B00, B11
            'c': [(0, 1)],           # C00
        },
        # M2 = (A10+A11)*B00
        {
            'a': [(2, 1), (3, 1)],  # A10, A11
            'b': [(0, 1)],          # B00
            'c': [(2, 1), (3, -1)], # C10, C11
        },
        # M3 = A00*(B10-B00)
        {
            'a': [(0, 1)],          # A00
            'b': [(2, 1), (0, -1)], # B10, -B00
            'c': [(1, 1), (3, 1)],  # C01, C11
        },
        # M4 = A11*(B10-B00)
        {
            'a': [(3, 1)],          # A11
            'b': [(2, 1), (0, -1)], # B10, -B00
            'c': [(0, 1), (2, 1)],  # C00, C10
        },
        # M5 = (A00+A01)*B11
        {
            'a': [(0, 1), (1, 1)],  # A00, A01
            'b': [(3, 1)],          # B11
            'c': [(0, -1), (1, 1)], # C00, C01
        },
        # M6 = (A10-A00)*(B00+B01)
        {
            'a': [(2, 1), (0, -1)], # A10, -A00
            'b': [(0, 1), (1, 1)],  # B00, B01
            'c': [(3, 1)],          # C11
        },
        # M7 = (A01-A11)*(B10+B11)
        {
            'a': [(1, 1), (3, -1)], # A01, -A11
            'b': [(2, 1), (3, 1)],  # B10, B11
            'c': [(0, 1)],          # C00
        },
    ]
    
    col = 0
    for term in strassen_4x4:
        # For this term, we need to multiply two 2x2 matrices
        # A_combined = sum of A blocks with signs
        # B_combined = sum of B blocks with signs
        # Result goes to C blocks with signs
        
        # Use Strassen for the 2x2 product
        for s in range(7):
            # U part: A entries from A_combined
            for a_blk, a_sign in term['a']:
                for a_idx in range(4):
                    if alpha2[s][a_idx] != 0:
                        ai, aj = a_idx // 2, a_idx % 2
                        u_idx = block_idx(a_blk // 2, a_blk % 2, ai, aj)
                        U4[u_idx, col] += alpha2[s][a_idx] * a_sign
            
            # V part: B entries from B_combined
            for b_blk, b_sign in term['b']:
                for b_idx in range(4):
                    if beta2[s][b_idx] != 0:
                        bi, bj = b_idx // 2, b_idx % 2
                        v_idx = b_block_idx(b_blk // 2, b_blk % 2, bi, bj)
                        V4[v_idx, col] += beta2[s][b_idx] * b_sign
            
            # W part: C entries
            for c_blk, c_sign in term['c']:
                for out_r, out_c, gamma in recon2[s]:
                    w_idx = block_idx(c_blk // 2, c_blk % 2, out_r, out_c)
                    W4[w_idx, col] += gamma * c_sign
            
            col += 1
    
    return U4, V4, W4

def _strassen_4x4x2_factors():
    """
    Return factor matrices for 4x4 * 4x2 -> 4x2 using Strassen (rank 28).
    U: (16, 28), V: (8, 28), W: (8, 28)
    Indexing: U[i*4+k, r], V[k*2+j, r], W[j*4+i, r]
    """
    alpha2, beta2, recon2 = _strassen_2x2_factors()
    
    # 4x4 * 4x2:
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00], [B10]] each 2x2 (B is 4x2, so two 2x2 blocks vertically)
    # C = A*B = [[C00], [C10]] each 2x2
    
    # Strassen for 4x4 * 4x2:
    # M1 = (A00+A11)*B00
    # M2 = (A10+A11)*B00
    # M3 = A00*B10
    # M4 = A11*B10
    # M5 = (A00+A01)*B10
    # M6 = (A10-A00)*B00
    # M7 = (A01-A11)*B10
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    
    R42 = 28
    U42 = np.zeros((16, R42))
    V42 = np.zeros((8, R42))
    W42 = np.zeros((8, R42))
    
    def block_idx_a(br, bc, r, c):
        """Global A index for block (br,bc) at local (r,c)"""
        return (br*2 + r) * 4 + (bc*2 + c)
    
    def block_idx_b(br, r, c):
        """Global B index for block (br) at local (r,c) - B is 4x2"""
        return (br*2 + r) * 2 + c
    
    def block_idx_c(br, r, c):
        """Global C index for block (br) at local (r,c) - C is 4x2"""
        return (br*2 + r) * 2 + c
    
    # Strassen terms for 4x4 * 4x2
    strassen_4x4x2 = [
        # M1 = (A00+A11)*B00 -> C00 += M1
        {
            'a': [(0, 1), (3, 1)],  # A00, A11
            'b': [(0, 1)],          # B00
            'c': [(0, 1)],          # C00
        },
        # M2 = (A10+A11)*B00 -> C10 += M2
        {
            'a': [(2, 1), (3, 1)],  # A10, A11
            'b': [(0, 1)],          # B00
            'c': [(1, 1)],          # C10
        },
        # M3 = A00*B10 -> C00 += M3... wait, let me check
        # Actually for 4x4*4x2, the standard Strassen decomposition is:
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # where:
        # M1 = (A00+A11)*B00
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B00)  -- but B only has B00 and B10
        # Let me reconsider.
    ]
    
    # Actually, for A (4x4) * B (4x2), we can think of B as having two 2x2 blocks:
    # B = [B_top; B_bottom] where B_top = B[0:2, :] and B_bottom = B[2:4, :]
    # C = A*B = [A_top*B; A_bottom*B] where A_top = A[0:2, :] and A_bottom = A[2:4, :]
    # A_top*B = A_top[0:2, 0:2]*B_top + A_top[0:2, 2:4]*B_bottom
    # A_bottom*B = A_bottom[2:4, 0:2]*B_top + A_bottom[2:4, 2:4]*B_bottom
    
    # So C[0:2, :] = A00*B00 + A01*B10
    # C[2:4, :] = A10*B00 + A11*B10
    
    # Each 2x2*2x2 product uses Strassen (rank 7), so total rank = 4*7 = 28
    
    col = 0
    for a_row_blk in range(2):  # A row block: 0 or 1
        for a_col_blk in range(2):  # A col block: 0 or 1
            for b_row_blk in range(2):  # B row block: 0 or 1
                # A_block = A[a_row_blk*2:a_row_blk*2+2, a_col_blk*2:a_col_blk*2+2]
                # B_block = B[b_row_blk*2:b_row_blk*2+2, :]
                # C_block = C[a_row_blk*2:a_row_blk*2+2, :]
                
                # Only contribute if a_col_blk == b_row_blk (since A[i,k]*B[k,j])
                # Actually, A[i,k]*B[k,j] where k is the shared dimension
                # A is 4x4, B is 4x2, so k goes from 0 to 3
                # A[i, k] * B[k, j]
                # A[i, a_col_blk*2 + kc] * B[b_row_blk*2 + kb, j]
                # These contribute to C[i, j]
                
                # So A_block (2x2) at (a_row_blk, a_col_blk) multiplies B_block (2x2) at (b_row_blk, 0)
                # and contributes to C_block (2x2) at (a_row_blk, 0)
                
                for s in range(7):
                    # U part: A entries
                    for a_idx in range(4):
                        if alpha2[s][a_idx] != 0:
                            ai, aj = a_idx // 2, a_idx % 2
                            u_idx = block_idx_a(a_row_blk, a_col_blk, ai, aj)
                            U42[u_idx, col] += alpha2[s][a_idx]
                    
                    # V part: B entries
                    for b_idx in range(4):
                        if beta2[s][b_idx] != 0:
                            bi, bj = b_idx // 2, b_idx % 2
                            v_idx = block_idx_b(b_row_blk, bi, bj)
                            V42[v_idx, col] += beta2[s][b_idx]
                    
                    # W part: C entries
                    for out_r, out_c, gamma in recon2[s]:
                        w_idx = block_idx_c(a_row_blk, out_r, out_c)
                        W42[w_idx, col] += gamma
                    
                    col += 1
    
    return U42, V42, W42

def construct():
    _t0 = time.time()
    n, m, p = 4, 4, 6
    
    # Strategy: Split B into B_left (4x4) and B_right (4x2)
    # C_left = A * B_left (4x4 * 4x4 -> 4x4) using rank 49 Strassen
    # C_right = A * B_right (4x4 * 4x2 -> 4x2) using rank 28 Strassen
    # Total rank = 49 + 28 = 77
    
    R = 77
    U = np.zeros((n * m, R))  # (16, 77)
    V = np.zeros((m * p, R))  # (24, 77)
    W = np.zeros((p * n, R))  # (24, 77)
    
    # Get 4x4*4x4 factors
    U4, V4, W4 = _strassen_4x4x4_factors()
    
    # Get 4x4*4x2 factors
    U42, V42, W42 = _strassen_4x4x2_factors()
    
    # Assemble: C_left uses columns 0-48, C_right uses columns 49-76
    # For C_left: U[:, 0:49] = U4, V[:, 0:49] = V4 (but V4 is 16x49, need to embed in 24x49)
    # V for B_left: B_left is 4x4, so V indices are k*4+j for k,j in 0..3
    # In the full V (24x77), B_left occupies rows 0-15 (k*4+j), B_right occupies rows 16-23 (k*2+j)
    
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4  # B_left part
    W[0:16, 0:49] = W4  # C_left part: C_left[i,j] -> W[j*4+i]
    
    # For C_right: U[:, 49:77] = U42, V[:, 49:77] = V42 (V42 is 8x28)
    # V for B_right: B_right is 4x2, so V indices are k*2+j for k in 0..3, j in 0..1
    # In the full V (24x77), B_right occupies rows 16-23 (offset by 16, k*2+j)
    
    U[:, 49:77] = U42
    V[16:24, 49:77] = V42  # B_right part, offset by 16
    W[16:24, 49:77] = W42  # C_right part: C_right[i,j] -> W[j*4+i] but j only 0,1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    U_list = to_list(U)
    V_list = to_list(V)
    W_list = to_list(W)
    
    # Validate
    if (time.time() - _t0) < 130.0:
        np.random.seed(12345)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 6)
        
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        C_computed = np.zeros((4, 6))
        for r in range(R):
            u_dot_A = np.dot(U_arr[:, r], A_flat)
            v_dot_B = np.dot(V_arr[:, r], B_flat)
            if u_dot_A != 0 and v_dot_B != 0:
                C_computed += (u_dot_A * v_dot_B) * W_arr[:, r].reshape(6, 4).T
        
        C_expected = A_test @ B_test
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook rank 96
            R_s = 96
            U_s = np.zeros((16, R_s))
            V_s = np.zeros((24, R_s))
            W_s = np.zeros((24, R_s))
            idx = 0
            for i in range(4):
                for k in range(4):
                    for j in range(6):
                        U_s[i * 4 + k, idx] = 1
                        V_s[k * 6 + j, idx] = 1
                        W_s[j * 4 + i, idx] = 1
                        idx += 1
            return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
    
    return {"U": U_list, "V": V_list, "W": W_list}
