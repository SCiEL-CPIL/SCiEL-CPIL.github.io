"""Rewritten program for 4x4x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 135  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb.copy(), V_fb.copy(), W_fb.copy()
    
    # Strategy: Use Strassen-based decomposition
    # 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # Each 4x4x4 can use rank-49 Strassen decomposition
    # Each 4x4x1 uses rank-16 schoolbook
    # Total: 49 + 49 + 16 = 114
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication, rank 7.
        A is 2x2, B is 2x2, C = A*B is 2x2.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        Indexing: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A = [[a00, a01], [a10, a11]] -> indices 0,1,2,3
        # B = [[b00, b01], [b10, b11]] -> indices 0,1,2,3
        # C = A*B: c[i,k] = sum_j a[i,j]*b[j,k]
        # W index for c[i,k] is k*2+i
        
        # Strassen:
        # M1 = (a00+a11)*(b00+b11) -> contributes to c00(+1), c11(+1)
        U[0, 0] = 1; U[3, 0] = 1  # a00, a11
        V[0, 0] = 1; V[3, 0] = 1  # b00, b11
        W[0, 0] = 1; W[3, 0] = 1  # c[0,0]=W[0], c[1,1]=W[3]
        
        # M2 = (a10+a11)*b00 -> contributes to c10(+1)
        U[2, 1] = 1; U[3, 1] = 1  # a10, a11
        V[0, 1] = 1  # b00
        W[2, 1] = 1  # c[1,0]=W[0*2+1]=W[2]
        
        # M3 = a00*(b01-b11) -> contributes to c01(+1)
        U[0, 2] = 1  # a00
        V[1, 2] = 1; V[3, 2] = -1  # b01, -b11
        W[1, 2] = 1  # c[0,1]=W[1*2+0]=W[2]? No: W[k*2+i] for c[0,1] is W[1*2+0]=W[2]
        # Wait, c[i,k] with i=0, k=1: W[k*n+i] = W[1*2+0] = W[2]
        # But I already used W[2,1]=1 for M2. Let me recheck.
        # c[1,0]: i=1, k=0 -> W[0*2+1] = W[1]
        # c[0,1]: i=0, k=1 -> W[1*2+0] = W[2]
        # c[0,0]: i=0, k=0 -> W[0*2+0] = W[0]
        # c[1,1]: i=1, k=1 -> W[1*2+1] = W[3]
        
        # Let me redo the W indexing carefully:
        # W[k*n + i] where n=2
        # c[0,0]: k=0, i=0 -> W[0]
        # c[0,1]: k=1, i=0 -> W[2]
        # c[1,0]: k=0, i=1 -> W[1]
        # c[1,1]: k=1, i=1 -> W[3]
        
        # M1: c[0,0]+1, c[1,1]+1 -> W[0]=1, W[3]=1 ✓
        # M2: c[1,0]+1 -> W[1]=1 (not W[2]!)
        # Let me fix M2:
        W[2, 1] = 0  # clear wrong value
        W[1, 1] = 1  # c[1,0] -> W[1]
        
        # M3: c[0,1]+1 -> W[2]=1
        W[1, 2] = 0  # clear wrong value  
        W[2, 2] = 1  # c[0,1] -> W[2]
        
        # M4 = a11*(b10-b00) -> contributes to c10(+1), c00(+1)
        U[3, 3] = 1  # a11
        V[2, 3] = 1; V[0, 3] = -1  # b10, -b00
        W[1, 3] = 1; W[0, 3] = 1  # c[1,0]=W[1], c[0,0]=W[0]
        
        # M5 = (a00+a01)*b11 -> contributes to c00(-1), c01(+1)
        U[0, 4] = 1; U[1, 4] = 1  # a00, a01
        V[3, 4] = 1  # b11
        W[0, 4] = -1; W[2, 4] = 1  # c[0,0]=-1, c[0,1]=+1
        
        # M6 = (a10-a00)*(b00+b01) -> contributes to c11(+1)
        U[2, 5] = 1; U[0, 5] = -1  # a10, -a00
        V[0, 5] = 1; V[1, 5] = 1  # b00, b01
        W[3, 5] = 1  # c[1,1]=W[3]
        
        # M7 = (a01-a11)*(b10+b11) -> contributes to c00(+1)
        U[1, 6] = 1; U[3, 6] = -1  # a01, -a11
        V[2, 6] = 1; V[3, 6] = 1  # b10, b11
        W[0, 6] = 1  # c[0,0]=W[0]
        
        return U, V, W
    
    def build_strassen_4x4x4():
        """Build rank-49 Strassen decomposition for 4x4x4.
        U: (16, 49), V: (16, 49), W: (16, 49)
        A is 4x4, B is 4x4, C = A*B is 4x4.
        U[i*m+j] where m=4, V[j*p+k] where p=4, W[k*n+i] where n=4.
        """
        # Use recursive Strassen: 4x4 = 2x2 blocks, each 2x2 product uses rank-7
        # Top level: 7 multiplications of 2x2 matrices
        # Each 2x2 matrix multiplication: rank 7
        # Total: 7 * 7 = 49
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Get the base 2x2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2x2()
        
        # Top-level Strassen for 4x4 matrices:
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # 
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M_i is a 2x2 matrix product. For the tensor decomposition,
        # each M_i = L*R where L and R are 2x2 matrices (possibly sums/diffs of blocks).
        # The product L*R has a rank-7 decomposition.
        
        # For each top-level term, we need to:
        # 1. Determine which A blocks contribute to L (with signs)
        # 2. Determine which B blocks contribute to R (with signs)
        # 3. Apply the 2x2x2 Strassen decomposition
        # 4. Map the result to the appropriate C blocks with appropriate signs
        
        # Block positions in 4x4 matrix (0-indexed):
        # Block 0: rows 0-1, cols 0-1 (top-left)
        # Block 1: rows 0-1, cols 2-3 (top-right)
        # Block 2: rows 2-3, cols 0-1 (bottom-left)
        # Block 3: rows 2-3, cols 2-3 (bottom-right)
        
        # For A block b (0-3): row_offset = (b//2)*2, col_offset = (b%2)*2
        # For B block b (0-3): row_offset = (b//2)*2, col_offset = (b%2)*2
        # For C block b (0-3): row_offset = (b//2)*2, col_offset = (b%2)*2
        
        def block_offsets(b):
            """Return (row_offset, col_offset) for block b in a 4x4 matrix."""
            return ((b // 2) * 2, (b % 2) * 2)
        
        # For each top-level Strassen term, define:
        # - L_blocks: list of (block_index, sign) for A
        # - R_blocks: list of (block_index, sign) for B
        # - C_contrib: list of (block_index, sign) for C output
        
        top_terms = [
            # M1 = (A11+A22)*(B11+B22) -> C11(+1), C22(+1)
            {'L': [(0, 1), (3, 1)], 'R': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            # M2 = (A21+A22)*B11 -> C21(+1)
            {'L': [(2, 1), (3, 1)], 'R': [(0, 1)], 'C': [(2, 1)]},
            # M3 = A11*(B12-B22) -> C12(+1)
            {'L': [(0, 1)], 'R': [(1, 1), (3, -1)], 'C': [(1, 1)]},
            # M4 = A22*(B21-B11) -> C11(+1), C21(+1)
            {'L': [(3, 1)], 'R': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            # M5 = (A11+A12)*B22 -> C11(-1), C12(+1)
            {'L': [(0, 1), (1, 1)], 'R': [(3, 1)], 'C': [(0, -1), (1, 1)]},
            # M6 = (A21-A11)*(B11+B12) -> C22(+1)
            {'L': [(2, 1), (0, -1)], 'R': [(0, 1), (1, 1)], 'C': [(3, 1)]},
            # M7 = (A12-A22)*(B21+B22) -> C11(+1)
            {'L': [(1, 1), (3, -1)], 'R': [(2, 1), (3, 1)], 'C': [(0, 1)]},
        ]
        
        for t_idx, term in enumerate(top_terms):
            term_base = t_idx * 7
            
            # For each sub-term in the 2x2x2 Strassen decomposition:
            for sub_idx in range(7):
                global_term = term_base + sub_idx
                
                # Get the 2x2x2 Strassen coefficients
                # U2[:, sub_idx] gives coefficients for A elements
                # V2[:, sub_idx] gives coefficients for B elements
                # W2[:, sub_idx] gives coefficients for C elements
                
                # Map A coefficients:
                # U2 has 4 entries for 2x2 A: [a00, a01, a10, a11]
                # For each A block in L, add the corresponding U2 coefficients
                for a_block, a_sign in term['L']:
                    a_row_off, a_col_off = block_offsets(a_block)
                    # A block local indices (li, lj) -> global (a_row_off+li, a_col_off+lj)
                    # U2 index: li*2 + lj
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            global_idx = (a_row_off + li) * 4 + (a_col_off + lj)
                            U[global_idx, global_term] += U2[local_idx, sub_idx] * a_sign
                
                # Map B coefficients:
                # V2 has 4 entries for 2x2 B: [b00, b01, b10, b11]
                # For each B block in R, add the corresponding V2 coefficients
                for b_block, b_sign in term['R']:
                    b_row_off, b_col_off = block_offsets(b_block)
                    # B block local indices (lj, lk) -> global (b_row_off+lj, b_col_off+lk)
                    # V2 index: lj*2 + lk
                    for lj in range(2):
                        for lk in range(2):
                            local_idx = lj * 2 + lk
                            global_idx = (b_row_off + lj) * 4 + (b_col_off + lk)
                            V[global_idx, global_term] += V2[local_idx, sub_idx] * b_sign
                
                # Map C coefficients:
                # W2 has 4 entries for 2x2 C: [c00, c10, c01, c11] (W[k*2+i] ordering)
                # Actually W2 indices: W2[k*2+i] for c[i,k]
                # For each C block in C_contrib, add the corresponding W2 coefficients
                for c_block, c_sign in term['C']:
                    c_row_off, c_col_off = block_offsets(c_block)
                    # C block local indices (li, lk) -> global (c_row_off+li, c_col_off+lk)
                    # W2 index: lk*2 + li (since W[k*n+i] with n=2)
                    for li in range(2):
                        for lk in range(2):
                            local_idx = lk * 2 + li
                            global_idx = (c_col_off + lk) * 4 + (c_row_off + li)
                            W[global_idx, global_term] += W2[local_idx, sub_idx] * c_sign
        
        return U, V, W
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U,V,W give a valid decomposition of the matrix multiplication tensor."""
        nm = n * m
        mp = m * p
        pn = p * n
        rank = U.shape[1]
        
        # Compute C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # and check if it equals 1 for all i,k
        
        # Vectorized verification
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    total += np.dot(u_row, v_row) * W[k * n + i, :]
                if abs(total - 1.0) > 1e-8:
                    return False
        return True
    
    # Build the Strassen-based decomposition
    try:
        U_s4, V_s4, W_s4 = build_strassen_4x4x4()
        
        # Verify the 4x4x4 decomposition
        if verify_decomposition(U_s4, V_s4, W_s4, 4, 4, 4):
            # Now build 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
            # First 4x4x4: columns 0-3 of B
            # Second 4x4x4: columns 4-7 of B
            # 4x4x1: column 8 of B
            
            total_rank = 49 + 49 + 16  # 114
            U_4x4x9 = np.zeros((16, total_rank))
            V_4x4x9 = np.zeros((36, total_rank))
            W_4x4x9 = np.zeros((36, total_rank))
            
            # First 4x4x4 (B columns 0-3)
            U_4x4x9[:, 0:49] = U_s4
            V_4x4x9[0:16, 0:49] = V_s4  # B rows 0-3, cols 0-3
            W_4x4x9[0:16, 0:49] = W_s4  # C rows 0-3, cols 0-3 -> W[k*4+i] for k=0..3, i=0..3
            
            # Second 4x4x4 (B columns 4-7)
            U_4x4x9[:, 49:98] = U_s4
            V_4x4x9[16:32, 49:98] = V_s4  # B rows 0-3, cols 4-7
            W_4x4x9[16:32, 49:98] = W_s4  # C rows 0-3, cols 4-7
            
            # 4x4x1 (B column 8)
            # Schoolbook: 16 terms
            for i in range(4):
                for j in range(4):
                    col_idx = 98 + i * 4 + j
                    U_4x4x9[i * 4 + j, col_idx] = 1
                    V_4x4x9[j * 9 + 8, col_idx] = 1  # B[j, 8]
                    W_4x4x9[8 * 4 + i, col_idx] = 1  # C[i, 8] -> W[8*4+i]
            
            # Verify the full decomposition
            if verify_decomposition(U_4x4x9, V_4x4x9, W_4x4x9, 4, 4, 9):
                best_rank = total_rank
                best_U = U_4x4x9
                best_V = V_4x4x9
                best_W = W_4x4x9
                print(f"Strassen 4x4x9 decomposition verified with rank {total_rank}")
            else:
                print("Strassen 4x4x9 decomposition failed verification")
        else:
            print("Strassen 4x4x4 decomposition failed verification")
    except Exception as e:
        print(f"Error building Strassen decomposition: {e}")
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass  # Return best found
    
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
