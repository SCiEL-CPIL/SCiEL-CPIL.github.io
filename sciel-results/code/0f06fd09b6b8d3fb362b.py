"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 145  # 145s budget, 5s safety margin before 150s kill

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition is exact using vectorized operations."""
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]  # shape (m, rank)
                V_cols = np.array([V[j*p + k, :] for j in range(m)])  # shape (m, rank)
                W_val = W[k*n + i, :]  # shape (rank,)
                
                temp = U_slice * W_val  # shape (m, rank)
                result = temp @ V_cols.T  # shape (m, m)
                
                expected = np.eye(m)
                if np.max(np.abs(result - expected)) > tol:
                    return False
        return True

    def strassen_2x2():
        """
        Rank-7 tensor decomposition for 2x2 matrix multiplication.
        A (2x2) times B (2x2) = C (2x2)
        Input indices: u = row*2 + col for A, v = row*2 + col for B
        Output indices: w = row*2 + col for C
        """
        U = np.zeros((4, 7))  # A entries: a,b,c,d -> indices 0,1,2,3
        V = np.zeros((4, 7))  # B entries: e,f,g,h -> indices 0,1,2,3
        W = np.zeros((4, 7))  # C entries: p,q,r,s -> indices 0,1,2,3
        
        # M1 = (a+d)(e+g)
        U[0, 0] = 1; U[3, 0] = 1  # a, d
        V[0, 0] = 1; V[2, 0] = 1  # e, g
        W[0, 0] = 1; W[3, 0] = 1  # p, s
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1  # c, d
        V[0, 1] = 1  # e
        W[2, 1] = 1; W[3, 1] = -1  # r, s
        
        # M3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1; V[3, 2] = -1  # f, h
        W[1, 2] = 1; W[3, 2] = 1  # q, s
        
        # M4 = d(g-e)
        U[3, 3] = 1  # d
        V[2, 3] = 1; V[0, 3] = -1  # g, e
        W[0, 3] = 1; W[2, 3] = 1  # p, r
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1  # a, b
        V[3, 4] = 1  # h
        W[0, 4] = -1; W[1, 4] = 1  # p, q
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1  # c, -a
        V[0, 5] = 1; V[1, 5] = 1  # e, f
        W[3, 5] = -1  # s
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1  # b, -d
        V[2, 6] = 1; V[3, 6] = 1  # g, h
        W[0, 6] = 1  # p
        
        return U, V, W

    def strassen_4x4():
        """
        Rank-49 tensor decomposition for 4x4 matrix multiplication using block Strassen.
        A (4x4) times B (4x4) = C (4x4)
        """
        U2, V2, W2 = strassen_2x2()
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # 4x4 partitioned into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        #
        # Block-level Strassen products:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # Block (bi, bj) of a 4x4 matrix:
        #   A entry (r, c) -> index r*4 + c
        #   Block (bi, bj) covers rows [2*bi, 2*bi+1], cols [2*bj, 2*bj+1]
        #   Local (lr, lc) in block -> global row 2*bi+lr, col 2*bj+lc -> index (2*bi+lr)*4 + (2*bj+lc)
        
        # For each block-level Strassen product k (0..6), we need to compute
        # (linear combo of A blocks) * (linear combo of B blocks)
        # Each such product is a 2x2 * 2x2 multiplication, rank 7.
        # Total: 7 * 7 = 49 rank-1 terms.
        
        # Define block-level Strassen coefficients
        # A_blocks: A00, A01, A10, A11 (indices 0,1,2,3)
        # B_blocks: B00, B01, B10, B11 (indices 0,1,2,3)
        
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # For each block product k, the A-side is a linear combination of 2x2 blocks
        # and the B-side is a linear combination of 2x2 blocks.
        # The product (sum_a a_s * A_s) * (sum_b b_t * B_t) = sum_{s,t} a_s * b_t * A_s * B_t
        # Each A_s * B_t is a 2x2 * 2x2 product, rank 7.
        
        # But we can be smarter: (sum_a a_s * A_s) is itself a 2x2 matrix,
        # and (sum_b b_t * B_t) is itself a 2x2 matrix.
        # Their product is a 2x2 * 2x2 multiplication, which has rank 7.
        
        # So for block product k with A-coefficients a_k (4 values for 4 blocks)
        # and B-coefficients b_k (4 values), we compute:
        # A_combined = sum_s a_k[s] * A_s  (a 2x2 matrix)
        # B_combined = sum_t b_k[t] * B_t  (a 2x2 matrix)
        # C_block = A_combined * B_combined (rank 7 via Strassen)
        
        # The 4 entries of A_combined are linear combinations of the 16 entries of A.
        # Similarly for B_combined.
        
        # For U: entry (r, c) of A has coefficient sum over s where A_s contains (r,c) of a_k[s]
        # Block A_s = A_{s//2, s%2} covers rows [2*(s//2), 2*(s//2)+1], cols [2*(s%2), 2*(s%2)+1]
        # Entry (r, c) is in block A_s iff s//2 == r//2 and s%2 == c//2, i.e., s = (r//2)*2 + (c//2)
        
        # So for entry (r, c) of A, the only block containing it is s = (r//2)*2 + (c//2)
        # And the coefficient is a_k[s].
        
        # Similarly for B: entry (r, c) of B is in block B_t where t = (r//2)*2 + (c//2)
        # And the coefficient is b_k[t].
        
        # Now, A_combined * B_combined is a 2x2 * 2x2 product.
        # Let A_combined = [[a11, a12], [a21, a22]] where a11 = A_combined[0,0], etc.
        # Let B_combined = [[b11, b12], [b21, b22]]
        
        # The Strassen decomposition for A_combined * B_combined:
        # U2[l] gives the coefficient of A_combined entry l (0..3) in Strassen product k2
        # V2[l] gives the coefficient of B_combined entry l (0..3) in Strassen product k2
        # W2[l] gives the coefficient of C_block entry l (0..3) in Strassen product k2
        
        # So for the full 4x4 * 4x4:
        # U[entry(r,c), col] = U2[local_entry] * a_k[block_of(r,c)]
        # where local_entry is the position of (r,c) within its 2x2 block
        
        # Block of (r,c) in A: s = (r//2)*2 + (c//2)
        # Local position in block: lr = r % 2, lc = c % 2, local_entry = lr*2 + lc
        
        # Similarly for V:
        # V[entry(r,c), col] = V2[local_entry] * b_k[block_of(r,c)]
        
        # For W:
        # W[entry(r,c), col] = W2[local_entry] * c_k[block_of(r,c)]
        # where c_k is the coefficient of C_block in the final output
        
        # Block-level Strassen:
        a_coeffs = [
            [1, 0, 0, 1],  # M1: A00 + A11
            [0, 0, 1, 1],  # M2: A10 + A11
            [1, 0, 0, 0],  # M3: A00
            [0, 0, 0, 1],  # M4: A11
            [1, 1, 0, 0],  # M5: A00 + A01
            [-1, 0, 1, 0], # M6: A10 - A00
            [0, 1, 0, -1], # M7: A01 - A11
        ]
        b_coeffs = [
            [1, 0, 0, 1],  # M1: B00 + B11
            [1, 0, 0, 0],  # M2: B00
            [0, 1, 0, -1], # M3: B01 - B11
            [0, 0, 1, -1], # M4: B10 - B00
            [0, 0, 0, 1],  # M5: B11
            [1, 1, 0, 0],  # M6: B00 + B01
            [0, 0, 1, 1],  # M7: B10 + B11
        ]
        # C output coefficients:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        c_coeffs = [
            [1, 0, 0, 1],  # C00: M1, M4, -M5, M7
            [0, 1, 0, 0],  # C01: M3, M5
            [0, 0, 1, 0],  # C10: M2, M4
            [1, -1, 1, -1],# C11: M1, -M2, M3, -M6
        ]
        # Wait, let me re-check:
        # C00 = M1 + M4 - M5 + M7 -> coeffs for M1..M7: [1, 0, 0, 1, -1, 0, 1]
        # C01 = M3 + M5 -> [0, 0, 1, 0, 1, 0, 0]
        # C10 = M2 + M4 -> [0, 1, 0, 1, 0, 0, 0]
        # C11 = M1 - M2 + M3 - M6 -> [1, -1, 1, 0, 0, -1, 0]
        
        c_coeffs = [
            [1, 0, 0, 1, -1, 0, 1],  # C00
            [0, 0, 1, 0, 1, 0, 0],   # C01
            [0, 1, 0, 1, 0, 0, 0],   # C10
            [1, -1, 1, 0, 0, -1, 0], # C11
        ]
        
        for k in range(7):  # block-level Strassen product
            a_k = a_coeffs[k]  # 4 values for A00, A01, A10, A11
            b_k = b_coeffs[k]  # 4 values for B00, B01, B10, B11
            
            for k2 in range(7):  # 2x2 Strassen product within each block product
                col = k * 7 + k2
                
                # Fill U: for each entry (r, c) of 4x4 A
                for r in range(4):
                    for c in range(4):
                        s = (r // 2) * 2 + (c // 2)  # block index
                        lr = r % 2
                        lc = c % 2
                        local_entry = lr * 2 + lc
                        U[r * 4 + c, col] = U2[local_entry] * a_k[s]
                
                # Fill V: for each entry (r, c) of 4x4 B
                for r in range(4):
                    for c in range(4):
                        t = (r // 2) * 2 + (c // 2)  # block index
                        lr = r % 2
                        lc = c % 2
                        local_entry = lr * 2 + lc
                        V[r * 4 + c, col] = V2[local_entry] * b_k[t]
                
                # Fill W: for each entry (r, c) of 4x4 C
                for r in range(4):
                    for c in range(4):
                        w_block = (r // 2) * 2 + (c // 2)  # block index in C
                        lr = r % 2
                        lc = c % 2
                        local_entry = lr * 2 + lc
                        W[r * 4 + c, col] = W2[local_entry] * c_coeffs[w_block][k]
        
        return U, V, W

    def assemble_4x4_4x9():
        """
        Assemble 4x4 * 4x9 decomposition.
        Split 4x9 into two 4x4 blocks and one 4x1 block.
        Use Strassen rank-49 for each 4x4*4x4 subproblem.
        Use schoolbook rank-16 for the 4x4*4x1 subproblem.
        Total rank: 2*49 + 16 = 114.
        """
        U_4x4, V_4x4, W_4x4 = strassen_4x4()
        rank_total = 2 * 49 + 16  # 114
        U = np.zeros((16, rank_total))
        V = np.zeros((36, rank_total))  # m*p = 4*9 = 36
        W = np.zeros((36, rank_total))  # p*n = 9*4 = 36
        
        # B is 4x9, split into B[:, 0:4] (4x4) and B[:, 4:8] (4x4) and B[:, 8] (4x1)
        # C is 9x4, split into C[0:4, :] (4x4), C[4:8, :] (4x4), C[8, :] (4x1)
        
        # First 4x4*4x4: A * B[:, 0:4] -> C[0:4, :]
        # U: same as U_4x4
        # V: B[:, 0:4] entries are at indices 0..15 (row*4+col for 4x4)
        # W: C[0:4, :] entries are at indices 0..15 (row*4+col for 4x4)
        U[:, 0:49] = U_4x4
        V[0:16, 0:49] = V_4x4
        W[0:16, 0:49] = W_4x4
        
        # Second 4x4*4x4: A * B[:, 4:8] -> C[4:8, :]
        # U: same as U_4x4
        # V: B[:, 4:8] entries are at indices 16..31 (row*4+col+16 for 4x4)
        # W: C[4:8, :] entries are at indices 16..31 (row*4+col+16 for 4x4)
        U[:, 49:98] = U_4x4
        V[16:32, 49:98] = V_4x4
        W[16:32, 49:98] = W_4x4
        
        # Third: A * B[:, 8] -> C[8, :]
        # Schoolbook for 4x4 * 4x1: rank 16
        # A entry (i, j) at index i*4+j
        # B entry (j, 8) at index j*4+8
        # C entry (8, i) at index 8*4+i = 32+i
        col = 98
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, col] = 1
                V[j * 4 + 8, col] = 1
                W[8 * 4 + i, col] = 1
                col += 1
        
        return U, V, W

    def try_optimized_rank(r, num_restarts=10):
        """Try to find a valid decomposition with rank r using random initialization + lstsq."""
        if r < 36:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(restart * 12345 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            # Pre-compute A: A[(u,v), c] = U[u,c] * V[v,c]
            # Shape: (n*m * m*p, r) = (576, r)
            A = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, r)
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                i_out = w % n  # output column index in n
                k_out = w // n  # output row index in p
                
                b = np.zeros(n * m * m * p)
                for j in range(m):
                    u = i_out * m + j
                    v = j * p + k_out
                    b[u * (m * p) + v] = 1.0
                
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    valid = False
                    break
            
            if not valid:
                continue
            
            if check_validity(U, V, W):
                return U, V, W, r
        
        return None

    # Get schoolbook solution
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try the Strassen-based assembly (rank 114)
    if time.time() < deadline - 30:
        U_str, V_str, W_str = assemble_4x4_4x9()
        rank_str = U_str.shape[1]
        if check_validity(U_str, V_str, W_str):
            best_U, best_V, best_W, best_rank = U_str, V_str, W_str, rank_str
        else:
            # Fallback to schoolbook if Strassen assembly is invalid
            pass
    
    # Try lower ranks with random search
    for r in range(36, best_rank):
        if time.time() > deadline - 15:
            break
        
        result = try_optimized_rank(r, num_restarts=5)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
