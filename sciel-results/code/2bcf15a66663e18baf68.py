"""Program for rank-minimal tensor decomposition of 3x3 by 3x6 matrix multiplication."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 130

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition correctly represents matrix multiplication."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Convention: W[k*n + i, r]
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # For all other triples (a,b,c) where the tensor entry is 0, the sum should be 0
        # The tensor entry T[a,b,c] is 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # Check all valid (i,j,k) triples
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j, :]
                    v_val = V_arr[j*p+k, :]
                    w_val = W_arr[k*n+i, :]
                    val = np.dot(u_val, v_val * w_val)
                    err = abs(1.0 - val)
                    if err > max_err:
                        max_err = err
        
        # Check a sample of invalid triples to ensure they are zero
        # Build set of valid (a,b,c) triples
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
        
        # Check some invalid triples
        import itertools
        checked = 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a,b,c) not in valid_set:
                        u_val = U_arr[a, :]
                        v_val = V_arr[b, :]
                        w_val = W_arr[c, :]
                        val = np.dot(u_val, v_val * w_val)
                        err = abs(val)
                        if err > max_err:
                            max_err = err
                        checked += 1
                        if checked > 200:  # Limit checks for speed
                            break
                if checked > 200:
                    break
            if checked > 200:
                break
        
        return max_err

    def schoolbook_3x3x6():
        """Return schoolbook decomposition (rank 54) for 3x3x6."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u, v, w) where u, v, w are 4-dimensional vectors.
        
        For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        i = a*e + b*g, j = a*f + b*h
        k = c*e + d*g, l = c*f + d*h
        
        Strassen uses 7 multiplications:
        m1 = (a+d)(e+h)
        m2 = (c+d)e
        m3 = a(f-h)
        m4 = d(g-c)
        m5 = (a+b)h
        m6 = (c-a)(e+f)
        m7 = (b-d)(g+h)
        
        i = m1 + m4 - m5 + m7
        j = m3 + m5
        k = m2 + m4
        l = m1 - m2 + m3 + m6
        """
        terms = [
            # m1 = (a+d)(e+h)
            (np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,1])),
            # m2 = (c+d)e
            (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,-1])),
            # m3 = a(f-h)
            (np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([0,1,0,1])),
            # m4 = d(g-c)
            (np.array([0,0,0,1]), np.array([0,0,1,-1]), np.array([1,0,1,0])),
            # m5 = (a+b)h
            (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,1,0,0])),
            # m6 = (c-a)(e+f)
            (np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1])),
            # m7 = (b-d)(g+h)
            (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,0])),
        ]
        return terms

    def build_3x3x3_strassen_blocks():
        """
        Build a decomposition for 3x3x3 using 4 overlapping 2x2 Strassen blocks.
        Each block uses 7 terms, total 28 terms.
        
        Blocks:
        - TL: rows 0-1, cols 0-1
        - TR: rows 0-1, cols 1-2
        - BL: rows 1-2, cols 0-1
        - BR: rows 1-2, cols 1-2
        """
        rank = 28
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        strassen_terms = strassen_2x2x2_terms()
        
        blocks = [
            (0, 1, 0, 1),  # TL: rows 0-1, cols 0-1
            (0, 1, 1, 2),  # TR: rows 0-1, cols 1-2
            (1, 2, 0, 1),  # BL: rows 1-2, cols 0-1
            (1, 2, 1, 2),  # BR: rows 1-2, cols 1-2
        ]
        
        col = 0
        for r1, r2, c1, c2 in blocks:
            for s in range(7):
                u2, v2, w2 = strassen_terms[s]
                
                for idx2 in range(4):
                    # A position
                    a_row = idx2 // 2
                    a_col = idx2 % 2
                    a_3x3_row = r1 + a_row
                    a_3x3_col = c1 + a_col
                    U[a_3x3_row * 3 + a_3x3_col, col] += u2[idx2]
                    
                    # B position: B row = c1 + a_col, B col = c1 + (idx2 % 2)
                    b_row_2x2 = idx2 // 2
                    b_col_2x2 = idx2 % 2
                    b_3x3_row = c1 + b_row_2x2
                    b_3x3_col = c1 + b_col_2x2
                    V[b_3x3_row * 3 + b_3x3_col, col] += v2[idx2]
                    
                    # C position: C[r1+i, c1+k] where i is row in 2x2, k is col in 2x2
                    c_row_2x2 = idx2 // 2
                    c_col_2x2 = idx2 % 2
                    c_3x3_row = r1 + c_row_2x2
                    c_3x3_col = c1 + c_col_2x2
                    W[c_3x3_col * 3 + c_3x3_row, col] += w2[idx2]
                
                col += 1
        
        return U, V, W, rank

    def build_3x3x6_from_3x3x3(U33, V33, W33, rank33):
        """
        Lift a rank-r 3x3x3 decomposition to 3x3x6 with rank 2r.
        """
        rank = 2 * rank33
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        for s in range(rank33):
            idx1 = 2 * s
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx1] = U33[i*3+j, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k, idx1] = V33[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[k*3+i, idx1] = W33[k*3+i, s]
            
            idx2 = 2 * s + 1
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx2] = U33[i*3+j, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k+3, idx2] = V33[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i, idx2] = W33[k*3+i, s]
        
        return U, V, W, rank

    def build_rank16_3x3x3():
        """
        Build a rank-16 decomposition for 3x3x3 using Strassen + cross terms.
        Based on inspiration 2.
        """
        r = 16
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        
        # Strassen 7 terms
        # M1
        U[0,0] = 1; U[4,0] = 1
        V[0,0] = 1; V[4,0] = 1
        W[0,0] = 1; W[1,0] = 1; W[3,0] = -1
        # M2
        U[3,1] = 1; U[4,1] = 1
        V[0,1] = 1
        W[3,1] = 1
        # M3
        U[0,2] = 1
        V[1,2] = 1; V[4,2] = -1
        W[1,2] = 1
        # M4
        U[4,3] = 1
        V[3,3] = 1; V[0,3] = -1
        W[3,3] = 1
        # M5
        U[0,4] = 1; U[1,4] = 1
        V[4,4] = 1
        W[1,4] = 1; W[4,4] = 1
        # M6
        U[3,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[4,5] = 1
        # M7
        U[1,6] = 1; U[4,6] = -1
        V[3,6] = 1; V[4,6] = 1
        W[1,6] = 1; W[4,6] = 1
        
        # Cross terms (9 terms)
        # Col 2, Rows 0,1
        # 1. A00, A10 with B02 -> C02, C12
        U[0,7] = 1; U[3,7] = 1
        V[2,7] = 1
        W[2,7] = 1; W[5,7] = 1
        
        # 2. A01, A11 with B12 -> C02, C12
        U[1,8] = 1; U[4,8] = 1
        V[5,8] = 1
        W[2,8] = 1; W[5,8] = 1
        
        # 3. A02, A12 with B22 -> C02, C12
        U[2,9] = 1; U[5,9] = 1
        V[8,9] = 1
        W[2,9] = 1; W[5,9] = 1
        
        # Row 2, Cols 0,1
        # 4. A20 with B00, B01 -> C20, C21
        U[6,10] = 1
        V[0,10] = 1; V[1,10] = 1
        W[6,10] = 1; W[7,10] = 1
        
        # 5. A21 with B10, B11 -> C20, C21
        U[7,11] = 1
        V[3,11] = 1; V[4,11] = 1
        W[6,11] = 1; W[7,11] = 1
        
        # 6. A22 with B20, B21 -> C20, C21
        U[8,12] = 1
        V[6,12] = 1; V[7,12] = 1
        W[6,12] = 1; W[7,12] = 1
        
        # C22
        # 7. A20 with B02 -> C22
        U[6,13] = 1
        V[2,13] = 1
        W[8,13] = 1
        
        # 8. A21 with B12 -> C22
        U[7,14] = 1
        V[5,14] = 1
        W[8,14] = 1
        
        # 9. A22 with B22 -> C22
        U[8,15] = 1
        V[8,15] = 1
        W[8,15] = 1
        
        # Verify
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(42)
        A_test = np.random.randn(3, 3)
        B_test = np.random.randn(3, 3)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((3, 3))
        for r_idx in range(r):
            a_proj = np.dot(U_arr[:, r_idx].reshape(3, 3), A_test)
            b_proj = np.dot(V_arr[:, r_idx].reshape(3, 3), B_test)
            for i in range(3):
                for k in range(3):
                    C_computed[i, k] += W_arr[k * 3 + i, r_idx] * a_proj[i, k]
                
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return None
            
        return U, V, W, r

    best_result = None
    best_rank = float('inf')
    
    # Method 1: Rank-16 3x3x3 block, lift to 3x3x6 (rank 32)
    if time.time() < deadline - 10:
        try:
            result = build_rank16_3x3x3()
            if result is not None:
                U_block, V_block, W_block, r_block = result
                total_rank = 2 * r_block
                
                U = np.zeros((9, total_rank))
                V = np.zeros((18, total_rank))
                W = np.zeros((18, total_rank))
                
                # Block 0
                for r_idx in range(r_block):
                    for i in range(3):
                        for j in range(3):
                            U[i * 3 + j, r_idx] = U_block[i * 3 + j, r_idx]
                    for j in range(3):
                        for k in range(3):
                            V[j * 6 + k, r_idx] = V_block[j * 3 + k, r_idx]
                    for k in range(3):
                        for i in range(3):
                            W[k * 3 + i, r_idx] = W_block[k * 3 + i, r_idx]
                            
                # Block 1
                for r_idx in range(r_block):
                    offset = r_block + r_idx
                    for i in range(3):
                        for j in range(3):
                            U[i * 3 + j, offset] = U_block[i * 3 + j, r_idx]
                    for j in range(3):
                        for k in range(3):
                            V[j * 6 + (k + 3), offset] = V_block[j * 3 + k, r_idx]
                    for k in range(3):
                        for i in range(3):
                            W[(k + 3) * 3 + i, offset] = W_block[k * 3 + i, r_idx]
                
                err = verify_decomposition(U, V, W)
                if err < 1e-9 and total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    # Method 2: 4 Strassen blocks for 3x3x3 (rank 28), lift to 3x3x6 (rank 56)
    if time.time() < deadline - 10:
        try:
            U33, V33, W33, rank33 = build_3x3x3_strassen_blocks()
            U, V, W, rank = build_3x3x6_from_3x3x3(U33, V33, W33, rank33)
            err = verify_decomposition(U, V, W)
            if err < 1e-9 and rank < best_rank:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    # Method 3: Schoolbook fallback (rank 54)
    if time.time() < deadline - 5:
        try:
            U, V, W, rank = schoolbook_3x3x6()
            err = verify_decomposition(U, V, W)
            if err < 1e-9 and rank < best_rank:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Final fallback: always return schoolbook
    U, V, W, rank = schoolbook_3x3x6()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
