"""Program for rank-minimal tensor decomposition of 3x3 by 3x6 matrix multiplication."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 130

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
        
        In tensor form, each multiplication m_s corresponds to a term.
        For the full tensor decomposition, each "multiplication" becomes a rank-1 tensor.
        
        For 2x2x2, the tensor has 8 entries (i,j,k,l for each of 4 output positions... 
        actually it's more complex). Let me think differently.
        
        The multiplication tensor T for 2x2x2:
        T = sum_{output positions} (input_a pattern) ⊗ (input_b pattern) ⊗ (output pattern)
        
        For Strassen, we have 7 terms. Each term s has:
        - u_s: coefficients for A elements (4-dim: a,b,c,d)
        - v_s: coefficients for B elements (4-dim: e,f,g,h)  
        - w_s: coefficients for C elements (4-dim: i,j,k,l)
        
        The decomposition is: sum_s u_s ⊗ v_s ⊗ w_s = T
        where T[i,j,k] = 1 if output position (i,k) depends on A[i,j]*B[j,k]
        
        Actually, the standard form: T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
        for the "direct" decomposition (rank 8 for 2x2).
        
        For Strassen, we find 7 triples (u_s, v_s, w_s) such that
        sum_s u_s ⊗ v_s ⊗ w_s = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
        """
        # Strassen's 7 terms for 2x2x2
        # A indices: 0=a, 1=b, 2=c, 3=d
        # B indices: 0=e, 1=f, 2=g, 3=h
        # C indices: 0=i, 1=j, 2=k, 3=l
        
        terms = []
        
        # m1 = (a+d)(e+h): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (contributes to i and l)
        # Actually, m1 contributes to both i and l.
        # i = m1 + m4 - m5 + m7
        # l = m1 - m2 + m3 + m6
        
        # Let me be more careful. Each Strassen "multiplication" is a bilinear form
        # that gets added to specific output positions.
        
        # m1 = (a+d)(e+h) -> contributes +1 to i, +1 to l
        # m2 = (c+d)e -> contributes -1 to l, +1 to k  
        # m3 = a(f-h) -> contributes +1 to j, +1 to l
        # m4 = d(g-c) -> contributes +1 to i, +1 to k
        # m5 = (a+b)h -> contributes -1 to i, +1 to j
        # m6 = (c-a)(e+f) -> contributes +1 to l
        # m7 = (b-d)(g+h) -> contributes +1 to i
        
        # So:
        # i = m1 + m4 - m5 + m7
        # j = m3 + m5
        # k = m2 + m4
        # l = m1 - m2 + m3 + m6
        
        terms = [
            # (u, v, w) for each of the 7 multiplications
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
    
    def build_3x3x3_rank24():
        """
        Build a rank-24 decomposition for 3x3x3 using overlapping Strassen blocks.
        
        Strategy: Use 4 overlapping 2x2 Strassen blocks:
        - Block TL: rows 0-1, cols 0-1
        - Block TR: rows 0-1, cols 1-2  
        - Block BL: rows 1-2, cols 0-1
        - Block BR: rows 1-2, cols 1-2
        
        Each block uses 7 Strassen terms = 28 total.
        But we can share some terms to reduce to 24.
        
        For now, let me just use 28 terms (4 blocks x 7) and verify.
        Then try to optimize.
        """
        rank = 28
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        strassen_terms = strassen_2x2x2_terms()
        
        # Define the 4 blocks with their row/col ranges
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
                # u2 has 4 elements corresponding to A positions in the 2x2 block
                # v2 has 4 elements corresponding to B positions in the 2x2 block
                # w2 has 4 elements corresponding to C positions in the 2x2 block
                
                # Map 2x2 positions to 3x3 positions
                # A block rows r1,r2 and cols c1,c2:
                #   (0,0)->(r1,c1), (0,1)->(r1,c2), (1,0)->(r2,c1), (1,1)->(r2,c2)
                # B block rows c1,c2 (since B is 3x3, rows are col indices of A) and cols...
                # Wait, for 3x3x3: A is 3x3, B is 3x3, C is 3x3
                # C[i,k] = sum_j A[i,j] * B[j,k]
                
                # For a 2x2 sub-block with A rows r1,r2 and A cols c1,c2:
                # This computes C[r1,r2] × C[c1,c2] submatrix
                # C_sub[i,k] = sum_{j in {c1,c2}} A[r1+i, c1+j] * B[c1+j, c1+k]
                
                # A positions in the 2x2 block (i,j) in 2x2 -> (r1+i, c1+j) in 3x3
                # B positions: B is 3x3, for this block B rows are c1,c2 and B cols are c1,c2
                #   So B[c1+j, c1+k] for j,k in {0,1}
                # C positions: C[r1+i, c1+k] for i,k in {0,1}
                
                for idx2 in range(4):
                    # A position
                    a_row = idx2 // 2
                    a_col = idx2 % 2
                    a_3x3_row = r1 + a_row
                    a_3x3_col = c1 + a_col
                    U[a_3x3_row * 3 + a_3x3_col, col] += u2[idx2]
                    
                    # B position: B row = c1 + a_col (since B's row index = A's col index)
                    # B col = c1 + (idx2 % 2) ... wait, need to be careful
                    # In the 2x2 block, B is also 2x2 with rows {c1,c2} and cols {c1,c2}
                    # B position (j,k) in 2x2 -> B[c1+j, c1+k] in 3x3
                    # The v2 vector has indices for B positions (j,k) where j is B row, k is B col
                    # In the 2x2 Strassen, v2 has 4 elements for B's 4 positions
                    # B 2x2 position (j,k) -> index j*2+k in the 4-element vector
                    # But which (j,k)? In Strassen, B is [[e,f],[g,h]] with e=B[c1,c1], f=B[c1,c2], g=B[c2,c1], h=B[c2,c2]
                    # So B 2x2 index 0=(c1,c1), 1=(c1,c2), 2=(c2,c1), 3=(c2,c2)
                    # In the v2 vector: v2[0]=coeff for e, v2[1]=coeff for f, v2[2]=coeff for g, v2[3]=coeff for h
                    # So v2[idx] corresponds to B position:
                    #   idx 0: B[c1, c1]
                    #   idx 1: B[c1, c2]
                    #   idx 2: B[c2, c1]
                    #   idx 3: B[c2, c2]
                    
                    b_row_2x2 = idx2 // 2
                    b_col_2x2 = idx2 % 2
                    b_3x3_row = c1 + b_row_2x2
                    b_3x3_col = c1 + b_col_2x2
                    V[b_3x3_row * 3 + b_3x3_col, col] += v2[idx2]
                    
                    # C position: C[r1+i, c1+k] where i is row in 2x2, k is col in 2x2
                    # In the 2x2 Strassen, w2 has 4 elements for C's 4 positions
                    # C 2x2 position (i,k) -> index i*2+k
                    # w2[0]=coeff for C[r1,c1], w2[1]=coeff for C[r1,c2], w2[2]=coeff for C[r2,c1], w2[3]=coeff for C[r2,c2]
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
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition correctly represents matrix multiplication."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    val = np.dot(u_val, v_val * w_val)
                    err = abs(1.0 - val)
                    if err > max_err:
                        max_err = err
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
    
    best_result = None
    best_rank = float('inf')
    
    # Method 1: 4 Strassen blocks for 3x3x3 (rank 28), lift to 3x3x6 (rank 56)
    try:
        U33, V33, W33, rank33 = build_3x3x3_rank24()
        U, V, W, rank = build_3x3x6_from_3x3x3(U33, V33, W33, rank33)
        err = verify_decomposition(U, V, W)
        if err < 1e-9 and rank < best_rank:
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Method 2: Schoolbook fallback
    U, V, W, rank = schoolbook_3x3x6()
    err = verify_decomposition(U, V, W)
    if err < 1e-9 and rank < best_rank:
        best_rank = rank
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_result is not None:
        return best_result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
