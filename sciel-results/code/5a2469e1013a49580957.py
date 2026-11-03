def _generate_strassen_4x4():
    """
    Generates U, V, W for 4x4 * 4x4 matrix multiplication using Strassen's algorithm.
    Returns U, V, W of shape (16, 49), (16, 49), (16, 49) respectively.
    Indices:
    U: i*4 + j
    V: j*4 + k
    W: k*4 + i
    """
    n = 4
    R = 49
    U = np.zeros((n*n, R))
    V = np.zeros((n*n, R))
    W = np.zeros((n*n, R))
    
    # We will use the recursive Strassen structure.
    # Strassen 2x2 * 2x2 uses 7 multiplications.
    # 4x4 * 4x4 uses 7 "super" multiplications, each of which is 2x2 * 2x2.
    # So we have 7 groups of 7 scalar multiplications.
    
    # Let's define the 7 super-multiplications M_1 ... M_7
    # Each M_l is computed as X_l * Y_l where X_l, Y_l are 2x2 matrices.
    # X_l = sum_{a,b} alpha_{l,a,b} A_{a,b}
    # Y_l = sum_{c,d} beta_{l,c,d} B_{c,d}
    # Where A_{a,b} is a 2x2 block.
    
    # It is easier to just generate the 2x2 Strassen and "blow it up".
    # A 4x4 matrix is 4 blocks of 2x2.
    # Let's index the 2x2 blocks as (0,0), (0,1), (1,0), (1,1).
    # Let's call them A00, A01, A10, A11.
    
    # Strassen 2x2 (of blocks) coefficients:
    # M1: (A00+A11) * (B00+B11)
    # M2: (A10+A11) * B00
    # M3: A00 * (B01-B11)
    # M4: A11 * (B10-B00)
    # M5: (A00+A01) * B11
    # M6: (A10-A00) * (B00+B01)
    # M7: (A01-A11) * (B10+B11)
    
    # We need to expand these into scalar multiplications.
    # Each block multiplication (e.g., A00 * B00) is a 2x2 * 2x2 multiplication.
    # We can use the 7-term Strassen for each block multiplication.
    # Or, we can just use the 4-term schoolbook for the inner 2x2 * 2x2 to keep it simple?
    # No, we want minimal rank. Using Strassen for the inner blocks is better.
    # Rank = 7 * 7 = 49.
    
    # Let's define a helper to get the U,V,W for a 2x2 * 2x2 Strassen.
    def strassen_2x2():
        # Returns U, V, W for 2x2 * 2x2.
        # U: (4, 7), V: (4, 7), W: (4, 7)
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11)
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[3, 0] = 1 # C00 += M1, C11 += M1
        
        # M2: (A10+A11)B00
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        w[2, 1] = 1 # C10 += M2
        
        # M3: A00(B01-B11)
        u[0, 2] = 1
        v[1, 2] = 1; v[3, 2] = -1
        w[1, 2] = 1 # C01 += M3
        
        # M4: A11(B10-B00)
        u[3, 3] = 1
        v[2, 3] = 1; v[0, 3] = -1
        w[2, 3] = 1 # C10 += M4
        w[0, 3] = 1 # C00 += M4
        
        # M5: (A00+A01)B11
        u[0, 4] = 1; u[1, 4] = 1
        v[3, 4] = 1
        w[0, 4] = -1 # C00 -= M5
        w[1, 4] = 1 # C01 += M5
        
        # M6: (A10-A00)(B00+B01)
        u[2, 5] = 1; u[0, 5] = -1
        v[0, 5] = 1; v[1, 5] = 1
        w[1, 5] = 1 # C11 += M6
        
        # M7: (A01-A11)(B10+B11)
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        w[0, 6] = 1 # C00 += M7
        w[1, 6] = -1 # C11 -= M7
        
        # Verify C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check W entries:
        # w[0,0]=1 (M1), w[3,0]=1 (M1) -> C00+=M1, C11+=M1. Correct.
        # w[2,1]=1 (M2) -> C10+=M2. Correct.
        # w[1,2]=1 (M3) -> C01+=M3. Correct.
        # w[2,3]=1 (M4), w[0,3]=1 (M4) -> C10+=M4, C00+=M4. Correct.
        # w[0,4]=-1 (M5), w[1,4]=1 (M5) -> C00-=M5, C01+=M5. Correct.
        # w[1,5]=1 (M6) -> C11+=M6. Correct.
        # w[0,6]=1 (M7), w[1,6]=-1 (M7) -> C00+=M7, C11-=M7. Correct.
        
        return u, v, w

    u2, v2, w2 = strassen_2x2()
    
    # Now, for 4x4, we have 7 super-steps.
    # Each super-step l involves adding/subtracting blocks of A and B.
    # Let's define the block coefficients for the 7 super-steps.
    # Blocks are indexed 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    
    # M1: (A0+A3)(B0+B3)
    # M2: (A2+A3)B0
    # M3: A0(B1-B3)
    # M4: A3(B2-B0)
    # M5: (A0+A1)B3
    # M6: (A2-A0)(B0+B1)
    # M7: (A1-A3)(B2+B3)
    
    block_coeffs_U = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: 1}, # M2
        {0: 1},       # M3
        {3: 1},       # M4
        {0: 1, 1: 1}, # M5
        {2: 1, 0: -1},# M6
        {1: 1, 3: -1} # M7
    ]
    
    block_coeffs_V = [
        {0: 1, 3: 1}, # M1
        {0: 1},       # M2
        {1: 1, 3: -1},# M3
        {2: 1, 0: -1},# M4
        {3: 1},       # M5
        {0: 1, 1: 1}, # M6
        {2: 1, 3: 1}  # M7
    ]
    
    block_coeffs_W = [
        # M1 contributes to C0 (A00 part) and C3 (A11 part)?
        # C = [C00 C01; C10 C11]
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        {0: 1, 3: 1}, # M1
        {2: 1},       # M2
        {1: 1},       # M3
        {0: 1, 2: 1}, # M4
        {0: -1, 1: 1},# M5
        {3: 1},       # M6
        {0: 1, 3: -1} # M7
    ]
    
    # Now we iterate over the 7 super-steps.
    # For each super-step l, we have a "virtual" 2x2 * 2x2 multiplication.
    # The inputs are linear combinations of the 2x2 blocks of A and B.
    # The output is a linear combination of the 2x2 blocks of C.
    # We replace each "block multiplication" with the 2x2 Strassen decomposition.
    
    for l in range(7):
        # Get the 2x2 Strassen U,V,W for this super-step
        # The "A" in the 2x2 Strassen corresponds to the block combination for U.
        # The "B" corresponds to the block combination for V.
        # The "C" corresponds to the block combination for W.
        
        # We need to map the 2x2 Strassen indices to the 4x4 indices.
        # The 2x2 Strassen u2 has shape (4, 7).
        # u2[i, k] where i is 0..3 (index in 2x2 block) and k is 0..6 (term in Strassen).
        # But wait, the 2x2 Strassen u2 is for a single 2x2 matrix.
        # Here, the "2x2 matrix" is actually a 2x2 block of 2x2 matrices.
        # So the "elements" of the virtual 2x2 matrix are 2x2 blocks.
        
        # Let's be precise.
        # The virtual 2x2 matrix $X$ has elements $X_{ab}$ which are 2x2 matrices.
        # $X_{ab} = \sum_{block} \alpha_{l, block} A_{block}$.
        # The Strassen decomposition of $X Y = Z$ uses 7 terms.
        # Term $t$ (0..6) involves $U^{(t)} X$ and $V^{(t)} Y$ and $W^{(t)} Z$.
        # $U^{(t)}$ is a 2x2 matrix of coefficients for the elements of X.
        # $V^{(t)}$ is a 2x2 matrix of coefficients for the elements of Y.
        # $W^{(t)}$ is a 2x2 matrix of coefficients for the elements of Z.
        
        # We already have u2, v2, w2 from strassen_2x2().
        # u2[ab, t] is the coefficient of $X_{ab}$ in term $t$.
        
        for t in range(7):
            col = l * 7 + t
            
            # Process U
            for a in range(2):
                for b in range(2):
                    coeff = u2[a*2+b, t]
                    if coeff != 0:
                        # This term uses $X_{ab}$ with coefficient coeff.
                        # $X_{ab}$ is a linear combo of A blocks.
                        for block_idx, alpha in block_coeffs_U[l].items():
                            if alpha != 0:
                                # The contribution to U is coeff * alpha.
                                # The index in U for block (a,b) and scalar index (i,j) within block.
                                # Block (a,b) of A (4x4) corresponds to rows 2a..2a+1, cols 2b..2b+1.
                                # U index is i*4+j.
                                for i in range(2):
                                    for j in range(2):
                                        i_global = 2*a + i
                                        j_global = 2*b + j
                                        u_idx = i_global * 4 + j_global
                                        U[u_idx, col] += coeff * alpha
                                        
            # Process V
            for c in range(2):
                for d in range(2):
                    coeff = v2[c*2+d, t]
                    if coeff != 0:
                        for block_idx, beta in block_coeffs_V[l].items():
                            if beta != 0:
                                # Block (c,d) of B.
                                # V index is j*4+k.
                                # Block (c,d) of B corresponds to rows 2c..2c+1, cols 2d..2d+1.
                                # Note: In V, the first index is j (row of B), second is k (col of B).
                                # So block (c,d) means j in 2c..2c+1, k in 2d..2d+1.
                                for i in range(2):
                                    for j in range(2):
                                        j_global = 2*c + i
                                        k_global = 2*d + j
                                        v_idx = j_global * 4 + k_global
                                        V[v_idx, col] += coeff * beta

            # Process W
            for e in range(2):
                for f in range(2):
                    coeff = w2[e*2+f, t]
                    if coeff != 0:
                        for block_idx, gamma in block_coeffs_W[l].items():
                            if gamma != 0:
                                # Block (e,f) of C.
                                # W index is k*4+i.
                                # Block (e,f) of C corresponds to rows 2e..2e+1, cols 2f..2f+1.
                                # Note: In W, the first index is k (row of C? No, C is 4x4, so row is i? No.)
                                # Problem statement: W[k*n+i, r].
                                # C is 4x4. i is row, k is col.
                                # So W index is k*4+i.
                                # Block (e,f) of C means row i in 2e..2e+1, col k in 2f..2f+1.
                                # So i in 2e..2e+1, k in 2f..2f+1.
                                for i in range(2):
                                    for j in range(2):
                                        i_global = 2*e + i
                                        k_global = 2*f + j
                                        w_idx = k_global * 4 + i_global
                                        W[w_idx, col] += coeff * gamma

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Generate Strassen 4x4 * 4x4
    U1, V1, W1 = _generate_strassen_4x4()
    
    # We need 4x4 * 4x8.
    # Split B into B1 (cols 0-3) and B2 (cols 4-7).
    # U is the same for both.
    # V1 is for B1 (cols 0-3). V2 is for B2 (cols 4-7).
    # W1 is for C1 (cols 0-3). W2 is for C2 (cols 4-7).
    
    R1 = U1.shape[1] # 49
    R_total = 2 * R1 # 98
    
    U = np.zeros((n*m, R_total))
    V = np.zeros((m*p, R_total))
    W = np.zeros((p*n, R_total))
    
    # Part 1: k in 0..3
    U[:, :R1] = U1
    # V1 has shape (16, 49). Indices j*4+k with k in 0..3.
    # We need to place this in V[:, :R1] which has indices j*8+k.
    # For j in 0..3, k in 0..3:
    for j in range(4):
        for k in range(4):
            V[j*8 + k, :R1] = V1[j*4 + k, :]
    # W1 has shape (16, 49). Indices k*4+i with k in 0..3.
    # We need to place this in W[:, :R1] which has indices k*4+i (since n=4, p=4 for this part? No, p=8 for full, but for this part k is 0..3).
    # W index is k*n + i. n=4.
    # So W[k*4 + i, :R1] = W1[k*4 + i, :]
    W[:, :R1] = W1
    
    # Part 2: k in 4..7
    # U is the same.
    U[:, R1:] = U1
    # V2: For j in 0..3, k in 4..7.
    # Let k' = k - 4.
    # V[j*8 + k, R1:] = V1[j*4 + k', :]
    for j in range(4):
        for k in range(4):
            k_prime = k
            k_global = 4 + k_prime
            V[j*8 + k_global, R1:] = V1[j*4 + k_prime, :]
            
    # W2: For k in 4..7.
    # W[k*4 + i, R1:] = W1[(k-4)*4 + i, :]
    for k in range(4):
        k_prime = k
        k_global = 4 + k_prime
        # W row index is k_global * 4 + i
        # W1 row index is k_prime * 4 + i
        for i in range(4):
            W[k_global*4 + i, R1:] = W1[k_prime*4 + i, :]
            
    # Verify
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(8):
                s = 0
                for r in range(R_total):
                    s += U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to schoolbook if Strassen implementation has a bug
        U_school = np.zeros((n * m, n * m * p))
        V_school = np.zeros((m * p, n * m * p))
        W_school = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
