import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # The total rank will be the sum of ranks of the blocks.
    
    # Block definitions
    # J indices: 0,1 (pair 1), 2,3 (pair 2), 4 (single)
    # K indices: 0,1 (pair 1), 2,3 (pair 2), 4 (single)
    
    # We accumulate U, V, W.
    # Since we don't know the total rank ahead of time easily without counting, 
    # we can use lists and then convert to arrays.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a rank-1 term
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # Helper to add a Strassen decomposition for a 2x2x2 block
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global mapping:
    # i_global = i
    # j_global = j_offset + j
    # k_global = k_offset + k
    
    def add_strassen_block(j_off, k_off):
        # Strassen factors for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M2 - M3
        # C12 = M5 + M3
        # C21 = M2 + M6
        # C22 = M1 - M4 + M5 - M7
        
        # We need to express each Mr as a sum of rank-1 terms?
        # No, each Mr is a scalar product of linear combinations.
        # In the tensor decomposition, each Mr corresponds to ONE rank-1 term in the U,V,W space?
        # Yes.
        # Term r:
        # u_r is the vector of coefficients for A.
        # v_r is the vector of coefficients for B.
        # w_r is the matrix of coefficients for C.
        
        # For M1:
        # u has 1s at (0,0) and (1,1).
        # v has 1s at (0,0) and (0,1)?? No, B11 is (0,0), B22 is (1,1) in local 2x2?
        # Wait, B is 2x2. B11 is (0,0), B12 is (0,1), B21 is (1,0), B22 is (1,1).
        # M1 = (A11+A22)(B11+B22).
        # u1: A11=1, A22=1.
        # v1: B11=1, B22=1.
        # w1: Contributes to C11 and C22.
        # C11 += M1. C22 += M1.
        # So w1 has 1 at (0,0) and 1 at (1,1).
        
        # Let's define the 7 terms for Strassen.
        # Each term is (u, v, w).
        
        # Term 1: M1
        u1 = [0]*4; u1[0]=1; u1[3]=1 # A11, A22 (indices 0,1,2,3 for 2x2)
        v1 = [0]*4; v1[0]=1; v1[3]=1 # B11, B22
        w1 = [0]*4; w1[0]=1; w1[3]=1 # C11, C22
        # Map to global
        add_strassen_term(u1, v1, w1, j_off, k_off)

        # Term 2: M2 = (A21+A22)B11
        u2 = [0]*4; u2[2]=1; u2[3]=1 # A21, A22
        v2 = [0]*4; v2[0]=1 # B11
        w2 = [0]*4; w2[2]=1; w2[3]=1 # C21, C22?
        # C21 += M2. C22 -= M2.
        # So w2 has 1 at (1,0) and -1 at (1,1).
        w2 = [0]*4; w2[2]=1; w2[3]=-1
        add_strassen_term(u2, v2, w2, j_off, k_off)

        # Term 3: M3 = A11(B21-B22)
        u3 = [0]*4; u3[0]=1 # A11
        v3 = [0]*4; v3[1]=1; v3[3]=-1 # B21, B22
        w3 = [0]*4; w3[0]=1; w3[1]=-1 # C11, C12?
        # C11 -= M3. C12 += M3.
        # So w3 has -1 at (0,0) and 1 at (0,1).
        w3 = [0]*4; w3[0]=-1; w3[1]=1
        add_strassen_term(u3, v3, w3, j_off, k_off)

        # Term 4: M4 = A22(B12-B11)
        u4 = [0]*4; u4[3]=1 # A22
        v4 = [0]*4; v4[1]=1; v4[0]=-1 # B12, B11
        w4 = [0]*4; w4[0]=1; w4[3]=-1 # C11, C22?
        # C11 += M4. C22 -= M4.
        # So w4 has 1 at (0,0) and -1 at (1,1).
        w4 = [0]*4; w4[0]=1; w4[3]=-1
        add_strassen_term(u4, v4, w4, j_off, k_off)

        # Term 5: M5 = (A11+A12)B22
        u5 = [0]*4; u5[0]=1; u5[1]=1 # A11, A12
        v5 = [0]*4; v5[3]=1 # B22
        w5 = [0]*4; w5[1]=1; w5[3]=1 # C12, C22?
        # C12 += M5. C22 += M5.
        # So w5 has 1 at (0,1) and 1 at (1,1).
        w5 = [0]*4; w5[1]=1; w5[3]=1
        add_strassen_term(u5, v5, w5, j_off, k_off)

        # Term 6: M6 = (A21-A11)(B11+B12)
        u6 = [0]*4; u6[2]=1; u6[0]=-1 # A21, A11
        v6 = [0]*4; v6[0]=1; v6[1]=1 # B11, B12
        w6 = [0]*4; w6[2]=1 # C21?
        # C21 += M6.
        # So w6 has 1 at (1,0).
        w6 = [0]*4; w6[2]=1
        add_strassen_term(u6, v6, w6, j_off, k_off)

        # Term 7: M7 = (A12-A22)(B21+B22)
        u7 = [0]*4; u7[1]=1; u7[3]=-1 # A12, A22
        v7 = [0]*4; v7[1]=1; v7[3]=1 # B21, B22
        w7 = [0]*4; w7[3]=-1 # C22?
        # C22 -= M7.
        # So w7 has -1 at (1,1).
        w7 = [0]*4; w7[3]=-1
        add_strassen_term(u7, v7, w7, j_off, k_off)

    def add_strassen_term(u_loc, v_loc, w_loc, j_off, k_off):
        # u_loc is length 4 (2x2). Flatten to global A (2x5).
        # v_loc is length 4 (2x2). Flatten to global B (5x5).
        # w_loc is length 4 (2x2). Flatten to global C (2x5).
        
        u_g = [0]*(n*m)
        v_g = [0]*(m*p)
        w_g = [0]*(p*n)
        
        # Map A (2x2) to A (2x5)
        # Local (i, j) -> Global (i, j_off + j)
        for i in range(2):
            for j in range(2):
                idx_loc = i*2 + j
                idx_g = i*m + (j_off + j)
                u_g[idx_g] = u_loc[idx_loc]
                
        # Map B (2x2) to B (5x5)
        # Local (j, k) -> Global (j_off + j, k_off + k)
        for j in range(2):
            for k in range(2):
                idx_loc = j*2 + k
                idx_g = (j_off + j)*p + (k_off + k)
                v_g[idx_g] = v_loc[idx_loc]
                
        # Map C (2x2) to C (2x5)
        # Local (i, k) -> Global (i, k_off + k)
        # Note: W index is k*n + i.
        for i in range(2):
            for k in range(2):
                idx_loc = i*2 + k
                idx_g = (k_off + k)*n + i
                w_g[idx_g] = w_loc[idx_loc]
                
        add_term(u_g, v_g, w_g)

    # Add the 4 Strassen blocks
    add_strassen_block(0, 0) # J=0,1 K=0,1
    add_strassen_block(2, 0) # J=2,3 K=0,1
    add_strassen_block(0, 2) # J=0,1 K=2,3
    add_strassen_block(2, 2) # J=2,3 K=2,3
    
    # Add the border blocks
    # J=4, K=4: M_{2,1,1}
    # Schoolbook: 2 terms.
    # Term 1: A_{0,4} B_{4,4} C_{0,4}
    u = [0]*(n*m); u[0*m+4] = 1
    v = [0]*(m*p); v[4*p+4] = 1
    w = [0]*(p*n); w[4*n+0] = 1
    add_term(u, v, w)
    
    # Term 2: A_{1,4} B_{4,4} C_{1,4}
    u = [0]*(n*m); u[1*m+4] = 1
    v = [0]*(m*p); v[4*p+4] = 1
    w = [0]*(p*n); w[4*n+1] = 1
    add_term(u, v, w)
    
    # J=0,1 K=4: M_{2,2,1}
    # Schoolbook: 4 terms.
    # C_{i,4} = A_{i,0}B_{0,4} + A_{i,1}B_{1,4}
    # i=0: C_{0,4} = A_{0,0}B_{0,4} + A_{0,1}B_{1,4}
    # i=1: C_{1,4} = A_{1,0}B_{0,4} + A_{1,1}B_{1,4}
    
    # Term: A_{0,0} B_{0,4} C_{0,4}
    u = [0]*(n*m); u[0*m+0] = 1
    v = [0]*(m*p); v[0*p+4] = 1
    w = [0]*(p*n); w[4*n+0] = 1
    add_term(u, v, w)
    
    # Term: A_{0,1} B_{1,4} C_{0,4}
    u = [0]*(n*m); u[0*m+1] = 1
    v = [0]*(m*p); v[1*p+4] = 1
    w = [0]*(p*n); w[4*n+0] = 1
    add_term(u, v, w)
    
    # Term: A_{1,0} B_{0,4} C_{1,4}
    u = [0]*(n*m); u[1*m+0] = 1
    v = [0]*(m*p); v[0*p+4] = 1
    w = [0]*(p*n); w[4*n+1] = 1
    add_term(u, v, w)
    
    # Term: A_{1,1} B_{1,4} C_{1,4}
    u = [0]*(n*m); u[1*m+1] = 1
    v = [0]*(m*p); v[1*p+4] = 1
    w = [0]*(p*n); w[4*n+1] = 1
    add_term(u, v, w)
    
    # J=2,3 K=4: M_{2,2,1}
    # Similar, with j offsets 2,3.
    
    # Term: A_{0,2} B_{2,4} C_{0,4}
    u = [0]*(n*m); u[0*m+2] = 1
    v = [0]*(m*p); v[2*p+4] = 1
    w = [0]*(p*n); w[4*n+0] = 1
    add_term(u, v, w)
    
    # Term: A_{0,3} B_{3,4} C_{0,4}
    u = [0]*(n*m); u[0*m+3] = 1
    v = [0]*(m*p); v[3*p+4] = 1
    w = [0]*(p*n); w[4*n+0] = 1
    add_term(u, v, w)
    
    # Term: A_{1,2} B_{2,4} C_{1,4}
    u = [0]*(n*m); u[1*m+2] = 1
    v = [0]*(m*p); v[2*p+4] = 1
    w = [0]*(p*n); w[4*n+1] = 1
    add_term(u, v, w)
    
    # Term: A_{1,3} B_{3,4} C_{1,4}
    u = [0]*(n*m); u[1*m+3] = 1
    v = [0]*(m*p); v[3*p+4] = 1
    w = [0]*(p*n); w[4*n+1] = 1
    add_term(u, v, w)
    
    # J=4 K=0,1: M_{2,1,2}
    # C_{i,k} = A_{i,4} B_{4,k}
    # i=0,1; k=0,1
    
    # Term: A_{0,4} B_{4,0} C_{0,0}
    u = [0]*(n*m); u[0*m+4] = 1
    v = [0]*(m*p); v[4*p+0] = 1
    w = [0]*(p*n); w[0*n+0] = 1
    add_term(u, v, w)
    
    # Term: A_{0,4} B_{4,1} C_{0,1}
    u = [0]*(n*m); u[0*m+4] = 1
    v = [0]*(m*p); v[4*p+1] = 1
    w = [0]*(p*n); w[1*n+0] = 1
    add_term(u, v, w)
    
    # Term: A_{1,4} B_{4,0} C_{1,0}
    u = [0]*(n*m); u[1*m+4] = 1
    v = [0]*(m*p); v[4*p+0] = 1
    w = [0]*(p*n); w[0*n+1] = 1
    add_term(u, v, w)
    
    # Term: A_{1,4} B_{4,1} C_{1,1}
    u = [0]*(n*m); u[1*m+4] = 1
    v = [0]*(m*p); v[4*p+1] = 1
    w = [0]*(p*n); w[1*n+1] = 1
    add_term(u, v, w)
    
    # J=4 K=2,3: M_{2,1,2}
    # i=0,1; k=2,3
    
    # Term: A_{0,4} B_{4,2} C_{0,2}
    u = [0]*(n*m); u[0*m+4] = 1
    v = [0]*(m*p); v[4*p+2] = 1
    w = [0]*(p*n); w[2*n+0] = 1
    add_term(u, v, w)
    
    # Term: A_{0,4} B_{4,3} C_{0,3}
    u = [0]*(n*m); u[0*m+4] = 1
    v = [0]*(m*p); v[4*p+3] = 1
    w = [0]*(p*n); w[3*n+0] = 1
    add_term(u, v, w)
    
    # Term: A_{1,4} B_{4,2} C_{1,2}
    u = [0]*(n*m); u[1*m+4] = 1
    v = [0]*(m*p); v[4*p+2] = 1
    w = [0]*(p*n); w[2*n+1] = 1
    add_term(u, v, w)
    
    # Term: A_{1,4} B_{4,3} C_{1,3}
    u = [0]*(n*m); u[1*m+4] = 1
    v = [0]*(m*p); v[4*p+3] = 1
    w = [0]*(p*n); w[3*n+1] = 1
    add_term(u, v, w)
    
    # Convert to numpy arrays
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U, "V": V, "W": W}
