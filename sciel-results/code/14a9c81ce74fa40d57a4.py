import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition.
    # Split m=5 into 4+1, p=5 into 4+1.
    # A is 2x5, B is 5x5.
    # A1 (2x4), A2 (2x1)
    # B11 (4x4), B12 (4x1)
    # B21 (1x4), B22 (1x1)
    # C1 (2x4) = A1 B11 + A2 B21
    # C2 (2x1) = A1 B12 + A2 B22
    
    # We need to decompose:
    # 1. A1 B11 (2x4 x 4x4) using Strassen blocks (2x2x2)
    # 2. A2 B21 (2x1 x 1x4) using Schoolbook
    # 3. A1 B12 (2x4 x 4x1) using Schoolbook
    # 4. A2 B22 (2x1 x 1x1) using Schoolbook
    
    # We will build U, V, W lists.
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_mat):
        # u_vec: length n*m (10)
        # v_vec: length m*p (25)
        # w_mat: length p*n (10)
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_mat)

    # Helper to create zero vectors
    def zero_u():
        return [0.0] * (n * m)
    def zero_v():
        return [0.0] * (m * p)
    def zero_w():
        return [0.0] * (p * n)

    # Strassen 2x2x2 decomposition
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # Local indices: i_l, j_l, k_l
    # Global indices: i_g, j_g, k_g
    
    def strassen_block(i_offset, j_offset, k_offset):
        # Adds 7 terms for a 2x2x2 block at given offsets
        # A is at i_offset, j_offset
        # B is at j_offset, k_offset
        # C is at i_offset, k_offset
        
        # M1 = (A11 + A22)(B11 + B22)
        u1 = zero_u()
        v1 = zero_v()
        w1 = zero_w()
        u1[i_offset*m + j_offset] = 1.0
        u1[(i_offset+1)*m + (j_offset+1)] = 1.0
        v1[j_offset*p + k_offset] = 1.0
        v1[(j_offset+1)*p + (k_offset+1)] = 1.0
        w1[k_offset*n + i_offset] = 1.0
        w1[(k_offset+1)*n + (i_offset+1)] = 1.0
        # C11 += M1, C22 += M1
        # w1 is the matrix W_k.
        # M1 contributes to C11 and C22 with coeff 1.
        # So w1 has 1 at (i_offset, k_offset) and (i_offset+1, k_offset+1).
        add_term(u1, v1, w1)
        
        # M2 = (A21 + A22) B11
        u2 = zero_u()
        v2 = zero_v()
        w2 = zero_w()
        u2[(i_offset+1)*m + j_offset] = 1.0
        u2[(i_offset+1)*m + (j_offset+1)] = 1.0
        v2[j_offset*p + k_offset] = 1.0
        # C21 += M2, C22 -= M2
        w2[(k_offset+1)*n + i_offset] = 1.0 # C21 (i=1, k=0)
        w2[(k_offset+1)*n + (i_offset+1)] = -1.0 # C22 (i=1, k=1)
        add_term(u2, v2, w2)
        
        # M3 = A11 (B12 - B22)
        u3 = zero_u()
        v3 = zero_v()
        w3 = zero_w()
        u3[i_offset*m + j_offset] = 1.0
        v3[j_offset*p + (k_offset+1)] = 1.0
        v3[(j_offset+1)*p + (k_offset+1)] = -1.0
        # C12 += M3, C22 += M3
        w3[k_offset*n + (i_offset+1)] = 1.0 # C12 (i=0, k=1)
        w3[(k_offset+1)*n + (i_offset+1)] = 1.0 # C22 (i=1, k=1)
        add_term(u3, v3, w3)
        
        # M4 = A22 (B21 - B11)
        u4 = zero_u()
        v4 = zero_v()
        w4 = zero_w()
        u4[(i_offset+1)*m + (j_offset+1)] = 1.0
        v4[(j_offset+1)*p + k_offset] = 1.0
        v4[j_offset*p + k_offset] = -1.0
        # C11 += M4, C21 += M4
        w4[k_offset*n + i_offset] = 1.0 # C11 (i=0, k=0)
        w4[(k_offset+1)*n + i_offset] = 1.0 # C21 (i=1, k=0)
        add_term(u4, v4, w4)
        
        # M5 = (A11 + A12) B22
        u5 = zero_u()
        v5 = zero_v()
        w5 = zero_w()
        u5[i_offset*m + j_offset] = 1.0
        u5[i_offset*m + (j_offset+1)] = 1.0
        v5[(j_offset+1)*p + (k_offset+1)] = 1.0
        # C11 -= M5, C12 += M5
        w5[k_offset*n + i_offset] = -1.0 # C11 (i=0, k=0)
        w5[k_offset*n + (i_offset+1)] = 1.0 # C12 (i=0, k=1)
        add_term(u5, v5, w5)
        
        # M6 = (A21 - A11)(B11 + B12)
        u6 = zero_u()
        v6 = zero_v()
        w6 = zero_w()
        u6[(i_offset+1)*m + j_offset] = 1.0
        u6[i_offset*m + j_offset] = -1.0
        v6[j_offset*p + k_offset] = 1.0
        v6[j_offset*p + (k_offset+1)] = 1.0
        # C22 += M6
        w6[(k_offset+1)*n + (i_offset+1)] = 1.0 # C22 (i=1, k=1)
        add_term(u6, v6, w6)
        
        # M7 = (A12 - A22)(B21 + B22)
        u7 = zero_u()
        v7 = zero_v()
        w7 = zero_w()
        u7[i_offset*m + (j_offset+1)] = 1.0
        u7[(i_offset+1)*m + (j_offset+1)] = -1.0
        v7[(j_offset+1)*p + k_offset] = 1.0
        v7[(j_offset+1)*p + (k_offset+1)] = 1.0
        # C11 += M7
        w7[k_offset*n + i_offset] = 1.0 # C11 (i=0, k=0)
        add_term(u7, v7, w7)

    # Apply Strassen to A1 B11 (2x4 x 4x4)
    # A1 is rows 0-1, cols 0-3 of A.
    # B11 is rows 0-3, cols 0-3 of B.
    # C1 is rows 0-1, cols 0-3 of C.
    # Split 4 into 2+2.
    # Block 1: i_off=0, j_off=0, k_off=0 (2x2x2)
    strassen_block(0, 0, 0)
    # Block 2: i_off=0, j_off=2, k_off=0 (A1 col 2-3, B11 row 0-1? No)
    # A1 is 2x4. B11 is 4x4.
    # A1 = [A1_11 A1_12] (2x2, 2x2)
    # B11 = [B11_11 B11_12; B11_21 B11_22] (2x2 blocks)
    # C1 = [C1_11 C1_12] (2x2 blocks)
    # C1_11 = A1_11 B11_11 + A1_12 B11_21
    # C1_12 = A1_11 B11_12 + A1_12 B11_22
    
    # We need to call strassen_block for each 2x2x2 component?
    # No, Strassen is for 2x2x2 matrix mult.
    # Here we are doing 2x4 x 4x4.
    # We can view it as two 2x2 x 2x4 multiplications?
    # Or 2x4 x 2x2 and 2x4 x 2x2?
    
    # Let's use the block structure of Strassen directly on 2x4x4.
    # A = [A11 A12] (2x2, 2x2)
    # B = [B11 B12; B21 B22] (2x2, 2x2; 2x2, 2x2)
    # C = [C11 C12] (2x2, 2x2)
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    
    # We can compute A11 B11 (2x2x2) with Strassen (7 terms).
    # A12 B21 (2x2x2) with Strassen (7 terms).
    # A11 B12 (2x2x2) with Strassen (7 terms).
    # A12 B22 (2x2x2) with Strassen (7 terms).
    # Total 28 terms.
    
    # Block 1: A11 B11. i_off=0, j_off=0, k_off=0.
    strassen_block(0, 0, 0)
    # Block 2: A12 B21. i_off=0, j_off=2, k_off=0.
    # A12 is A cols 2,3. B21 is B rows 2,3, cols 0,1.
    strassen_block(0, 2, 0)
    # Block 3: A11 B12. i_off=0, j_off=0, k_off=2.
    # A11 is A cols 0,1. B12 is B rows 0,1, cols 2,3.
    strassen_block(0, 0, 2)
    # Block 4: A12 B22. i_off=0, j_off=2, k_off=2.
    strassen_block(0, 2, 2)
    
    # Now handle A2 B21 (2x1 x 1x4)
    # A2 is A col 4. B21 is B row 4, cols 0-3.
    # C2_part1 is C cols 0-3, row 0-1.
    # C[i, k] = A[i, 4] * B[4, k]
    # 2x4 = 8 terms.
    for i in range(2):
        for k in range(4):
            u = zero_u()
            v = zero_v()
            w = zero_w()
            u[i*m + 4] = 1.0
            v[4*p + k] = 1.0
            w[k*n + i] = 1.0
            add_term(u, v, w)
            
    # Handle A1 B12 (2x4 x 4x1)
    # A1 is A cols 0-3. B12 is B rows 0-3, col 4.
    # C2_part2 is C col 4, row 0-1.
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # 2x4 = 8 terms.
    for i in range(2):
        for j in range(4):
            u = zero_u()
            v = zero_v()
            w = zero_w()
            u[i*m + j] = 1.0
            v[j*p + 4] = 1.0
            w[4*n + i] = 1.0
            add_term(u, v, w)
            
    # Handle A2 B22 (2x1 x 1x1)
    # A2 is A col 4. B22 is B row 4, col 4.
    # C2_part3 is C col 4, row 0-1.
    # C[i, 4] += A[i, 4] * B[4, 4]
    # 2 terms.
    for i in range(2):
        u = zero_u()
        v = zero_v()
        w = zero_w()
        u[i*m + 4] = 1.0
        v[4*p + 4] = 1.0
        w[4*n + i] = 1.0
        add_term(u, v, w)
        
    rank = len(U_list)
    
    # Convert to lists of lists
    U = U_list
    V = V_list
    W = W_list
    
    return {"U": U, "V": V, "W": W}
