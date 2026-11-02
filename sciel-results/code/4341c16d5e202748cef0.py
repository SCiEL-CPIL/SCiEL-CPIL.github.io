import numpy as np

def construct():
    # Dimensions: 4x4 matrix * 4x8 matrix = 4x8 matrix
    n, m, p = 4, 4, 8
    
    # Helper to get Strassen decomposition for 2x2x2
    # Returns U, V, W matrices of shape (4, 7), (4, 7), (4, 7)
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U[i, r], V[j, r], W[k, r]
    # T_{i,j,k} = sum_r U[i,r] V[j,r] W[k,r]
    # Note: The tensor for matrix multiplication C = AB is T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # Wait, standard tensor T_{i,j,k} for C_{ik} = sum_j A_{ij} B_{jk}.
    # In the factorization U, V, W:
    # U corresponds to A indices (i, j) -> size n*m
    # V corresponds to B indices (j, k) -> size m*p
    # W corresponds to C indices (i, k) -> size n*p
    # The prompt says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank) -- Note: W[k*n+i, r]
    
    # Strassen for 2x2x2:
    # A is 2x2, B is 2x2, C is 2x2.
    # U_str (4, 7), V_str (4, 7), W_str (4, 7)
    # Index mapping for 2x2:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_str = np.zeros((4, 7))
    V_str = np.zeros((4, 7))
    W_str = np.zeros((4, 7))
    
    # P1 = (A00 + A11)(B00 + B11)
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[1, 0] = 1; W_str[2, 0] = 1; W_str[3, 0] = 1
    # Correction: W for C. C00 = P1 - P4 - P6 + P7?
    # Let's use the standard Strassen formulas for C:
    # C00 = P1 - P4 - P6 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 - P5 + P7
    
    # Let's define the 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U (A part)
    # P1: A00, A11
    U_str[0, 0] = 1; U_str[3, 0] = 1
    # P2: A10, A11
    U_str[2, 1] = 1; U_str[3, 1] = 1
    # P3: A00
    U_str[0, 2] = 1
    # P4: A11
    U_str[3, 3] = 1
    # P5: A00, A01
    U_str[0, 4] = 1; U_str[1, 4] = 1
    # P6: A10, A00
    U_str[2, 5] = 1; U_str[0, 5] = -1
    # P7: A01, A11
    U_str[1, 6] = 1; U_str[3, 6] = -1
    
    # V (B part)
    # P1: B00, B11
    V_str[0, 0] = 1; V_str[3, 0] = 1
    # P2: B00
    V_str[0, 1] = 1
    # P3: B01, B11
    V_str[1, 2] = 1; V_str[3, 2] = -1
    # P4: B10, B00
    V_str[2, 3] = 1; V_str[0, 3] = -1
    # P5: B11
    V_str[3, 4] = 1
    # P6: B00, B01
    V_str[0, 5] = 1; V_str[1, 5] = 1
    # P7: B10, B11
    V_str[2, 6] = 1; V_str[3, 6] = 1
    
    # W (C part)
    # C00 = P1 - P4 - P6 + P7
    W_str[0, 0] = 1; W_str[0, 3] = -1; W_str[0, 5] = -1; W_str[0, 6] = 1
    # C01 = P3 + P5
    W_str[1, 2] = 1; W_str[1, 4] = 1
    # C10 = P2 + P4
    W_str[2, 1] = 1; W_str[2, 3] = 1
    # C11 = P1 - P3 - P5 + P7
    W_str[3, 0] = 1; W_str[3, 2] = -1; W_str[3, 4] = -1; W_str[3, 6] = 1
    
    # Now construct 2x2x4 decomposition (rank 14)
    # This is two independent Strassen multiplications.
    # One for B cols 0,1 (indices 0,1 in B) -> C cols 0,1 (indices 0,1 in C)
    # One for B cols 2,3 (indices 2,3 in B) -> C cols 2,3 (indices 2,3 in C)
    # But wait, the 2x2x4 tensor has B size 2x4.
    # Indices for B: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)
    # Indices for C: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)
    
    # We map the Strassen V and W to these larger indices.
    # For the first block (cols 0,1):
    # B indices: 0:(0,0), 1:(0,1) -> map to Strassen B 0,1
    #           4:(1,0), 5:(1,1) -> map to Strassen B 2,3
    # C indices: 0:(0,0), 1:(0,1) -> map to Strassen C 0,1
    #           4:(1,0), 5:(1,1) -> map to Strassen C 2,3
    
    # For the second block (cols 2,3):
    # B indices: 2:(0,2), 3:(0,3) -> map to Strassen B 0,1
    #           6:(1,2), 7:(1,3) -> map to Strassen B 2,3
    # C indices: 2:(0,2), 3:(0,3) -> map to Strassen C 0,1
    #           6:(1,2), 7:(1,3) -> map to Strassen C 2,3
    
    # U for 2x2x4 is same as U_str (4, 7) but repeated for each block?
    # No, U is shared. But in the Kronecker product approach, we treat it as a tensor product.
    # Let's just build the 4x4x8 directly using Kronecker product of the decompositions.
    
    # M_4,4,8 = M_2,2,2 (for the "2x2" part of 4) x M_2,2,4 (for the "2x4" part of 8? No)
    # 4x4x8 = (2x2) x (2x4) ?
    # i = 2*i1 + i0
    # j = 2*j1 + j0
    # k = 2*k1 + k0
    # i1, j1 in 0..1 (2x2)
    # k1 in 0..3 (4), k0 in 0..1 (2)
    # So M_4,4,8 = M_2,2,4 (outer) x M_2,2,2 (inner)?
    # Outer: i1, j1, k1. A is 2x2, B is 2x4, C is 2x4.
    # Inner: i0, j0, k0. A is 2x2, B is 2x2, C is 2x2.
    # So Rank = Rank(M_2,2,4) * Rank(M_2,2,2).
    
    # We have Rank(M_2,2,2) = 7.
    # We have Rank(M_2,2,4) = 14 (two Strassens).
    # Total Rank = 98.
    
    # Let's build the factors U, V, W for 4x4x8.
    # U size: 4*4 = 16.
    # V size: 4*8 = 32.
    # W size: 8*4 = 32.
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # We iterate over the 14 products of the outer 2x2x4 and 7 products of the inner 2x2x2.
    # Outer product r_out in 0..13
    # Inner product r_in in 0..6
    # Total rank index r = r_out * 7 + r_in
    
    # Outer decomposition (2x2x4) factors:
    # U_out (4, 14), V_out (8, 14), W_out (8, 14)
    # We construct U_out, V_out, W_out from two Strassen instances.
    
    U_out = np.zeros((4, 14))
    V_out = np.zeros((8, 14))
    W_out = np.zeros((8, 14))
    
    # Instance 1: cols 0,1
    # U_out[:, 0:7] = U_str
    U_out[:, 0:7] = U_str
    # V_out mapping:
    # Strassen V indices 0,1,2,3 correspond to B(0,0), B(0,1), B(1,0), B(1,1)
    # In 2x2x4 B:
    # B(0,0) is index 0
    # B(0,1) is index 1
    # B(1,0) is index 4
    # B(1,1) is index 5
    V_out[0, 0:7] = V_str[0]
    V_out[1, 0:7] = V_str[1]
    V_out[4, 0:7] = V_str[2]
    V_out[5, 0:7] = V_str[3]
    # W_out mapping:
    # Strassen W indices 0,1,2,3 correspond to C(0,0), C(0,1), C(1,0), C(1,1)
    # In 2x2x4 C:
    # C(0,0) is index 0
    # C(0,1) is index 1
    # C(1,0) is index 4
    # C(1,1) is index 5
    W_out[0, 0:7] = W_str[0]
    W_out[1, 0:7] = W_str[1]
    W_out[4, 0:7] = W_str[2]
    W_out[5, 0:7] = W_str[3]
    
    # Instance 2: cols 2,3
    U_out[:, 7:14] = U_str
    # V_out mapping:
    # B(0,2) is index 2
    # B(0,3) is index 3
    # B(1,2) is index 6
    # B(1,3) is index 7
    V_out[2, 7:14] = V_str[0]
    V_out[3, 7:14] = V_str[1]
    V_out[6, 7:14] = V_str[2]
    V_out[7, 7:14] = V_str[3]
    # W_out mapping:
    # C(0,2) is index 2
    # C(0,3) is index 3
    # C(1,2) is index 6
    # C(1,3) is index 7
    W_out[2, 7:14] = W_str[0]
    W_out[3, 7:14] = W_str[1]
    W_out[6, 7:14] = W_str[2]
    W_out[7, 7:14] = W_str[3]
    
    # Now Kronecker product with Inner 2x2x2 (Strassen)
    # U_final[i_total, r_total]
    # i_total = i_out * 2 + i_in (since n=4, i_out in 0..1, i_in in 0..1? No, i is index of A)
    # A is 4x4. i in 0..3.
    # i = 2*i_out + i_in.
    # U_out has size 4 (i_out, j_out).
    # U_in has size 4 (i_in, j_in).
    # U_final index: (i_out, j_out, i_in, j_in) -> flattened?
    # The prompt says U is (n*m, rank). n=4, m=4. Size 16.
    # Index u = i*m + j = i*4 + j.
    # i = 2*i_out + i_in
    # j = 2*j_out + j_in
    # u = (2*i_out + i_in)*4 + (2*j_out + j_in)
    #   = 8*i_out + 4*i_in + 2*j_out + j_in
    
    # V_final index: (j_out, k_out, j_in, k_in)
    # j = 2*j_out + j_in
    # k = 2*k_out + k_in
    # v = j*p + k = (2*j_out + j_in)*8 + (2*k_out + k_in)
    #   = 16*j_out + 8*j_in + 2*k_out + k_in
    
    # W_final index: (k_out, i_out, k_in, i_in)
    # Prompt: W[k*n + i, r]
    # w = k*n + i = k*4 + i
    # k = 2*k_out + k_in
    # i = 2*i_out + i_in
    # w = (2*k_out + k_in)*4 + (2*i_out + i_in)
    #   = 8*k_out + 4*k_in + 2*i_out + i_in
    
    # r_total = r_out * 7 + r_in
    
    for r_out in range(14):
        for r_in in range(7):
            r_total = r_out * 7 + r_in
            # Sum over i_out, j_out, i_in, j_in
            for i_out in range(2):
                for j_out in range(2):
                    for i_in in range(2):
                        for j_in in range(2):
                            u_val = U_out[i_out*2 + j_out, r_out] * U_str[i_in*2 + j_in, r_in]
                            if u_val != 0:
                                u_idx = 8*i_out + 4*i_in + 2*j_out + j_in
                                U[u_idx, r_total] += u_val
                            
                            v_val = V_out[j_out*4 + k_out_placeholder, r_out] # Need k_out
                            # This loop is getting complex. Let's use numpy kron.
                            
    # Using numpy kron is easier and less error-prone.
    # U_final = kron(U_out, U_str) ?
    # U_out (4, 14), U_str (4, 7).
    # kron(U_out, U_str) will be (16, 98).
    # Does the index ordering match?
    # U_out index: i_out*2 + j_out.
    # U_str index: i_in*2 + j_in.
    # kron index: (i_out*2 + j_out)*4 + (i_in*2 + j_in)
    #             = 8*i_out + 4*j_out + 2*i_in + j_in.
    # My derived u_idx: 8*i_out + 4*i_in + 2*j_out + j_in.
    # Mismatch!
    # I need to permute the indices or reorder the Kronecker product.
    # I want u = i*4 + j where i=2*i_out+i_in, j=2*j_out+j_in.
    # u = (2*i_out+i_in)*4 + 2*j_out+j_in = 8*i_out + 4*i_in + 2*j_out + j_in.
    # The standard kron(U_out, U_str) gives 8*i_out + 4*j_out + 2*i_in + j_in.
    # So I need to swap the roles of i_in and j_out in the kron?
    # Actually, I can just construct the matrices manually or use a permutation.
    
    # Let's stick to the manual loop for correctness, it's small (16*98).
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    for r_out in range(14):
        for r_in in range(7):
            r_total = r_out * 7 + r_in
            for i_out in range(2):
                for j_out in range(2):
                    for i_in in range(2):
                        for j_in in range(2):
                            u_val = U_out[i_out*2 + j_out, r_out] * U_str[i_in*2 + j_in, r_in]
                            if u_val != 0:
                                u_idx = 8*i_out + 4*i_in + 2*j_out + j_in
                                U[u_idx, r_total] += u_val
                            
                            # V
                            for k_out in range(4):
                                for k_in in range(2):
                                    v_val = V_out[j_out*4 + k_out, r_out] * V_str[j_in*2 + k_in, r_in]
                                    if v_val != 0:
                                        v_idx = 16*j_out + 8*j_in + 2*k_out + k_in
                                        V[v_idx, r_total] += v_val
                                        
                            # W
                            for k_out in range(4):
                                for k_in in range(2):
                                    w_val = W_out[k_out*2 + i_out, r_out] * W_str[k_in*2 + i_in, r_in]
                                    if w_val != 0:
                                        w_idx = 8*k_out + 4*k_in + 2*i_out + i_in
                                        W[w_idx, r_total] += w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
