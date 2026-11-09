import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # We will build the decomposition recursively.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # Helper to add a decomposition of a sub-problem to the main one
    # sub_U, sub_V, sub_W are for the sub-problem of size n_sub, m_sub, p_sub
    # row_offset, col_offset for A (sub_U)
    # row_offset, col_offset for B (sub_V)
    # row_offset, col_offset for C (sub_W) -> Note: W is indexed by k*n + i
    # signs for U, V, W
    
    def add_decomp(sub_U, sub_V, sub_W, n_sub, m_sub, p_sub, 
                   a_row_off, a_col_off, 
                   b_row_off, b_col_off, 
                   c_row_off, c_col_off, # c_row is k, c_col is i? No, W is (p*n, rank)
                   rank_offset, 
                   s_u=1, s_v=1, s_w=1):
        # sub_U shape: (n_sub*m_sub, sub_rank)
        # sub_V shape: (m_sub*p_sub, sub_rank)
        # sub_W shape: (p_sub*n_sub, sub_rank)
        
        # We need to map indices.
        # A index: (i, j) -> i*m + j
        # Sub A index: (i_sub, j_sub) -> i_sub*m_sub + j_sub
        # Global A index: (a_row_off + i_sub) * m + (a_col_off + j_sub)
        
        # B index: (j, k) -> j*p + k
        # Sub B index: (j_sub, k_sub) -> j_sub*p_sub + k_sub
        # Global B index: (b_row_off + j_sub) * p + (b_col_off + k_sub)
        
        # C index: (i, k) -> k*n + i
        # Sub C index: (i_sub, k_sub) -> k_sub*n_sub + i_sub
        # Global C index: (c_row_off + k_sub) * n + (c_col_off + i_sub)
        
        sub_rank = sub_U.shape[1]
        
        for r in range(sub_rank):
            # Update U
            for idx_u in range(n_sub * m_sub):
                i_sub = idx_u // m_sub
                j_sub = idx_u % m_sub
                val = sub_U[idx_u][r]
                if val != 0:
                    global_idx_u = (a_row_off + i_sub) * m + (a_col_off + j_sub)
                    U[global_idx_u][rank_offset + r] += s_u * val
            
            # Update V
            for idx_v in range(m_sub * p_sub):
                j_sub = idx_v // p_sub
                k_sub = idx_v % p_sub
                val = sub_V[idx_v][r]
                if val != 0:
                    global_idx_v = (b_row_off + j_sub) * p + (b_col_off + k_sub)
                    V[global_idx_v][rank_offset + r] += s_v * val
            
            # Update W
            for idx_w in range(p_sub * n_sub):
                k_sub = idx_w // n_sub
                i_sub = idx_w % n_sub
                val = sub_W[idx_w][r]
                if val != 0:
                    global_idx_w = (c_row_off + k_sub) * n + (c_col_off + i_sub)
                    W[global_idx_w][rank_offset + r] += s_w * val

    # Base case: 2x2 * 2x4
    # Rank 14: Two 2x2*2x2 Strassen decompositions
    def base_2_2_2_4():
        # Returns U (4, 14), V (8, 14), W (8, 14)
        # Actually, let's just use schoolbook for 2x2*2x4 to be safe?
        # Schoolbook rank 16.
        # Strassen rank 14 is better.
        # Let's implement Strassen for 2x2*2x2 (rank 7) and call it twice.
        
        def strassen_2_2_2():
            # Rank 7
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # P1 = (A00+A11)(B00+B11)
            U[0,0] = 1; U[3,0] = 1
            V[0,0] = 1; V[3,0] = 1
            W[0,0] = 1; W[1,0] = 1; W[2,0] = -1; W[3,0] = 1 # C00=P1+P4-P5+P7, C01=P3+P5, C10=P2+P4, C11=P1-P2+P3+P6
            # Wait, W indices: W[k*n+i, r]. n=2.
            # C00 (i=0,k=0) -> W[0, r]
            # C01 (i=0,k=1) -> W[2, r]
            # C10 (i=1,k=0) -> W[1, r]
            # C11 (i=1,k=1) -> W[3, r]
            
            # Let's re-verify W coefficients for Strassen 2x2
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            W[0,0] = 1 # P1
            W[0,3] = 1 # P4
            W[0,4] = -1 # P5
            W[0,6] = 1 # P7
            
            W[2,2] = 1 # P3
            W[2,4] = 1 # P5
            
            W[1,1] = 1 # P2
            W[1,3] = 1 # P4
            
            W[3,0] = 1 # P1
            W[3,1] = -1 # P2
            W[3,2] = 1 # P3
            W[3,5] = 1 # P6
            
            # P2 = (A10+A11)B00
            U[2,1] = 1; U[3,1] = 1
            V[0,1] = 1
            
            # P3 = A00(B01-B11)
            U[0,2] = 1
            V[1,2] = 1; V[3,2] = -1
            
            # P4 = A11(B10-B00)
            U[3,3] = 1
            V[2,3] = 1; V[0,3] = -1
            
            # P5 = (A00+A01)B11
            U[0,4] = 1; U[1,4] = 1
            V[3,4] = 1
            
            # P6 = (A10-A00)(B00+B01)
            U[2,5] = 1; U[0,5] = -1
            V[0,5] = 1; V[1,5] = 1
            
            # P7 = (A01-A11)(B10+B11)
            U[1,6] = 1; U[3,6] = -1
            V[2,6] = 1; V[3,6] = 1
            
            return U, V, W

        U1, V1, W1 = strassen_2_2_2()
        U2, V2, W2 = strassen_2_2_2()
        
        # Combine for 2x2 * 2x4
        # B is 2x4. Split into B_left (2x2) and B_right (2x2).
        # V_left corresponds to B_left (indices 0..3 in V? No, V is 4x7 for 2x2*2x2)
        # For 2x2*2x4, V is 8x14.
        # V indices: j*4+k.
        # B_left: k=0,1. B_right: k=2,3.
        
        U = np.zeros((4, 14))
        V = np.zeros((8, 14))
        W = np.zeros((8, 14))
        
        # Copy first 7 columns from U1, V1, W1
        U[:, 0:7] = U1
        # Map V1 to V_left
        # V1 is 4x7. Indices j*2+k.
        # V is 8x14. Indices j*4+k.
        for j in range(2):
            for k in range(2):
                V[j*4+k, 0:7] = V1[j*2+k, :]
        # W is 8x14. Indices k*2+i.
        # W1 is 4x7. Indices k*2+i.
        # C_left corresponds to k=0,1.
        for k in range(2):
            for i in range(2):
                W[k*2+i, 0:7] = W1[k*2+i, :]
        
        # Copy next 7 columns from U2, V2, W2
        U[:, 7:14] = U2
        # Map V2 to V_right
        for j in range(2):
            for k in range(2):
                V[j*4+(k+2), 7:14] = V2[j*2+k, :]
        # Map W2 to W_right
        for k in range(2):
            for i in range(2):
                W[(k+2)*2+i, 7:14] = W2[k*2+i, :]
                
        return U, V, W

    # Initialize main matrices
    # We don't know the final rank yet. We can estimate or build dynamically.
    # Rank = 7 * 14 = 98.
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Recursive step for 4x4 * 4x8
    # A is 4x4, B is 4x8, C is 4x8.
    # n=4, m=4, p=8.
    # n/2=2, m/2=2, p/2=4.
    
    # P1: (A11+A22)(B11+B22)
    # A11: rows 0-1, cols 0-1. A22: rows 2-3, cols 2-3.
    # B11: rows 0-1, cols 0-3. B22: rows 2-3, cols 4-7.
    # Sub problem: 2x2 * 2x4.
    sub_U, sub_V, sub_W = base_2_2_2_4()
    
    # We need to add this decomposition twice? No, the sub-decomposition is for the product of the sums.
    # The sub-decomposition computes $A' B'$ where $A' = A_{11}+A_{22}$ and $B' = B_{11}+B_{22}$.
    # The U matrix of the sub-decomposition has coefficients for $A'$'s entries.
    # $A'$'s entries are linear combinations of $A$'s entries.
    # So we need to map the sub-U indices to global U indices with signs.
    
    # This is tricky with the `add_decomp` helper because it assumes the sub-decomposition is for a specific block.
    # Here, the sub-decomposition is for a *virtual* matrix $A'$.
    
    # Let's modify the approach:
    # Instead of a generic add_decomp, we will manually map the 14 terms of the base case for each of the 7 Strassen products.
    
    # This is complex. Let's use a simpler base case: Schoolbook 2x2*2x4 (rank 16).
    # Total rank 7 * 16 = 112.
    # This is still better than 128.
    # And much easier to implement correctly.
    
    def base_2_2_2_4_schoolbook():
        U = np.zeros((4, 16))
        V = np.zeros((8, 16))
        W = np.zeros((8, 16))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    U[i*2+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W

    U_base, V_base, W_base = base_2_2_2_4_schoolbook()
    
    # Define the 7 Strassen products
    # Each product is a 2x2 * 2x4 multiplication.
    # We will call add_decomp 7 times, each with the base decomposition.
    
    # Product 1: (A11 + A22) * (B11 + B22)
    # A' = A11 + A22.
    # A11 is at rows 0-1, cols 0-1.
    # A22 is at rows 2-3, cols 2-3.
    # The base U has 4 rows (2x2).
    # U[0] corresponds to A'[0,0] = A[0,0] + A[2,2]
    # U[1] corresponds to A'[0,1] = A[0,1] + A[2,3]
    # U[2] corresponds to A'[1,0] = A[1,0] + A[3,2]
    # U[3] corresponds to A'[1,1] = A[1,1] + A[3,3]
    
    # We need a helper that takes the base decomposition and a mapping of its U/V/W indices to global indices with signs.
    
    def add_mapped_decomp(base_U, base_V, base_W, 
                          u_map, v_map, w_map, 
                          rank_offset):
        # u_map: list of (global_idx, sign) for each of the 4 U rows
        # v_map: list of (global_idx, sign) for each of the 8 V rows
        # w_map: list of (global_idx, sign) for each of the 8 W rows
        
        for r in range(base_U.shape[1]):
            for i in range(4):
                val = base_U[i, r]
                if val != 0:
                    g_idx, sign = u_map[i]
                    U[g_idx][rank_offset + r] += sign * val
            
            for i in range(8):
                val = base_V[i, r]
                if val != 0:
                    g_idx, sign = v_map[i]
                    V[g_idx][rank_offset + r] += sign * val
            
            for i in range(8):
                val = base_W[i, r]
                if val != 0:
                    g_idx, sign = w_map[i]
                    W[g_idx][rank_offset + r] += sign * val

    # P1: (A11+A22)(B11+B22)
    # A11: (0,0),(0,1),(1,0),(1,1)
    # A22: (2,2),(2,3),(3,2),(3,3)
    # u_map for A':
    # A'[0,0] -> A[0,0] (+1), A[2,2] (+1)
    # A'[0,1] -> A[0,1] (+1), A[2,3] (+1)
    # A'[1,0] -> A[1,0] (+1), A[3,2] (+1)
    # A'[1,1] -> A[1,1] (+1), A[3,3] (+1)
    # But base_U is 4x16. Each row of base_U is a coefficient for one entry of A'.
    # We need to distribute these coefficients to the global U.
    # This means for each r, we add to U[global_idx] the sum of base_U[i,r] * sign for all i mapping to global_idx.
    # My `add_mapped_decomp` above assumes a 1-to-1 mapping.
    # I need to handle many-to-one mappings.
    
    def add_decomp_multi(base_U, base_V, base_W, 
                         u_maps, v_maps, w_maps, 
                         rank_offset):
        # u_maps: list of 4 lists of (global_idx, sign)
        # Each base_U[i, r] is added to all (g_idx, sign) in u_maps[i]
        
        for r in range(base_U.shape[1]):
            for i in range(4):
                val = base_U[i, r]
                if val != 0:
                    for g_idx, sign in u_maps[i]:
                        U[g_idx][rank_offset + r] += sign * val
            
            for i in range(8):
                val = base_V[i, r]
                if val != 0:
                    for g_idx, sign in v_maps[i]:
                        V[g_idx][rank_offset + r] += sign * val
            
            for i in range(8):
                val = base_W[i, r]
                if val != 0:
                    for g_idx, sign in w_maps[i]:
                        W[g_idx][rank_offset + r] += sign * val

    # P1: (A11+A22)(B11+B22)
    u_maps_p1 = [
        [(0*4+0, 1), (2*4+2, 1)], # A'[0,0]
        [(0*4+1, 1), (2*4+3, 1)], # A'[0,1]
        [(1*4+0, 1), (3*4+2, 1)], # A'[1,0]
        [(1*4+1, 1), (3*4+3, 1)]  # A'[1,1]
    ]
    # B11: rows 0-1, cols 0-3. B22: rows 2-3, cols 4-7.
    # B'[0,0] = B[0,0] + B[2,4]
    # B'[0,1] = B[0,1] + B[2,5]
    # ...
    v_maps_p1 = []
    for j in range(2):
        for k in range(4):
            # B' index j*4+k
            # B11 index j*4+k
            # B22 index (j+2)*4 + (k+4)
            v_maps_p1.append([(j*4+k, 1), ((j+2)*4+(k+4), 1)])
            
    # C' = C11 + C22? No, C' is the result of P1.
    # C11 is at rows 0-1, cols 0-3. C22 is at rows 2-3, cols 4-7.
    # C'[0,0] -> C[0,0]
    # C'[0,1] -> C[0,1]
    # ...
    # Wait, the output C is 4x8.
    # P1 contributes to C11 and C22?
    # No, P1 is one of the 7 products.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # So P1 contributes to C11 and C22.
    # C11 is 2x4. C22 is 2x4.
    # The base_W is 8x16 (2x4).
    # We need to map base_W to C11 and C22.
    # But P1 is a single 2x4 matrix.
    # It contributes to C11 (top-left) and C22 (bottom-right)?
    # No, in Strassen, the 7 products are 2x2 matrices.
    # Here, the products are 2x4 matrices.
    # P1 is 2x4.
    # It contributes to C11 (2x4) and C22 (2x4)?
    # C11 is the top-left 2x4 block of C.
    # C22 is the bottom-right 2x4 block of C.
    # So P1 contributes to both?
    # Yes, C11 = P1 + ... and C22 = P1 - ...
    # So the same P1 values are added to C11 and C22 with different signs.
    
    # This means we cannot just add the base_W once.
    # We need to add it to C11 with sign +1 and to C22 with sign +1 (for P1 term in C11) and -1 (for P1 term in C22)?
    # No, the decomposition is a sum of rank-1 tensors.
    # Each rank-1 tensor is $u \otimes v \otimes w$.
    # If P1 is computed, it is a matrix.
    # The contribution of P1 to C11 is $+1 \cdot P1$.
    # The contribution of P1 to C22 is $+1 \cdot P1$ (in the formula C22 = P1 - P2 + P3 + P6).
    
    # So for each rank-1 term in P1's decomposition, we add it to C11 and C22.
    
    w_maps_p1 = []
    for k in range(4):
        for i in range(2):
            # C' index k*2+i
            # C11 index: row i, col k. Global index k*4 + i.
            # C22 index: row i+2, col k+4. Global index (k+4)*4 + (i+2).
            w_maps_p1.append([(k*4+i, 1), ((k+4)*4+(i+2), 1)])
            
    add_decomp_multi(U_base, V_base, W_base, u_maps_p1, v_maps_p1, w_maps_p1, 0)

    # P2: (A21+A22)B11
    # A21: rows 2-3, cols 0-1. A22: rows 2-3, cols 2-3.
    # A' = A21 + A22.
    u_maps_p2 = [
        [(2*4+0, 1), (2*4+2, 1)], # A'[0,0] = A[2,0]+A[2,2]
        [(2*4+1, 1), (2*4+3, 1)], # A'[0,1] = A[2,1]+A[2,3]
        [(3*4+0, 1), (3*4+2, 1)], # A'[1,0] = A[3,0]+A[3,2]
        [(3*4+1, 1), (3*4+3, 1)]  # A'[1,1] = A[3,1]+A[3,3]
    ]
    # B11: rows 0-1, cols 0-3.
    v_maps_p2 = []
    for j in range(2):
        for k in range(4):
            v_maps_p2.append([(j*4+k, 1)])
            
    # P2 contributes to C21 (bottom-left) and C22 (bottom-right).
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    w_maps_p2 = []
    for k in range(4):
        for i in range(2):
            # C21: row i+2, col k. Global (k)*4 + (i+2).
            # C22: row i+2, col k+4. Global (k+4)*4 + (i+2).
            w_maps_p2.append([(k*4+(i+2), 1), ((k+4)*4+(i+2), -1)])
            
    add_decomp_multi(U_base, V_base, W_base, u_maps_p2, v_maps_p2, w_maps_p2, 16)

    # P3: A11(B12-B22)
    u_maps_p3 = [
        [(0*4+0, 1)], # A[0,0]
        [(0*4+1, 1)], # A[0,1]
        [(1*4+0, 1)], # A[1,0]
        [(1*4+1, 1)]  # A[1,1]
    ]
    # B12: rows 0-1, cols 4-7. B22: rows 2-3, cols 4-7.
    v_maps_p3 = []
    for j in range(2):
        for k in range(4):
            # B' = B12 - B22
            # B12[j, k] is at global j*4 + (k+4)
            # B22[j, k] is at global (j+2)*4 + (k+4)
            v_maps_p3.append([(j*4+(k+4), 1), ((j+2)*4+(k+4), -1)])
            
    # P3 contributes to C12 (top-right) and C22 (bottom-right).
    # C12 = P3 + P5
    # C22 = P1 - P2 + P3 + P6
    w_maps_p3 = []
    for k in range(4):
        for i in range(2):
            # C12: row i, col k+4. Global (k+4)*4 + i.
            # C22: row i+2, col k+4. Global (k+4)*4 + (i+2).
            w_maps_p3.append([((k+4)*4+i, 1), ((k+4)*4+(i+2), 1)])
            
    add_decomp_multi(U_base, V_base, W_base, u_maps_p3, v_maps_p3, w_maps_p3, 32)

    # P4: A22(B21-B11)
    u_maps_p4 = [
        [(2*4+2, 1)], # A[2,2]
        [(2*4+3, 1)], # A[2,3]
        [(3*4+2, 1)], # A[3,2]
        [(3*4+3, 1)]  # A[3,3]
    ]
    # B21: rows 2-3, cols 0-3. B11: rows 0-1, cols 0-3.
    v_maps_p4 = []
    for j in range(2):
        for k in range(4):
            # B' = B21 - B11
            # B21[j, k] is at global (j+2)*4 + k
            # B11[j, k] is at global j*4 + k
            v_maps_p4.append([((j+2)*4+k, 1), (j*4+k, -1)])
            
    # P4 contributes to C11 (top-left) and C21 (bottom-left).
    # C11 = P1 + P4 - P5 + P7
    # C21 = P2 + P4
    w_maps_p4 = []
    for k in range(4):
        for i in range(2):
            # C11: row i, col k. Global k*4 + i.
            # C21: row i+2, col k. Global k*4 + (i+2).
            w_maps_p4.append([(k*4+i, 1), (k*4+(i+2), 1)])
            
    add_decomp_multi(U_base, V_base, W_base, u_maps_p4, v_maps_p4, w_maps_p4, 48)

    # P5: (A11+A12)B22
    u_maps_p5 = [
        [(0*4+0, 1), (0*4+2, 1)], # A[0,0]+A[0,2]
        [(0*4+1, 1), (0*4+3, 1)], # A[0,1]+A[0,3]
        [(1*4+0, 1), (1*4+2, 1)], # A[1,0]+A[1,2]
        [(1*4+1, 1), (1*4+3, 1)]  # A[1,1]+A[1,3]
    ]
    # B22: rows 2-3, cols 4-7.
    v_maps_p5 = []
    for j in range(2):
        for k in range(4):
            v_maps_p5.append([((j+2)*4+(k+4), 1)])
            
    # P5 contributes to C11 (top-left) and C12 (top-right).
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    w_maps_p5 = []
    for k in range(4):
        for i in range(2):
            # C11: row i, col k. Global k*4 + i.
            # C12: row i, col k+4. Global (k+4)*4 + i.
            w_maps_p5.append([(k*4+i, -1), ((k+4)*4+i, 1)])
            
    add_decomp_multi(U_base, V_base, W_base, u_maps_p5, v_maps_p5, w_maps_p5, 64)

    # P6: (A21-A11)(B11+B12)
    u_maps_p6 = [
        [(2*4+0, 1), (0*4+0, -1)], # A[2,0]-A[0,0]
        [(2*4+1, 1), (0*4+1, -1)], # A[2,1]-A[0,1]
        [(3*4+0, 1), (1*4+0, -1)], # A[3,0]-A[1,0]
        [(3*4+1, 1), (1*4+1, -1)]  # A[3,1]-A[1,1]
    ]
    # B11: rows 0-1, cols 0-3. B12: rows 0-1, cols 4-7.
    v_maps_p6 = []
    for j in range(2):
        for k in range(4):
            # B' = B11 + B12
            # B11[j, k] is at global j*4 + k
            # B12[j, k] is at global j*4 + (k+4)
            v_maps_p6.append([(j*4+k, 1), (j*4+(k+4), 1)])
            
    # P6 contributes to C22 (bottom-right).
    # C22 = P1 - P2 + P3 + P6
    w_maps_p6 = []
    for k in range(4):
        for i in range(2):
            # C22: row i+2, col k+4. Global (k+4)*4 + (i+2).
            w_maps_p6.append([((k+4)*4+(i+2), 1)])
            
    add_decomp_multi(U_base, V_base, W_base, u_maps_p6, v_maps_p6, w_maps_p6, 80)

    # P7: (A12-A22)(B21+B22)
    u_maps_p7 = [
        [(0*4+2, 1), (2*4+2, -1)], # A[0,2]-A[2,2]
        [(0*4+3, 1), (2*4+3, -1)], # A[0,3]-A[2,3]
        [(1*4+2, 1), (3*4+2, -1)], # A[1,2]-A[3,2]
        [(1*4+3, 1), (3*4+3, -1)]  # A[1,3]-A[3,3]
    ]
    # B21: rows 2-3, cols 0-3. B22: rows 2-3, cols 4-7.
    v_maps_p7 = []
    for j in range(2):
        for k in range(4):
            # B' = B21 + B22
            # B21[j, k] is at global (j+2)*4 + k
            # B22[j, k] is at global (j+2)*4 + (k+4)
            v_maps_p7.append([((j+2)*4+k, 1), ((j+2)*4+(k+4), 1)])
            
    # P7 contributes to C11 (top-left).
    # C11 = P1 + P4 - P5 + P7
    w_maps_p7 = []
    for k in range(4):
        for i in range(2):
            # C11: row i, col k. Global k*4 + i.
            w_maps_p7.append([(k*4+i, 1)])
            
    add_decomp_multi(U_base, V_base, W_base, u_maps_p7, v_maps_p7, w_maps_p7, 96)

    return {"U": U, "V": V, "W": W}
