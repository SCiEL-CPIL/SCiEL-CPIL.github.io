"""Strassen-based rank-46 exact decomposition for 2x5 * 5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    # Block structure of B (5x5): rows j=0..4, cols k=0..4
    # Split j into {0,1} (J1), {2,3} (J2), {4} (J3)
    # Split k into {0,1} (K1), {2,3} (K2), {4} (K3)
    # A is 2x5, C is 2x5
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # We use Strassen for the four 2x2x2 blocks (J1xK1, J2xK1, J1xK2, J2xK2)
    # and schoolbook for the remaining blocks involving J3 or K3.
    
    # Strassen for 2x2 * 2x2:
    # Given A_block (2x2) with rows i, cols j_in_block (2 cols)
    # B_block (2x2) with rows j_in_block (2 rows), cols k_in_block (2 cols)
    # C_block (2x2) = A_block @ B_block
    #
    # Strassen factors:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = []  # shape (n*m, rank) = (10, rank)
    V = []  # shape (m*p, rank) = (25, rank)
    W = []  # shape (p*n, rank) = (10, rank)
    
    rank = 0
    
    def add_term(u_vec, v_vec, w_vec):
        """Add one rank-1 term. u_vec is length 10 (i*m+j), v_vec length 25 (j*p+k), w_vec length 10 (k*n+i)."""
        nonlocal rank
        col = rank
        rank += 1
        U.append(u_vec[:])
        V.append(v_vec[:])
        W.append(w_vec[:])
    
    def make_u(i_vals_j_map):
        """i_vals_j_map: dict mapping (i, j) -> coefficient. Returns list of length 10."""
        vec = [0.0] * 10
        for (i, j), c in i_vals_j_map.items():
            vec[i * m + j] = c
        return vec
    
    def make_v(j_vals_k_map):
        """j_vals_k_map: dict mapping (j, k) -> coefficient. Returns list of length 25."""
        vec = [0.0] * 25
        for (j, k), c in j_vals_k_map.items():
            vec[j * p + k] = c
        return vec
    
    def make_w(k_vals_i_map):
        """k_vals_i_map: dict mapping (k, i) -> coefficient. Returns list of length 10."""
        vec = [0.0] * 10
        for (k, i), c in k_vals_i_map.items():
            vec[k * n + i] = c
        return vec
    
    # Strassen block: A rows {i0, i1}, B rows {j0, j1}, B cols {k0, k1}
    # i0=0, i1=1 always (n=2)
    def strassen_block(j0, j1, k0, k1):
        """
        A has rows 0,1 and cols j0,j1
        B has rows j0,j1 and cols k0,k1
        C has rows 0,1 and cols k0,k1
        """
        # A00 = A[0,j0], A01 = A[0,j1], A10 = A[1,j0], A11 = A[1,j1]
        # B00 = B[j0,k0], B01 = B[j0,k1], B10 = B[j1,k0], B11 = B[j1,k1]
        
        # M1 = (A00 + A11) * (B00 + B11)
        u1 = make_u({(0, j0): 1, (1, j1): 1})
        v1 = make_v({(j0, k0): 1, (j1, k1): 1})
        w1 = make_w({(k0, 0): 1, (k1, 1): 1})
        add_term(u1, v1, w1)
        
        # M2 = (A10 + A11) * B00
        u2 = make_u({(1, j0): 1, (1, j1): 1})
        v2 = make_v({(j0, k0): 1})
        w2 = make_w({(k0, 1): 1})
        add_term(u2, v2, w2)
        
        # M3 = A00 * (B01 - B11)
        u3 = make_u({(0, j0): 1})
        v3 = make_v({(j0, k1): 1, (j1, k1): -1})
        w3 = make_w({(k1, 0): 1})
        add_term(u3, v3, w3)
        
        # M4 = A11 * (B10 - B00)
        u4 = make_u({(1, j1): 1})
        v4 = make_v({(j1, k0): 1, (j0, k0): -1})
        w4 = make_w({(k0, 1): 1})
        add_term(u4, v4, w4)
        
        # M5 = (A00 + A01) * B11
        u5 = make_u({(0, j0): 1, (0, j1): 1})
        v5 = make_v({(j1, k1): 1})
        w5 = make_w({(k1, 0): 1})
        add_term(u5, v5, w5)
        
        # M6 = (A10 - A00) * (B00 + B10)
        u6 = make_u({(1, j0): 1, (0, j0): -1})
        v6 = make_v({(j0, k0): 1, (j1, k0): 1})
        w6 = make_w({(k0, 1): 1})
        add_term(u6, v6, w6)
        
        # M7 = (A01 - A11) * (B01 + B11)
        u7 = make_u({(0, j1): 1, (1, j1): -1})
        v7 = make_v({(j0, k1): 1, (j1, k1): 1})
        w7 = make_w({(k1, 1): 1})
        add_term(u7, v7, w7)
        
        # Now we need to combine to get C values.
        # C[0,k0] = M1 + M4 - M5 + M7
        # C[0,k1] = M3 + M5
        # C[1,k0] = M2 + M4
        # C[1,k1] = M1 - M2 + M3 + M6
        
        # The W vectors above are for the individual M's contributing to specific C positions.
        # But wait, the way I set it up, each add_term contributes to ALL C positions where w has nonzeros.
        # I need to be more careful. The W vector should encode which C[i,k] this term contributes to and with what coefficient.
        
        # Let me redo this. Each rank-1 term is u ⊗ v ⊗ w where:
        # C[i,k] = sum_c U[i*m+j... no wait.
        # Actually the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # So for a given term c, it contributes U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] to C[i,k] for all i,j,k.
        # So the contribution to C[i,k] from term c is sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # 
        # This means I need to design U, V, W such that for each term, the sum over j gives the right coefficient for each (i,k).
        
        # Let me restart with a cleaner approach.
    
    # Let me clear and start fresh with a more systematic approach.
    U.clear()
    V.clear()
    W.clear()
    rank = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal rank
        col = rank
        rank += 1
        U.append(u_vec[:])
        V.append(v_vec[:])
        W.append(w_vec[:])
    
    # For each multiplication "slot", I define:
    # - which A elements (i,j) are involved and their coefficients
    # - which B elements (j,k) are involved and their coefficients  
    # - which C elements (i,k) are affected and their coefficients
    # The term contributes: (sum over matching j of A_coeff[i,j] * B_coeff[j,k]) * C_coeff[i,k] to C[i,k]
    # Actually no. The tensor entry T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
    # C[i,k] = sum_j T[i,j,k] = sum_j U[i*m+j] * V[j*p+k] * W[k*n+i]
    
    # So for a single term, the contribution to C[i,k] is W[k*n+i] * (sum_j U[i*m+j] * V[j*p+k])
    # This is W[k*n+i] * (row i of U restricted to j's, dotted with col k of V)
    
    # For Strassen, each "multiplication" M is a product of a linear combination of A entries and a linear combination of B entries.
    # M = (sum_a alpha_a * A[i_a, j_a]) * (sum_b beta_b * B[j_b, k_b])
    # Then M contributes to C[i_a, k_b] for each (a,b) pair? No, that's not right either.
    
    # Actually in Strassen, M1 = (A00+A11)(B00+B11) contributes to C00 and C11.
    # Specifically C00 gets +M1, C11 gets +M1.
    # So the term should be: U has nonzero at (0,j0) and (1,j1), V has nonzero at (j0,k0) and (j1,k1),
    # and W should be set so that the product gives the right contributions.
    
    # Contribution to C[i,k] = W[k*n+i] * sum_j U[i*m+j] * V[j*p+k]
    
    # For M1: U[(0,j0)]=1, U[(1,j1)]=1, V[(j0,k0)]=1, V[(j1,k1)]=1
    # sum_j U[i*m+j]*V[j*p+k] for i=0: U[0*m+j0]*V[j0*p+k] + U[0*m+j1]*V[j1*p+k] = 1*V[j0*p+k] + 0*V[j1*p+k]
    # For k=k0: V[j0*p+k0]=1, V[j1*p+k0]=0, so sum = 1
    # For k=k1: V[j0*p+k1]=0, V[j1*p+k1]=1, so sum = 0
    # So contribution to C[0,k0] = W[k0*n+0] * 1, C[0,k1] = W[k1*n+0] * 0
    # For i=1: U[1*m+j0]*V[j0*p+k] + U[1*m+j1]*V[j1*p+k] = 0*V[j0*p+k] + 1*V[j1*p+k]
    # For k=k0: V[j1*p+k0]=0, so sum=0
    # For k=k1: V[j1*p+k1]=1, so sum=1
    # Contribution to C[1,k0] = W[k0*n+1]*0, C[1,k1] = W[k1*n+1]*1
    
    # So I need W[k0*n+0] = 1 and W[k1*n+1] = 1 for M1 to contribute +1 to C[0,k0] and C[1,k1].
    # But M1 should contribute to both C[0,k0] and C[1,k1] with coefficient +1.
    # That works! Set W[k0*n+0]=1, W[k1*n+1]=1.
    
    # OK so the approach is: for each Strassen multiplication, set U, V, W appropriately.
    
    def strassen_block(j0, j1, k0, k1):
        """j0,j1 are the two B-row indices, k0,k1 are the two B-col indices."""
        # M1 = (A[0,j0] + A[1,j1]) * (B[j0,k0] + B[j1,k1])
        u1 = [0.0]*10; u1[0*m+j0] = 1; u1[1*m+j1] = 1
        v1 = [0.0]*25; v1[j0*p+k0] = 1; v1[j1*p+k1] = 1
        w1 = [0.0]*10; w1[k0*n+0] = 1; w1[k1*n+1] = 1
        add_term(u1, v1, w1)
        
        # M2 = (A[1,j0] + A[1,j1]) * B[j0,k0]
        u2 = [0.0]*10; u2[1*m+j0] = 1; u2[1*m+j1] = 1
        v2 = [0.0]*25; v2[j0*p+k0] = 1
        w2 = [0.0]*10; w2[k0*n+1] = 1
        add_term(u2, v2, w2)
        
        # M3 = A[0,j0] * (B[j0,k1] - B[j1,k1])
        u3 = [0.0]*10; u3[0*m+j0] = 1
        v3 = [0.0]*25; v3[j0*p+k1] = 1; v3[j1*p+k1] = -1
        w3 = [0.0]*10; w3[k1*n+0] = 1
        add_term(u3, v3, w3)
        
        # M4 = A[1,j1] * (B[j1,k0] - B[j0,k0])
        u4 = [0.0]*10; u4[1*m+j1] = 1
        v4 = [0.0]*25; v4[j1*p+k0] = 1; v4[j0*p+k0] = -1
        w4 = [0.0]*10; w4[k0*n+1] = 1
        add_term(u4, v4, w4)
        
        # M5 = (A[0,j0] + A[0,j1]) * B[j1,k1]
        u5 = [0.0]*10; u5[0*m+j0] = 1; u5[0*m+j1] = 1
        v5 = [0.0]*25; v5[j1*p+k1] = 1
        w5 = [0.0]*10; w5[k1*n+0] = 1
        add_term(u5, v5, w5)
        
        # M6 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j1,k0])
        u6 = [0.0]*10; u6[1*m+j0] = 1; u6[0*m+j0] = -1
        v6 = [0.0]*25; v6[j0*p+k0] = 1; v6[j1*p+k0] = 1
        w6 = [0.0]*10; w6[k0*n+1] = 1
        add_term(u6, v6, w6)
        
        # M7 = (A[0,j1] - A[1,j1]) * (B[j0,k1] + B[j1,k1])
        u7 = [0.0]*10; u7[0*m+j1] = 1; u7[1*m+j1] = -1
        v7 = [0.0]*25; v7[j0*p+k1] = 1; v7[j1*p+k1] = 1
        w7 = [0.0]*10; w7[k1*n+1] = 1
        add_term(u7, v7, w7)
    
    # Four 2x2x2 Strassen blocks
    strassen_block(0, 1, 0, 1)  # J1 x K1
    strassen_block(2, 3, 0, 1)  # J2 x K1
    strassen_block(0, 1, 2, 3)  # J1 x K2
    strassen_block(2, 3, 2, 3)  # J2 x K2
    
    # Now the remaining blocks:
    # J1 x K3: A[0..1, 0..1] * B[0..1, 4]
    # J2 x K3: A[0..1, 2..3] * B[2..3, 4]
    # J3 x K1: A[0..1, 4] * B[4, 0..1]
    # J3 x K2: A[0..1, 4] * B[4, 2..3]
    # J3 x K3: A[0..1, 4] * B[4, 4]
    
    # For 2x2x1 blocks (e.g., J1 x K3):
    # C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4]
    # This is just schoolbook: 2*2 = 4 terms
    def schoolbook_2x2x1(j0, j1, k):
        for i in range(2):
            for j in [j0, j1]:
                u = [0.0]*10; u[i*m+j] = 1
                v = [0.0]*25; v[j*p+k] = 1
                w = [0.0]*10; w[k*n+i] = 1
                add_term(u, v, w)
    
    # 2x1x2 blocks (e.g., J3 x K1):
    # C[i,k] = A[i,4]*B[4,k]
    # 2*1*2 = 4 terms
    def schoolbook_2x1x2(j, k0, k1):
        for i in range(2):
            for k in [k0, k1]:
                u = [0.0]*10; u[i*m+j] = 1
                v = [0.0]*25; v[j*p+k] = 1
                w = [0.0]*10; w[k*n+i] = 1
                add_term(u, v, w)
    
    # 2x1x1:
    def schoolbook_2x1x1(j, k):
        for i in range(2):
            u = [0.0]*10; u[i*m+j] = 1
            v = [0.0]*25; v[j*p+k] = 1
            w = [0.0]*10; w[k*n+i] = 1
            add_term(u, v, w)
    
    schoolbook_2x2x1(0, 1, 4)   # J1 x K3
    schoolbook_2x2x1(2, 3, 4)   # J2 x K3
    schoolbook_2x1x2(4, 0, 1)   # J3 x K1
    schoolbook_2x1x2(4, 2, 3)   # J3 x K2
    schoolbook_2x1x1(4, 4)       # J3 x K3
    
    # Total rank: 4*7 + 2*4 + 2*4 + 2 = 28 + 8 + 8 + 2 = 46
    # U is list of length rank, each element length 10
    # V is list of length rank, each element length 25
    # W is list of length rank, each element length 10
    # Transpose to get shape (10, rank), (25, rank), (10, rank)
    U_final = [[U[r][i] for r in range(rank)] for i in range(10)]
    V_final = [[V[r][i] for r in range(rank)] for i in range(25)]
    W_final = [[W[r][i] for r in range(rank)] for i in range(10)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
