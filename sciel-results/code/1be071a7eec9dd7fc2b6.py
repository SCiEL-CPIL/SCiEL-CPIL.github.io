import time
import numpy as np

def construct():
    n, m, p = 2, 5, 5
    
    # Total rank estimate
    R = 46
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Helper to add a term
    # u_idx: global index in U (i*m + j)
    # v_idx: global index in V (j*p + k)
    # w_idx: global index in W (k*n + i)
    # coeff: coefficient for this rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff
        col += 1

    # Strassen 2x2x2
    # Local A: (0,0) (0,1) (1,0) (1,1)
    # Local B: (0,0) (0,1) (1,0) (1,1)
    # Local C: (0,0) (0,1) (1,0) (1,1)
    def add_strassen_222(a_off, b_off, c_off):
        # a_off: offset for A indices (j offset)
        # b_off: offset for B indices (j' offset for rows, k offset for cols)
        # b_row_off, b_col_off
        # c_off: offset for C indices (k' offset)
        
        # M1 = (A00+A11)(B00+B11)
        add_term(a_off*1 + 0, b_row_off*5 + b_col_off + 0, (b_col_off + 0)*2 + 0, 1) # C00 part? No, Strassen sums later.
        # This approach is tricky because Strassen is a linear combination.
        # Better to explicitly list the 7 multiplications and their contributions to C.
        
        pass

    # Let's just hardcode the 7 Strassen terms as rank-1 tensors in the sum.
    # The decomposition is sum of rank-1 tensors.
    # Strassen uses 7 multiplications.
    # The output C is linear combination of these 7 products.
    # So we can create 7 rank-1 tensors for the 7 products, and then the W matrix will have the coefficients for the linear combination.
    
    # Strassen Products:
    # P1 = (A00+A11) * (B00+B11)
    # P2 = (A10+A11) * B00
    # P3 = A00 * (B10-B11)
    # P4 = A11 * (B01-B00)
    # P5 = (A00+A01) * B11
    # P6 = (A10-A00) * (B00+B01)
    # P7 = (A01-A11) * (B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map these to global indices.
    # For a block A_i (rows 0-1 of A, cols j0..j1) and B_ij (rows j0..j1 of B, cols k0..k1)
    
    def add_strassen_block(a_j_start, b_j_start, b_k_start):
        # a_j_start: start col index in A (0, 2, 4)
        # b_j_start: start row index in B (0, 2, 4)
        # b_k_start: start col index in B (0, 2, 4)
        
        # P1
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        # U[0*5 + a_j_start] += 1
        # U[1*5 + a_j_start+1] += 1
        # V[0*5 + b_k_start] += 1
        # V[1*5 + b_k_start+1] += 1
        # W[0*2 + 0] += 1
        # W[1*2 + 1] += 1
        
        # This is getting complicated to manage with `col` pointer if I do it one by one.
        # I will just create the matrices for each block and add them.
        
        pass

    # Simpler: Generate the U, V, W for each of the 9 blocks and concatenate/sum.
    
    blocks = [
        # (a_j_start, b_j_start, b_k_start, n_sub, m_sub, p_sub)
        (0, 0, 0, 2, 2, 2), # A0 B00
        (0, 2, 0, 2, 2, 2), # A0 B01 (B rows 0-1, Cols 2-3? No. B is 5x5. 
                            # My split:
                            # A: 2x2, 2x2, 2x1. Cols 0-1, 2-3, 4.
                            # B: Rows 0-1, 2-3, 4. Cols 0-1, 2-3, 4.
                            # B00: R0-1, C0-1.
                            # B01: R0-1, C2-3.
                            # B02: R0-1, C4.
                            # B10: R2-3, C0-1.
                            # B11: R2-3, C2-3.
                            # B12: R2-3, C4.
                            # B20: R4, C0-1.
                            # B21: R4, C2-3.
                            # B22: R4, C4.
        ]
        
    # Let's refine the block list
    # A blocks:
    # A0: cols 0-1
    # A1: cols 2-3
    # A2: col 4
    
    # B blocks:
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B02: rows 0-1, col 4
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    # B12: rows 2-3, col 4
    # B20: row 4, cols 0-1
    # B21: row 4, cols 2-3
    # B22: row 4, col 4
    
    # Terms:
    # A0 B00: 2x2x2
    # A0 B01: 2x2x2
    # A0 B02: 2x2x1
    # A1 B10: 2x2x2
    # A1 B11: 2x2x2
    # A1 B12: 2x2x1
    # A2 B20: 2x1x2
    # A2 B21: 2x1x2
    # A2 B22: 2x1x1
    
    # I will write a helper to generate U, V, W for a specific sub-problem and add to global.
    
    def get_strassen_222():
        # Returns U, V, W for 2x2x2 Strassen
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # P2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # P3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # P4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # P5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # P6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # P7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_221():
        # 2x2x1
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    def get_212():
        # 2x1x2
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        return U, V, W

    def get_211():
        # 2x1x1
        U = np.zeros((2, 2))
        V = np.zeros((1, 2))
        W = np.zeros((2, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[0, 1] = 1; W[1, 1] = 1
        return U, V, W

    # Mapping function
    # u_local (i, j) -> global (i, j+a_j_start)
    # v_local (j', k) -> global (j'+b_j_start, k+b_k_start)
    # w_local (k', i) -> global (k'+b_k_start, i)
    
    def add_block(a_j_start, b_j_start, b_k_start, U_sub, V_sub, W_sub):
        nonlocal col
        n_sub, m_sub = U_sub.shape[0] // 2, U_sub.shape[0] % 2 # This is wrong, U is (n*m, R)
        # U_sub shape (n_sub*m_sub, R_sub)
        # V_sub shape (m_sub*p_sub, R_sub)
        # W_sub shape (p_sub*n_sub, R_sub)
        
        R_sub = U_sub.shape[1]
        
        for r in range(R_sub):
            # U
            for u_idx in range(U_sub.shape[0]):
                i = u_idx // m_sub
                j = u_idx % m_sub
                g_u = i * 5 + (j + a_j_start)
                U[g_u, col + r] += U_sub[u_idx, r]
                
            # V
            for v_idx in range(V_sub.shape[0]):
                j_prime = v_idx // p_sub
                k = v_idx % p_sub
                g_v = (j_prime + b_j_start) * 5 + (k + b_k_start)
                V[g_v, col + r] += V_sub[v_idx, r]
                
            # W
            for w_idx in range(W_sub.shape[0]):
                k_prime = w_idx // n_sub
                i = w_idx % n_sub
                g_w = (k_prime + b_k_start) * 2 + i
                W[g_w, col + r] += W_sub[w_idx, r]
                
        col += R_sub

    # Execute
    U0, V0, W0 = get_strassen_222()
    add_block(0, 0, 0, U0, V0, W0) # A0 B00
    
    U1, V1, W1 = get_strassen_222()
    add_block(0, 0, 2, U1, V1, W1) # A0 B01 (B cols 2-3)
    
    U2, V2, W2 = get_221()
    add_block(0, 0, 4, U2, V2, W2) # A0 B02 (B col 4)
    
    U3, V3, W3 = get_strassen_222()
    add_block(2, 2, 0, U3, V3, W3) # A1 B10 (A cols 2-3, B rows 2-3)
    
    U4, V4, W4 = get_strassen_222()
    add_block(2, 2, 2, U4, V4, W4) # A1 B11
    
    U5, V5, W5 = get_221()
    add_block(2, 2, 4, U5, V5, W5) # A1 B12
    
    U6, V6, W6 = get_212()
    add_block(4, 4, 0, U6, V6, W6) # A2 B20 (A col 4, B row 4, B cols 0-1)
    
    U7, V7, W7 = get_212()
    add_block(4, 4, 2, U7, V7, W7) # A2 B21
    
    U8, V8, W8 = get_211()
    add_block(4, 4, 4, U8, V8, W8) # A2 B22
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

