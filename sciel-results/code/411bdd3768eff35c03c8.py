import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Returns a rank-36 tensor decomposition for 2x4 by 4x5 matrix multiplication.
    Strategy: 
    1. Split output p=5 into p1=4 and p2=1.
    2. C = C_4 + C_1.
    3. C_1 (2x1) uses schoolbook rank 8.
    4. C_4 (2x4) is computed as A_L B_T + A_R B_B where A_L, A_R are 2x2 and B_T, B_B are 2x4.
       This requires two 2x2x4 multiplications.
    5. Each 2x2x4 is split into two 2x2x2 multiplications using Strassen (rank 7 each).
       Total rank for C_4 is 7 + 7 + 7 + 7 = 28.
    6. Total rank = 28 + 8 = 36.
    """
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (8, 36)
    # V shape: (m*p, rank) = (20, 36)
    # W shape: (p*n, rank) = (10, 36)
    
    U = np.zeros((n * m, 36))
    V = np.zeros((m * p, 36))
    W = np.zeros((p * n, 36))
    
    rank = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal rank
        U[:, rank] = u_vec
        V[:, rank] = v_vec
        W[:, rank] = w_vec
        rank += 1

    # --- Part 1: C_1 (last column of C, index k=4) ---
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Schoolbook: 2*4 = 8 terms.
    for i in range(n):
        for j in range(m):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            u[i * m + j] = 1.0
            v[j * p + 4] = 1.0
            w[4 * n + i] = 1.0
            
            add_term(u, v, w)
            
    # --- Part 2: C_4 (first 4 columns of C, indices k=0..3) ---
    # C[:, 0:4] = A_L B_T[:, 0:4] + A_R B_B[:, 0:4]
    # A_L = A[:, 0:2], A_R = A[:, 2:4]
    # B_T = B[0:2, :], B_B = B[2:4, :]
    
    # We need to compute M_L = A_L B_T and M_R = A_R B_B.
    # Each is a 2x2 matrix multiplication by a 2x4 matrix.
    # Split B_T and B_B into two 2x2 blocks: cols 0-1 and 2-3.
    
    # --- Sub-part 2a: M_L ---
    # M_L[:, 0:2] = A_L B_T[:, 0:2] (2x2x2 Strassen)
    # M_L[:, 2:4] = A_L B_T[:, 2:4] (2x2x2 Strassen)
    
    # Strassen 2x2x2 decomposition
    # Inputs: A (2x2), B (2x2)
    # Output: C (2x2)
    # Terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def get_strassen_terms(a_offset, b_offset, w_offset):
        """
        Generates the 7 Strassen terms for a 2x2x2 multiplication.
        a_offset: index in U (flattened A) for A[0,0]
        b_offset: index in V (flattened B) for B[0,0]
        w_offset: index in W (flattened C) for C[0,0]
        """
        terms = []
        
        # M1
        u1 = np.zeros(n * m); v1 = np.zeros(m * p); w1 = np.zeros(p * n)
        u1[a_offset] = 1; u1[a_offset + 3] = 1 # A00, A11
        v1[b_offset] = 1; v1[b_offset + 3] = 1 # B00, B11
        w1[w_offset] = 1; w1[w_offset + 1] = 1; w1[w_offset + 2] = 1; w1[w_offset + 3] = 1 # C00, C01, C10, C11 (coeff 1)
        # Actually, we need to distribute the M's to the W's.
        # It is easier to define the 7 rank-1 tensors directly.
        
        # Let's define the 7 basis vectors for U, V, W corresponding to the Strassen products.
        # The output W vectors will be combinations of these.
        # But we are adding terms to the global U, V, W.
        # Each Strassen product M_k is a scalar.
        # C = sum_k M_k * W_k where W_k is a fixed 2x2 matrix pattern.
        # So for each Strassen product, we add 1 term to U, V, and W?
        # No, M_k is a product of a linear form in A and a linear form in B.
        # So M_k * W_k is a rank-1 tensor: u_k \otimes v_k \otimes w_k.
        # Yes.
        
        # M1: u=(1,0,0,1), v=(1,0,0,1), w=(1,1,1,1) -> C00+=1, C01+=1, C10+=1, C11+=1?
        # Wait, C00 = M1 + M4 - M5 + M7.
        # So the W vector for M1 should have 1 at C00, 0 at C01, 0 at C10, 0 at C11?
        # No, the tensor is T = sum u_k \otimes v_k \otimes w_k.
        # The coefficient of C_ij is sum_k u_k(A) v_k(B) w_k[i,j].
        # So w_k is the vector of coefficients for C_ij contributed by M_k.
        
        # M1 contributes to C00, C01, C10, C11?
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # w_M1 = [1, 0, 0, 1] (C00, C01, C10, C11)
        # w_M2 = [0, 0, 1, -1]
        # w_M3 = [0, 1, 0, 1]
        # w_M4 = [1, 0, 1, 0]
        # w_M5 = [-1, 1, 0, 0]
        # w_M6 = [0, 0, 0, 1]
        # w_M7 = [1, 0, 0, 0]
        
        # Let's verify C00: 1(M1) + 1(M4) - 1(M5) + 1(M7) = 1+1-1+1 = 2? No.
        # C00 = M1 + M4 - M5 + M7.
        # Coeff of M1 is 1. Coeff of M4 is 1. Coeff of M5 is -1. Coeff of M7 is 1.
        # So w_M1[0] = 1, w_M4[0] = 1, w_M5[0] = -1, w_M7[0] = 1.
        
        # Let's construct the 7 terms for a single 2x2x2 block.
        
        # u vectors (linear forms on A 2x2)
        # M1: A00+A11 -> [1, 0, 0, 1]
        # M2: A10+A11 -> [0, 0, 1, 1]
        # M3: A00 -> [1, 0, 0, 0]
        # M4: A11 -> [0, 0, 0, 1]
        # M5: A00+A01 -> [1, 1, 0, 0]
        # M6: A10-A00 -> [-1, 0, 1, 0]
        # M7: A01-A11 -> [0, 1, 0, -1]
        
        u_M1 = np.array([1, 0, 0, 1])
        u_M2 = np.array([0, 0, 1, 1])
        u_M3 = np.array([1, 0, 0, 0])
        u_M4 = np.array([0, 0, 0, 1])
        u_M5 = np.array([1, 1, 0, 0])
        u_M6 = np.array([-1, 0, 1, 0])
        u_M7 = np.array([0, 1, 0, -1])
        
        # v vectors (linear forms on B 2x2)
        # M1: B00+B11 -> [1, 0, 0, 1]
        # M2: B00 -> [1, 0, 0, 0]
        # M3: B01-B11 -> [0, 1, 0, -1]
        # M4: B10-B00 -> [-1, 0, 1, 0]
        # M5: B11 -> [0, 0, 0, 1]
        # M6: B00+B01 -> [1, 1, 0, 0]
        # M7: B10+B11 -> [0, 0, 1, 1]
        
        v_M1 = np.array([1, 0, 0, 1])
        v_M2 = np.array([1, 0, 0, 0])
        v_M3 = np.array([0, 1, 0, -1])
        v_M4 = np.array([-1, 0, 1, 0])
        v_M5 = np.array([0, 0, 0, 1])
        v_M6 = np.array([1, 1, 0, 0])
        v_M7 = np.array([0, 0, 1, 1])
        
        # w vectors (coefficients for C 2x2)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # w_M1: C00=1, C11=1 -> [1, 0, 0, 1]
        # w_M2: C10=1, C11=-1 -> [0, 0, 1, -1]
        # w_M3: C01=1, C11=1 -> [0, 1, 0, 1]
        # w_M4: C00=1, C10=1 -> [1, 0, 1, 0]
        # w_M5: C00=-1, C01=1 -> [-1, 1, 0, 0]
        # w_M6: C11=1 -> [0, 0, 0, 1]
        # w_M7: C00=1 -> [1, 0, 0, 0]
        
        w_M1 = np.array([1, 0, 0, 1])
        w_M2 = np.array([0, 0, 1, -1])
        w_M3 = np.array([0, 1, 0, 1])
        w_M4 = np.array([1, 0, 1, 0])
        w_M5 = np.array([-1, 1, 0, 0])
        w_M6 = np.array([0, 0, 0, 1])
        w_M7 = np.array([1, 0, 0, 0])
        
        # Now we map these local 2x2 indices to global indices.
        # a_offset is index of A[0,0] in global U.
        # b_offset is index of B[0,0] in global V.
        # w_offset is index of C[0,0] in global W.
        
        # We need to embed the local u, v, w into the global vectors.
        
        local_us = [u_M1, u_M2, u_M3, u_M4, u_M5, u_M6, u_M7]
        local_vs = [v_M1, v_M2, v_M3, v_M4, v_M5, v_M6, v_M7]
        local_ws = [w_M1, w_M2, w_M3, w_M4, w_M5, w_M6, w_M7]
        
        for i in range(7):
            gu = np.zeros(n * m)
            gv = np.zeros(m * p)
            gw = np.zeros(p * n)
            
            # Embed u
            # Local u indices 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
            # Global A indices: a_offset, a_offset+1, a_offset+2, a_offset+3
            gu[a_offset:a_offset+4] = local_us[i]
            
            # Embed v
            # Local v indices 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
            # Global B indices: b_offset, b_offset+1, b_offset+2, b_offset+3
            gv[b_offset:b_offset+4] = local_vs[i]
            
            # Embed w
            # Local w indices 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
            # Global C indices: w_offset, w_offset+1, w_offset+2, w_offset+3
            # Note: Global C is 2x4 or 2x2?
            # In our context, this 2x2x2 block produces a 2x2 sub-block of C.
            # The w_offset should point to the start of this 2x2 block in the flattened W.
            # Flattened W index for C[i, k] is k*n + i.
            # If the block is at columns k0, k0+1 and rows 0,1.
            # C[0, k0] -> index k0*n + 0
            # C[0, k0+1] -> index (k0+1)*n + 0
            # C[1, k0] -> index k0*n + 1
            # C[1, k0+1] -> index (k0+1)*n + 1
            # This is NOT contiguous in the flattened W if n=2.
            # Indices: k0*2+0, (k0+1)*2+0, k0*2+1, (k0+1)*2+1.
            # Order in local w: C00, C01, C10, C11.
            # Global indices:
            # C00 -> w_offset
            # C01 -> w_offset + 2 (since n=2, next col is +2)
            # C10 -> w_offset + 1
            # C11 -> w_offset + 3
            
            gw[w_offset] = local_ws[i][0]
            gw[w_offset + 2] = local_ws[i][1]
            gw[w_offset + 1] = local_ws[i][2]
            gw[w_offset + 3] = local_ws[i][3]
            
            add_term(gu, gv, gw)

    # --- Execute Strassen for the 4 blocks ---
    
    # Block 1: A_L (cols 0-1) * B_T (rows 0-1, cols 0-1)
    # A_L offset in U: 0
    # B_T offset in V: 0
    # Output C cols 0-1, rows 0-1.
    # C[0,0] index: 0*2+0 = 0.
    get_strassen_terms(0, 0, 0)
    
    # Block 2: A_L (cols 0-1) * B_T (rows 0-1, cols 2-3)
    # A_L offset in U: 0
    # B_T offset in V: 2 (B[0,2] is index 0*4+2 = 2)
    # Output C cols 2-3, rows 0-1.
    # C[0,2] index: 2*2+0 = 4.
    get_strassen_terms(0, 2, 4)
    
    # Block 3: A_R (cols 2-3) * B_B (rows 2-3, cols 0-1)
    # A_R offset in U: 2 (A[0,2] is index 0*4+2 = 2)
    # B_B offset in V: 8 (B[2,0] is index 2*4+0 = 8)
    # Output C cols 0-1, rows 0-1.
    # C[0,0] index: 0.
    get_strassen_terms(2, 8, 0)
    
    # Block 4: A_R (cols 2-3) * B_B (rows 2-3, cols 2-3)
    # A_R offset in U: 2
    # B_B offset in V: 10 (B[2,2] is index 2*4+2 = 10)
    # Output C cols 2-3, rows 0-1.
    # C[0,2] index: 4.
    get_strassen_terms(2, 10, 4)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
