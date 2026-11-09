import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Rank 22 algorithm for 3x3x3 (Strassen-based)
    def get_3x3x3_rank22():
        # Returns U, V, W for 3x3x3 multiplication
        # U: 9x22, V: 9x22, W: 9x22
        # Indices: i, j in 0..2 for U; j, k in 0..2 for V; k, i in 0..2 for W
        
        # Strassen's 7 terms for the 2x2 block (indices 0,1)
        # Plus 15 schoolbook terms for the rest?
        # Actually, let's just use schoolbook for 3x3x3 to be safe? No, rank 27.
        # Let's use the Strassen decomposition for the 2x2 part and schoolbook for the 3rd row/col interactions.
        
        # Terms 0-6: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # A11=A[0:2,0:2], A12=A[0:2,2], A21=A[2,0:2], A22=A[2,2]
        # B11=B[0:2,0:2], B12=B[0:2,2], B21=B[2,0:2], B22=B[2,2]
        
        # We need to map these to the global indices.
        # Global U[i,j], V[j,k], W[k,i]
        
        U = np.zeros((9, 22))
        V = np.zeros((9, 22))
        W = np.zeros((9, 22))
        
        # Helper to add a term
        def add_term(t, u_mask, v_mask, w_mask):
            # u_mask: dict {(i,j): val}
            for (i,j), val in u_mask.items():
                U[i*3+j, t] += val
            for (j,k), val in v_mask.items():
                V[j*3+k, t] += val
            for (k,i), val in w_mask.items():
                W[k*3+i, t] += val

        # Strassen M1: (A11+A22)(B11+B22)
        # A11: i,j in 0,1. A22: i=2,j=2.
        # B11: j,k in 0,1. B22: j=2,k=2.
        # Result C11: i,k in 0,1. C22: i=2,k=2.
        # Contribution to C11: M1. To C22: M1.
        # Wait, Strassen formula:
        # C11 = M1 - M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # Let's define the masks for each M
        # M1 = (A11 + A22) * (B11 + B22)
        # U1: A11 + A22. V1: B11 + B22.
        # W1: contributes to C11 and C22.
        # W1[i,k] for i,k in 0,1 is +1. W1[2,2] is +1.
        
        def mask_A11_plus_A22():
            m = {}
            for i in range(2):
                for j in range(2):
                    m[(i,j)] = 1
            m[(2,2)] = 1
            return m

        def mask_B11_plus_B22():
            m = {}
            for j in range(2):
                for k in range(2):
                    m[(j,k)] = 1
            m[(2,2)] = 1
            return m

        def mask_C11_plus_C22():
            m = {}
            for i in range(2):
                for k in range(2):
                    m[(k,i)] = 1
            m[(2,2)] = 1
            return m

        # M1
        add_term(0, mask_A11_plus_A22(), mask_B11_plus_B22(), mask_C11_plus_C22())
        
        # M2 = (A11 + A21) * B11
        # A11: i,j in 0,1. A21: i=2, j in 0,1.
        # B11: j,k in 0,1.
        # Result C21: i=2, k in 0,1.
        def mask_A11_plus_A21():
            m = {}
            for i in range(2):
                for j in range(2):
                    m[(i,j)] = 1
            for j in range(2):
                m[(2,j)] = 1
            return m

        def mask_B11():
            m = {}
            for j in range(2):
                for k in range(2):
                    m[(j,k)] = 1
            return m

        def mask_C21():
            m = {}
            for k in range(2):
                m[(k,2)] = 1
            return m

        add_term(1, mask_A11_plus_A21(), mask_B11(), mask_C21())

        # M3 = A11 * (B12 - B22)
        # A11: i,j in 0,1.
        # B12: j in 0,1, k=2. B22: j=2, k=2.
        # Result C12: i in 0,1, k=2.
        def mask_A11():
            m = {}
            for i in range(2):
                for j in range(2):
                    m[(i,j)] = 1
            return m

        def mask_B12_minus_B22():
            m = {}
            for j in range(2):
                m[(j,2)] = 1
            m[(2,2)] = -1
            return m

        def mask_C12():
            m = {}
            for i in range(2):
                m[(2,i)] = 1
            return m

        add_term(2, mask_A11(), mask_B12_minus_B22(), mask_C12())

        # M4 = A22 * (B21 - B11)
        # A22: i=2, j=2.
        # B21: j=2, k in 0,1. B11: j in 0,1, k in 0,1.
        # Result C21: i=2, k in 0,1.
        def mask_A22():
            return {(2,2): 1}

        def mask_B21_minus_B11():
            m = {}
            for k in range(2):
                m[(2,k)] = 1
            for j in range(2):
                for k in range(2):
                    m[(j,k)] = -1
            return m

        add_term(3, mask_A22(), mask_B21_minus_B11(), mask_C21())

        # M5 = (A12 + A22) * B22
        # A12: i in 0,1, j=2. A22: i=2, j=2.
        # B22: j=2, k=2.
        # Result C12: i in 0,1, k=2. C22: i=2, k=2.
        def mask_A12_plus_A22():
            m = {}
            for i in range(2):
                m[(i,2)] = 1
            m[(2,2)] = 1
            return m

        def mask_B22():
            return {(2,2): 1}

        def mask_C12_plus_C22():
            m = {}
            for i in range(2):
                m[(2,i)] = 1
            m[(2,2)] = 1
            return m

        add_term(4, mask_A12_plus_A22(), mask_B22(), mask_C12_plus_C22())

        # M6 = (A11 - A21) * (B11 + B21)
        # A11: i,j in 0,1. A21: i=2, j in 0,1.
        # B11: j,k in 0,1. B21: j=2, k in 0,1.
        # Result C22: i=2, k=2.
        def mask_A11_minus_A21():
            m = {}
            for i in range(2):
                for j in range(2):
                    m[(i,j)] = 1
            for j in range(2):
                m[(2,j)] = -1
            return m

        def mask_B11_plus_B21():
            m = {}
            for j in range(2):
                for k in range(2):
                    m[(j,k)] = 1
            for k in range(2):
                m[(2,k)] = 1
            return m

        def mask_C22():
            return {(2,2): 1}

        add_term(5, mask_A11_minus_A21(), mask_B11_plus_B21(), mask_C22())

        # M7 = (A12 - A22) * (B21 + B22)
        # A12: i in 0,1, j=2. A22: i=2, j=2.
        # B21: j=2, k in 0,1. B22: j=2, k=2.
        # Result C11: i in 0,1, k in 0,1.
        def mask_A12_minus_A22():
            m = {}
            for i in range(2):
                m[(i,2)] = 1
            m[(2,2)] = -1
            return m

        def mask_B21_plus_B22():
            m = {}
            for k in range(2):
                m[(2,k)] = 1
            m[(2,2)] = 1
            return m

        def mask_C11():
            m = {}
            for i in range(2):
                for k in range(2):
                    m[(k,i)] = 1
            return m

        add_term(6, mask_A12_minus_A22(), mask_B21_plus_B22(), mask_C11())

        # Now we have computed C11, C12, C21, C22 using 7 terms.
        # We still need to compute the terms involving the 3rd row/col that were NOT covered?
        # Strassen covers ALL of C11, C12, C21, C22.
        # But wait, the inputs A12, A21, B12, B21 were included in the Strassen terms.
        # So the 2x2 block of the result is fully computed.
        # What is missing?
        # The result C is 3x3.
        # C11, C12, C21, C22 are computed.
        # We need C13, C31, C33.
        
        # C13 = A11 B13 + A12 B23
        # C31 = A31 B11 + A32 B21
        # C33 = A31 B13 + A32 B23 + A33 B33
        
        # Note: A12, A21, B12, B21 are parts of the 3x3 matrices.
        # A12 is A[0:2, 2]. A21 is A[2, 0:2].
        # B12 is B[0:2, 2]. B21 is B[2, 0:2].
        
        # Let's add schoolbook terms for the remaining parts.
        # C13: i in 0,1, k=2.
        # Term 7: A[0,0] B[0,2] -> W[2,0]
        # Term 8: A[0,1] B[1,2] -> W[2,0]
        # Term 9: A[1,0] B[0,2] -> W[2,1]
        # Term 10: A[1,1] B[1,2] -> W[2,1]
        # Term 11: A[0,2] B[2,2] -> W[2,0]
        # Term 12: A[1,2] B[2,2] -> W[2,1]
        
        # C31: i=2, k in 0,1.
        # Term 13: A[2,0] B[0,0] -> W[0,2]
        # Term 14: A[2,0] B[0,1] -> W[1,2]
        # Term 15: A[2,1] B[1,0] -> W[0,2]
        # Term 16: A[2,1] B[1,1] -> W[1,2]
        # Term 17: A[2,2] B[2,0] -> W[0,2]
        # Term 18: A[2,2] B[2,1] -> W[1,2]
        
        # C33: i=2, k=2.
        # Term 19: A[2,0] B[0,2] -> W[2,2]
        # Term 20: A[2,1] B[1,2] -> W[2,2]
        # Term 21: A[2,2] B[2,2] -> W[2,2]
        
        # Wait, A[2,2] B[2,2] was already in M1?
        # M1 included A22 B22.
        # M1 contributed +1 to W[2,2].
        # So Term 21 is redundant?
        # Let's check M1 contribution to C22.
        # M1 = (A11+A22)(B11+B22).
        # The A22 B22 part is A[2,2] B[2,2].
        # This contributes to C[2,2].
        # So yes, A[2,2] B[2,2] is covered by M1.
        
        # So we only need 20 terms?
        # Let's add the 19 terms above (excluding the redundant one).
        
        # C13 terms
        add_term(7, {(0,0):1}, {(0,2):1}, {(2,0):1})
        add_term(8, {(0,1):1}, {(1,2):1}, {(2,0):1})
        add_term(9, {(1,0):1}, {(0,2):1}, {(2,1):1})
        add_term(10, {(1,1):1}, {(1,2):1}, {(2,1):1})
        add_term(11, {(0,2):1}, {(2,2):1}, {(2,0):1})
        add_term(12, {(1,2):1}, {(2,2):1}, {(2,1):1})
        
        # C31 terms
        add_term(13, {(2,0):1}, {(0,0):1}, {(0,2):1})
        add_term(14, {(2,0):1}, {(0,1):1}, {(1,2):1})
        add_term(15, {(2,1):1}, {(1,0):1}, {(0,2):1})
        add_term(16, {(2,1):1}, {(1,1):1}, {(1,2):1})
        add_term(17, {(2,2):1}, {(2,0):1}, {(0,2):1})
        add_term(18, {(2,2):1}, {(2,1):1}, {(1,2):1})
        
        # C33 terms (excluding A22 B22)
        add_term(19, {(2,0):1}, {(0,2):1}, {(2,2):1})
        add_term(20, {(2,1):1}, {(1,2):1}, {(2,2):1})
        # Term 21 (A22 B22) is already in M1.
        
        # Total rank 21?
        # We used indices 0 to 20. That's 21 terms.
        # But we need to verify if M1's contribution to C22 is exactly A22 B22 + ...
        # Yes.
        
        return U, V, W

    # 2. Construct Rank 18 algorithm for 3x3x2 (Schoolbook)
    def get_3x3x2_rank18():
        # U: 9x18, V: 6x18, W: 6x18
        # i in 0..2, j in 0..2, k in 0..1
        U = np.zeros((9, 18))
        V = np.zeros((6, 18))
        W = np.zeros((6, 18))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*3+j, col] = 1
                    V[j*2+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return U, V, W

    # 3. Construct Rank 15 algorithm for 3x1x5 (Schoolbook)
    def get_3x1x5_rank15():
        # U: 3x15, V: 5x15, W: 15x15
        # i in 0..2, j=0, k in 0..4
        U = np.zeros((3, 15))
        V = np.zeros((5, 15))
        W = np.zeros((15, 15))
        col = 0
        for i in range(3):
            for k in range(5):
                U[i, col] = 1
                V[k, col] = 1
                W[k*3+i, col] = 1
                col += 1
        return U, V, W

    # Combine
    U1, V1, W1 = get_3x3x3_rank22() # Actually 21 terms
    U2, V2, W2 = get_3x3x2_rank18()
    U3, V3, W3 = get_3x1x5_rank15()
    
    # Dimensions for 3x4x5
    # U: 3x4 (12) x Rank
    # V: 4x5 (20) x Rank
    # W: 5x3 (15) x Rank
    
    Rank = 21 + 18 + 15
    U = np.zeros((12, Rank))
    V = np.zeros((20, Rank))
    W = np.zeros((15, Rank))
    
    # Map Part 1: 3x3x3
    # A1 is A[0:3, 0:3]. B1a is B[0:3, 0:3].
    # U1 indices: i in 0..2, j in 0..2. Global U: i in 0..2, j in 0..2.
    # V1 indices: j in 0..2, k in 0..2. Global V: j in 0..2, k in 0..2.
    # W1 indices: k in 0..2, i in 0..2. Global W: k in 0..2, i in 0..2.
    
    for t in range(21):
        U[0:9, t] = U1[:, t]
        V[0:9, t] = V1[:, t]
        W[0:9, t] = W1[:, t]
        
    # Map Part 2: 3x3x2
    # A1 is A[0:3, 0:3]. B1b is B[0:3, 3:5].
    # U2 indices: i in 0..2, j in 0..2. Global U: i in 0..2, j in 0..2.
    # V2 indices: j in 0..2, k in 0..1. Global V: j in 0..2, k in 3..4.
    # W2 indices: k in 0..1, i in 0..2. Global W: k in 3..4, i in 0..2.
    
    offset = 21
    for t in range(18):
        U[0:9, offset+t] = U2[:, t]
        # V2[j*2+k] -> Global V[j*5 + (3+k)]
        for j in range(3):
            for k in range(2):
                V[j*5 + 3 + k, offset+t] = V2[j*2+k, t]
        # W2[k*3+i] -> Global W[(3+k)*3 + i]
        for k in range(2):
            for i in range(3):
                W[(3+k)*3 + i, offset+t] = W2[k*3+i, t]
                
    # Map Part 3: 3x1x5
    # A2 is A[0:3, 3]. B2 is B[3, 0:5].
    # U3 indices: i in 0..2, j=0. Global U: i in 0..2, j=3.
    # V3 indices: j=0, k in 0..4. Global V: j=3, k in 0..4.
    # W3 indices: k in 0..4, i in 0..2. Global W: k in 0..4, i in 0..2.
    
    offset = 21 + 18
    for t in range(15):
        U[i*3+3, offset+t] = U3[i, t] if False else U3[i, t] # U3 is 3x15
        # U3[i, t] -> Global U[i*3+3, offset+t]
        for i in range(3):
            U[i*3+3, offset+t] = U3[i, t]
        # V3[k, t] -> Global V[3*5 + k]
        for k in range(5):
            V[3*5 + k, offset+t] = V3[k, t]
        # W3[k*3+i, t] -> Global W[k*3 + i]
        for k in range(5):
            for i in range(3):
                W[k*3 + i, offset+t] = W3[k*3+i, t]

    # Verify
    U_3d = U.reshape(3, 4, -1)
    V_3d = V.reshape(4, 5, -1)
    W_3d = W.reshape(5, 3, -1)
    
    T_calc = np.zeros((5, 3, 3, 4, 4, 5), dtype=np.float64)
    for t in range(Rank):
        u = U_3d[:, :, t]
        v = V_3d[:, :, t]
        w = W_3d[:, :, t]
        T_calc += w[:, :, None, None, None, None] * \
                  u[None, None, :, :, None, None] * \
                  v[None, None, None, None, :, :]
                  
    T_target = np.zeros((5, 3, 3, 4, 4, 5), dtype=np.int8)
    for i in range(3):
        for j in range(4):
            for k in range(5):
                T_target[k, i, i, j, j, k] = 1
                
    if not np.allclose(T_calc, T_target, atol=1e-6):
        # Fallback to schoolbook
        def schoolbook():
            rank = 60
            U = np.zeros((12, rank))
            V = np.zeros((20, rank))
            W = np.zeros((15, rank))
            col = 0
            for i in range(3):
                for j in range(4):
                    for k in range(5):
                        U[i*4+j, col] = 1
                        V[j*5+k, col] = 1
                        W[k*3+i, col] = 1
                        col += 1
            return U, V, W
        U, V, W = schoolbook()
        Rank = 60

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
