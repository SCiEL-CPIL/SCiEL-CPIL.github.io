import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # Rank 56 for 4x4x4 Strassen
    # Rank 16 for 4x4x1 Schoolbook
    # Two 4x4x4 blocks and one 4x4x1 block
    R_total = 56 + 56 + 16
    
    U = np.zeros((dim_u, R_total))
    V = np.zeros((dim_v, R_total))
    W = np.zeros((dim_w, R_total))
    
    # Helper to add Strassen 4x4x4 factors
    # A is 4x4, B is 4x4, C is 4x4
    # We map the 16 entries of A, B, C to the Strassen terms
    
    # Strassen terms for 2x2 blocks
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x2)
    
    # P and Q for each term M_k
    # P_k and Q_k are 2x2 matrices
    # Their entries are linear combinations of the 16 entries of A and B
    
    # Let's define the linear combinations
    # A indices: 0-15 (row major)
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # B indices: 0-15
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # C indices: 0-15
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    # M_k is a 2x2 matrix.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to define P_k and Q_k for k=1..7
    # P_k, Q_k are 2x2 matrices.
    # P_k entries are linear combos of A.
    # Q_k entries are linear combos of B.
    
    # M1 = A11 * (B11 + B22)
    # P1 = A11, Q1 = B11 + B22
    # M2 = (A11 + A22) * B22
    # P2 = A11 + A22, Q2 = B22
    # M3 = (A21 - A11) * B11
    # P3 = A21 - A11, Q3 = B11
    # M4 = A22 * (B12 - B22)
    # P4 = A22, Q4 = B12 - B22
    # M5 = (A11 + A12) * B21
    # P5 = A11 + A12, Q5 = B21
    # M6 = (A21 + A22) * (B11 - B21)
    # P6 = A21 + A22, Q6 = B11 - B21
    # M7 = (A11 - A21) * (B12 + B22)
    # P7 = A11 - A21, Q7 = B12 + B22
    
    # Function to add a 2x2x2 multiplication term
    # P_coeffs: 4x16 matrix, P[i] = sum_a P_coeffs[i,a] A[a]
    # Q_coeffs: 4x16 matrix, Q[i] = sum_b Q_coeffs[i,b] B[b]
    # C_coeffs: 4x16 matrix, C[i] = sum_k C_coeffs[i,k] M_k? 
    # Actually, we add to the global U, V, W.
    
    def add_strassen_block(offset, P_coeffs, Q_coeffs, C_coeffs):
        # P_coeffs, Q_coeffs, C_coeffs are 4x16
        # We iterate over the 8 rank-1 terms of 2x2x2
        # Term (i, l, j) in 2x2x2: P[i,l] * Q[l,j] -> M[i,j]
        # Flattened P index: i*2 + l
        # Flattened Q index: l*2 + j
        # Flattened M index: i*2 + j
        
        for i in range(2):
            for l in range(2):
                for j in range(2):
                    p_idx = i * 2 + l
                    q_idx = l * 2 + j
                    m_idx = i * 2 + j
                    
                    # U contribution
                    # U[a, r] += P_coeffs[p_idx, a]
                    U[:, offset] += P_coeffs[p_idx, :]
                    
                    # V contribution
                    # V[b, r] += Q_coeffs[q_idx, b]
                    V[:, offset] += Q_coeffs[q_idx, :]
                    
                    # W contribution
                    # W[c, r] += C_coeffs[m_idx, c]
                    W[:, offset] += C_coeffs[m_idx, :]
                    
                    offset += 1
        return offset

    # Define the 7 terms
    # P_coeffs and Q_coeffs are 4x16
    # Initialize to 0
    P_coeffs_list = []
    Q_coeffs_list = []
    C_coeffs_list = []
    
    # A blocks indices
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    def make_coeffs(blocks, signs):
        # blocks: list of 4 indices
        # signs: list of 4 signs
        coeffs = np.zeros((4, 16))
        for i in range(4):
            coeffs[i, blocks[i]] = signs[i]
        return coeffs

    # M1: P=A11, Q=B11+B22
    P1 = np.zeros((4, 16))
    for i in range(4): P1[i, A11[i]] = 1
    Q1 = np.zeros((4, 16))
    for i in range(4): 
        Q1[i, B11[i]] = 1
        Q1[i, B22[i]] = 1
    C1 = np.zeros((4, 16))
    for i in range(4):
        C1[i, C11[i]] = 1
        C1[i, C22[i]] = 1 # C22 = M1 - M2 + M3 + M6
    
    # M2: P=A11+A22, Q=B22
    P2 = np.zeros((4, 16))
    for i in range(4):
        P2[i, A11[i]] = 1
        P2[i, A22[i]] = 1
    Q2 = np.zeros((4, 16))
    for i in range(4): Q2[i, B22[i]] = 1
    C2 = np.zeros((4, 16))
    for i in range(4):
        C2[i, C21[i]] = 1
        C2[i, C22[i]] = -1
        
    # M3: P=A21-A11, Q=B11
    P3 = np.zeros((4, 16))
    for i in range(4):
        P3[i, A21[i]] = 1
        P3[i, A11[i]] = -1
    Q3 = np.zeros((4, 16))
    for i in range(4): Q3[i, B11[i]] = 1
    C3 = np.zeros((4, 16))
    for i in range(4):
        C3[i, C12[i]] = 1
        C3[i, C22[i]] = 1
        
    # M4: P=A22, Q=B12-B22
    P4 = np.zeros((4, 16))
    for i in range(4): P4[i, A22[i]] = 1
    Q4 = np.zeros((4, 16))
    for i in range(4):
        Q4[i, B12[i]] = 1
        Q4[i, B22[i]] = -1
    C4 = np.zeros((4, 16))
    for i in range(4):
        C4[i, C11[i]] = 1
        C4[i, C21[i]] = 1
        
    # M5: P=A11+A12, Q=B21
    P5 = np.zeros((4, 16))
    for i in range(4):
        P5[i, A11[i]] = 1
        P5[i, A12[i]] = 1
    Q5 = np.zeros((4, 16))
    for i in range(4): Q5[i, B21[i]] = 1
    C5 = np.zeros((4, 16))
    for i in range(4):
        C5[i, C11[i]] = -1
        C5[i, C12[i]] = 1
        
    # M6: P=A21+A22, Q=B11-B21
    P6 = np.zeros((4, 16))
    for i in range(4):
        P6[i, A21[i]] = 1
        P6[i, A22[i]] = 1
    Q6 = np.zeros((4, 16))
    for i in range(4):
        Q6[i, B11[i]] = 1
        Q6[i, B21[i]] = -1
    C6 = np.zeros((4, 16))
    for i in range(4):
        C6[i, C22[i]] = 1
        
    # M7: P=A11-A21, Q=B12+B22
    P7 = np.zeros((4, 16))
    for i in range(4):
        P7[i, A11[i]] = 1
        P7[i, A21[i]] = -1
    Q7 = np.zeros((4, 16))
    for i in range(4):
        Q7[i, B12[i]] = 1
        Q7[i, B22[i]] = 1
    C7 = np.zeros((4, 16))
    for i in range(4):
        C7[i, C11[i]] = 1
        
    terms = [
        (P1, Q1, C1),
        (P2, Q2, C2),
        (P3, Q3, C3),
        (P4, Q4, C4),
        (P5, Q5, C5),
        (P6, Q6, C6),
        (P7, Q7, C7)
    ]
    
    # Block 1: B cols 0-3, C cols 0-3
    offset = 0
    for P, Q, C in terms:
        offset = add_strassen_block(offset, P, Q, C)
        
    # Block 2: B cols 4-7, C cols 4-7
    # We need to shift the V and W indices
    # V index for B col k is j*9 + k
    # In the helper, V index is b (0-15).
    # We need to map b to the correct column in the main V.
    
    # The helper function add_strassen_block adds to V[:, offset] using indices 0-15.
    # This assumes B is 4x4.
    # For Block 2, B is cols 4-7.
    # So V index j*9 + (4+k_prime) where k_prime is 0-3.
    # The helper uses V index b.
    # If b = j*4 + k_prime, we need to map to j*9 + 4 + k_prime.
    
    # Let's modify the approach:
    # Instead of using the helper directly, we will pass a mapping function or just do it manually for Block 2.
    # Or, we can create a dummy V and W for the 4x4x4 and then copy.
    
    # Let's create U2, V2, W2 for the second block
    U2 = np.zeros((16, 56))
    V2 = np.zeros((16, 56))
    W2 = np.zeros((16, 56))
    
    offset2 = 0
    for P, Q, C in terms:
        offset2 = add_strassen_block_to(offset2, U2, V2, W2, P, Q, C)
        
    # Copy U2 to U
    U[:, 56:112] = U2
    
    # Copy V2 to V with index mapping
    for j in range(4):
        for k_prime in range(4):
            v_idx_444 = j * 4 + k_prime
            v_idx_main = j * 9 + (4 + k_prime)
            V[v_idx_main, 56:112] = V2[v_idx_444, :]
            
    # Copy W2 to W with index mapping
    for k_prime in range(4):
        for i in range(4):
            w_idx_444 = k_prime * 4 + i
            w_idx_main = (4 + k_prime) * 4 + i
            W[w_idx_main, 56:112] = W2[w_idx_444, :]
            
    # Block 3: B col 8, C col 8
    # Rank 16
    for i in range(4):
        for j in range(4):
            r = 112 + i * 4 + j
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def add_strassen_block_to(offset, U, V, W, P_coeffs, Q_coeffs, C_coeffs):
    for i in range(2):
        for l in range(2):
            for j in range(2):
                p_idx = i * 2 + l
                q_idx = l * 2 + j
                m_idx = i * 2 + j
                
                U[:, offset] += P_coeffs[p_idx, :]
                V[:, offset] += Q_coeffs[q_idx, :]
                W[:, offset] += C_coeffs[m_idx, :]
                
                offset += 1
    return offset

# Note: The first call to add_strassen_block in the main function was using the global U, V, W.
# I defined add_strassen_block but didn't define add_strassen_block_to in the first draft.
# I will use add_strassen_block_to for both.

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    R_total = 56 + 56 + 16
    
    U = np.zeros((dim_u, R_total))
    V = np.zeros((dim_v, R_total))
    W = np.zeros((dim_w, R_total))
    
    # Define terms
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    def make_P(blocks, signs):
        P = np.zeros((4, 16))
        for i in range(4): P[i, blocks[i]] = signs[i]
        return P

    def make_Q(blocks, signs):
        Q = np.zeros((4, 16))
        for i in range(4): Q[i, blocks[i]] = signs[i]
        return Q

    def make_C(blocks, signs):
        C = np.zeros((4, 16))
        for i in range(4): C[i, blocks[i]] = signs[i]
        return C

    # M1
    P1 = make_P(A11, [1,1,1,1])
    Q1 = np.zeros((4, 16))
    for i in range(4):
        Q1[i, B11[i]] = 1
        Q1[i, B22[i]] = 1
    C1 = np.zeros((4, 16))
    for i in range(4):
        C1[i, C11[i]] = 1
        C1[i, C22[i]] = 1
        
    # M2
    P2 = np.zeros((4, 16))
    for i in range(4):
        P2[i, A11[i]] = 1
        P2[i, A22[i]] = 1
    Q2 = make_Q(B22, [1,1,1,1])
    C2 = np.zeros((4, 16))
    for i in range(4):
        C2[i, C21[i]] = 1
        C2[i, C22[i]] = -1
        
    # M3
    P3 = np.zeros((4, 16))
    for i in range(4):
        P3[i, A21[i]] = 1
        P3[i, A11[i]] = -1
    Q3 = make_Q(B11, [1,1,1,1])
    C3 = np.zeros((4, 16))
    for i in range(4):
        C3[i, C12[i]] = 1
        C3[i, C22[i]] = 1
        
    # M4
    P4 = make_P(A22, [1,1,1,1])
    Q4 = np.zeros((4, 16))
    for i in range(4):
        Q4[i, B12[i]] = 1
        Q4[i, B22[i]] = -1
    C4 = np.zeros((4, 16))
    for i in range(4):
        C4[i, C11[i]] = 1
        C4[i, C21[i]] = 1
        
    # M5
    P5 = np.zeros((4, 16))
    for i in range(4):
        P5[i, A11[i]] = 1
        P5[i, A12[i]] = 1
    Q5 = make_Q(B21, [1,1,1,1])
    C5 = np.zeros((4, 16))
    for i in range(4):
        C5[i, C11[i]] = -1
        C5[i, C12[i]] = 1
        
    # M6
    P6 = np.zeros((4, 16))
    for i in range(4):
        P6[i, A21[i]] = 1
        P6[i, A22[i]] = 1
    Q6 = np.zeros((4, 16))
    for i in range(4):
        Q6[i, B11[i]] = 1
        Q6[i, B21[i]] = -1
    C6 = np.zeros((4, 16))
    for i in range(4):
        C6[i, C22[i]] = 1
        
    # M7
    P7 = np.zeros((4, 16))
    for i in range(4):
        P7[i, A11[i]] = 1
        P7[i, A21[i]] = -1
    Q7 = np.zeros((4, 16))
    for i in range(4):
        Q7[i, B12[i]] = 1
        Q7[i, B22[i]] = 1
    C7 = np.zeros((4, 16))
    for i in range(4):
        C7[i, C11[i]] = 1
        
    terms = [
        (P1, Q1, C1),
        (P2, Q2, C2),
        (P3, Q3, C3),
        (P4, Q4, C4),
        (P5, Q5, C5),
        (P6, Q6, C6),
        (P7, Q7, C7)
    ]
    
    # Block 1
    U1 = np.zeros((16, 56))
    V1 = np.zeros((16, 56))
    W1 = np.zeros((16, 56))
    offset = 0
    for P, Q, C in terms:
        offset = add_strassen_block_to(offset, U1, V1, W1, P, Q, C)
        
    U[:, 0:56] = U1
    for j in range(4):
        for k_prime in range(4):
            v_idx_444 = j * 4 + k_prime
            v_idx_main = j * 9 + k_prime
            V[v_idx_main, 0:56] = V1[v_idx_444, :]
    for k_prime in range(4):
        for i in range(4):
            w_idx_444 = k_prime * 4 + i
            w_idx_main = k_prime * 4 + i
            W[w_idx_main, 0:56] = W1[w_idx_444, :]
            
    # Block 2
    U2 = np.zeros((16, 56))
    V2 = np.zeros((16, 56))
    W2 = np.zeros((16, 56))
    offset = 0
    for P, Q, C in terms:
        offset = add_strassen_block_to(offset, U2, V2, W2, P, Q, C)
        
    U[:, 56:112] = U2
    for j in range(4):
        for k_prime in range(4):
            v_idx_444 = j * 4 + k_prime
            v_idx_main = j * 9 + (4 + k_prime)
            V[v_idx_main, 56:112] = V2[v_idx_444, :]
    for k_prime in range(4):
        for i in range(4):
            w_idx_444 = k_prime * 4 + i
            w_idx_main = (4 + k_prime) * 4 + i
            W[w_idx_main, 56:112] = W2[w_idx_444, :]
            
    # Block 3
    for i in range(4):
        for j in range(4):
            r = 112 + i * 4 + j
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def add_strassen_block_to(offset, U, V, W, P_coeffs, Q_coeffs, C_coeffs):
    for i in range(2):
        for l in range(2):
            for j in range(2):
                p_idx = i * 2 + l
                q_idx = l * 2 + j
                m_idx = i * 2 + j
                
                U[:, offset] += P_coeffs[p_idx, :]
                V[:, offset] += Q_coeffs[q_idx, :]
                W[:, offset] += C_coeffs[m_idx, :]
                
                offset += 1
    return offset
