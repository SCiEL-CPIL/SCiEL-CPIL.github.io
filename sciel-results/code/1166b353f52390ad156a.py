import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Global factor matrices
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # We build them as lists of columns (vectors) and then stack
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_local, V_local, W_local, i_offset, j_offset, k_offset, n_local, m_local, p_local):
        """
        Add a local decomposition to the global one.
        U_local: (n_local * m_local, R_local)
        V_local: (m_local * p_local, R_local)
        W_local: (p_local * n_local, R_local)
        i_offset, j_offset, k_offset: offsets in global indices
        """
        R_local = U_local.shape[1]
        for r in range(R_local):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            # Map U indices
            # Local u index: i_l * m_local + j_l
            # Global u index: (i_offset + i_l) * m + (j_offset + j_l)
            for i_l in range(n_local):
                for j_l in range(m_local):
                    local_idx = i_l * m_local + j_l
                    global_idx = (i_offset + i_l) * m + (j_offset + j_l)
                    u_vec[global_idx] = U_local[local_idx, r]
                    
            # Map V indices
            # Local v index: j_l * p_local + k_l
            # Global v index: (j_offset + j_l) * p + (k_offset + k_l)
            for j_l in range(m_local):
                for k_l in range(p_local):
                    local_idx = j_l * p_local + k_l
                    global_idx = (j_offset + j_l) * p + (k_offset + k_l)
                    v_vec[global_idx] = V_local[local_idx, r]
                    
            # Map W indices
            # Local w index: k_l * n_local + i_l
            # Global w index: (k_offset + k_l) * n + (i_offset + i_l)
            for k_l in range(p_local):
                for i_l in range(n_local):
                    local_idx = k_l * n_local + i_l
                    global_idx = (k_offset + k_l) * n + (i_offset + i_l)
                    w_vec[global_idx] = W_local[local_idx, r]
                    
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

    def get_strassen_factors():
        """
        Returns U, V, W for 2x2x2 Strassen.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        # A11: (0,0), A22: (1,1)
        # B11: (0,0), B22: (1,1)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1 # C11 += M1
        W[3,0] = 1 # C22 += M1
        
        # M2 = (A21+A22)B11
        # A21: (1,0), A22: (1,1)
        # B11: (0,0)
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1 # C21 += M2
        W[3,1] = -1 # C22 -= M2
        
        # M3 = A11(B12-B22)
        # A11: (0,0)
        # B12: (0,1), B22: (1,1)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1 # C12 += M3
        W[3,2] = 1 # C22 += M3
        
        # M4 = A22(B21-B11)
        # A22: (1,1)
        # B21: (1,0), B11: (0,0)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1 # C11 += M4
        W[2,3] = 1 # C21 += M4
        
        # M5 = (A11+A12)B22
        # A11: (0,0), A12: (0,1)
        # B22: (1,1)
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1 # C11 -= M5
        W[1,4] = 1 # C12 += M5
        
        # M6 = (A21-A11)(B11+B12)
        # A21: (1,0), A11: (0,0)
        # B11: (0,0), B12: (0,1)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1 # C22 += M6
        
        # M7 = (A12-A22)(B21+B22)
        # A12: (0,1), A22: (1,1)
        # B21: (1,0), B22: (1,1)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1 # C11 += M7
        
        return U, V, W

    def get_schoolbook_factors(n_l, m_l, p_l):
        """
        Returns U, V, W for n_l x m_l x p_l Schoolbook.
        """
        R = n_l * m_l * p_l
        U = np.zeros((n_l * m_l, R))
        V = np.zeros((m_l * p_l, R))
        W = np.zeros((p_l * n_l, R))
        
        col = 0
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    U[i * m_l + j, col] = 1
                    V[j * p_l + k, col] = 1
                    W[k * n_l + i, col] = 1
                    col += 1
        return U, V, W

    # Partition:
    # n=3 -> 1, 2
    # m=5 -> 3, 2
    # p=5 -> 3, 2
    
    n_sizes = [1, 2]
    m_sizes = [3, 2]
    p_sizes = [3, 2]
    
    # Offsets
    n_offsets = [0, 1]
    m_offsets = [0, 3]
    p_offsets = [0, 3]
    
    # Block 1,1: n=1, m=3, p=3
    # C11 (1x3) = A11 (1x3) B11 (3x3) + A12 (1x2) B21 (2x3)
    # Term 1: A11 B11 (1x3x3)
    U, V, W = get_schoolbook_factors(1, 3, 3)
    add_block(U, V, W, n_offsets[0], m_offsets[0], p_offsets[0], 1, 3, 3)
    
    # Term 2: A12 B21 (1x2x3)
    U, V, W = get_schoolbook_factors(1, 2, 3)
    add_block(U, V, W, n_offsets[0], m_offsets[1], p_offsets[0], 1, 2, 3)
    
    # Block 1,2: n=1, m=3, p=2
    # C12 (1x2) = A11 (1x3) B12 (3x2) + A12 (1x2) B22 (2x2)
    # Term 1: A11 B12 (1x3x2)
    U, V, W = get_schoolbook_factors(1, 3, 2)
    add_block(U, V, W, n_offsets[0], m_offsets[0], p_offsets[1], 1, 3, 2)
    
    # Term 2: A12 B22 (1x2x2)
    U, V, W = get_schoolbook_factors(1, 2, 2)
    add_block(U, V, W, n_offsets[0], m_offsets[1], p_offsets[1], 1, 2, 2)
    
    # Block 2,1: n=2, m=3, p=3
    # C21 (2x3) = A21 (2x3) B11 (3x3) + A22 (2x2) B21 (2x3)
    # Term 1: A21 B11 (2x3x3)
    # Split m=3 into 1+2
    # A21 (2x3) = [A21a (2x1), A21b (2x2)]
    # B11 (3x3) = [B11a (1x3), B11b (2x3)]
    # Product = A21a B11a (2x1x3) + A21b B11b (2x2x3)
    
    # A21a B11a: n=2, m=1, p=3
    U, V, W = get_schoolbook_factors(2, 1, 3)
    # m_offset is 0, but we only use first 1 col of m-block 0?
    # Wait, m_offsets[0] is 0. m_size is 3.
    # We are using sub-block of m.
    # This logic is getting complicated with offsets.
    # It's easier to just construct the full 2x3x3 decomposition directly.
    
    # Let's use a helper for 2x3x3
    def get_2x3x3_factors():
        # Split m=3 into 1+2
        # U, V, W for 2x1x3
        U1, V1, W1 = get_schoolbook_factors(2, 1, 3)
        # U, V, W for 2x2x3
        # Split p=3 into 1+2
        U2a, V2a, W2a = get_schoolbook_factors(2, 2, 1)
        U2b, V2b, W2b = get_strassen_factors() # 2x2x2
        
        # Combine
        R = 6 + 4 + 7
        U = np.zeros((2*3, R))
        V = np.zeros((3*3, R))
        W = np.zeros((3*2, R))
        
        # Part 1: m=1
        U[:2, :6] = U1
        V[:3, :6] = V1 # V is 3x3, V1 is 1x3? No, V1 is (1*3, 6) = (3,6).
        # Wait, V1 shape is (m*p, R) = (1*3, 6) = (3,6).
        # In 2x3x3, V is (3*3, R) = (9, R).
        # The first m=1 corresponds to rows 0..2 of V?
        # V index: j*p + k.
        # For m=1, j=0. k=0..2. Indices 0,1,2.
        V[0:3, :6] = V1
        W[:6, :6] = W1 # W is (p*n, R) = (3*2, R) = (6, R).
        
        # Part 2: m=2 (j=1,2)
        # A21b is 2x2. B11b is 2x3.
        # We need to map V indices for j=1,2.
        # V indices for j=1: 3..5 (k=0..2)
        # V indices for j=2: 6..8 (k=0..2)
        # V2a is (2*1, 4) = (2,4). Corresponds to j=1,2, k=0.
        # V2b is (2*2, 7) = (4,7). Corresponds to j=1,2, k=1,2.
        
        # This manual mapping is error prone.
        # Let's stick to the 71 rank construction but implement it cleanly.
        
        pass

    # Given the complexity and time limit, I will return the Rank 75 Schoolbook if I can't implement 71 cleanly in time.
    # But 71 is significantly better.
    # Let's try to implement the 71 rank construction using the block addition helper.
    
    # Reset
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add schoolbook block
    def add_schoolbook(n_l, m_l, p_l, i_off, j_off, k_off):
        U, V, W = get_schoolbook_factors(n_l, m_l, p_l)
        add_block(U, V, W, i_off, j_off, k_off, n_l, m_l, p_l)
        
    # Helper to add strassen block
    def add_strassen(i_off, j_off, k_off):
        U, V, W = get_strassen_factors()
        add_block(U, V, W, i_off, j_off, k_off, 2, 2, 2)

    # Partition: n=1+2, m=3+2, p=3+2
    
    # C11 (1x3)
    add_schoolbook(1, 3, 3, 0, 0, 0) # A11 B11
    add_schoolbook(1, 2, 3, 0, 3, 0) # A12 B21
    
    # C12 (1x2)
    add_schoolbook(1, 3, 2, 0, 0, 3) # A11 B12
    add_schoolbook(1, 2, 2, 0, 3, 3) # A12 B22
    
    # C21 (2x3)
    # A21 (2x3) B11 (3x3)
    # Split m=3 into 1+2
    add_schoolbook(2, 1, 3, 1, 0, 0) # A21a (2x1) B11a (1x3)
    # A21b (2x2) B11b (2x3)
    # Split p=3 into 1+2
    add_schoolbook(2, 2, 1, 1, 1, 0) # A21b (2x2) B11b1 (2x1)
    add_strassen(1, 1, 1)             # A21b (2x2) B11b2 (2x2)
    
    # A22 (2x2) B21 (2x3)
    # Split p=3 into 1+2
    add_schoolbook(2, 2, 1, 1, 3, 0) # A22 (2x2) B21a (2x1)
    add_strassen(1, 3, 1)             # A22 (2x2) B21b (2x2)
    
    # C22 (2x2)
    # A21 (2x3) B12 (3x2)
    # Split m=3 into 1+2
    add_schoolbook(2, 1, 2, 1, 0, 3) # A21a (2x1) B12a (1x2)
    add_strassen(1, 1, 3)             # A21b (2x2) B12b (2x2)
    
    # A22 (2x2) B22 (2x2)
    add_strassen(1, 3, 3)             # A22 (2x2) B22 (2x2)
    
    # Total Rank:
    # C11: 9 + 6 = 15
    # C12: 6 + 4 = 10
    # C21: 6 + 4 + 7 + 4 + 7 = 28
    # C22: 4 + 7 + 7 = 18
    # Total: 71
    
    U = np.column_stack(U_cols) if U_cols else np.zeros((n*m, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((m*p, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((p*n, 0))
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
