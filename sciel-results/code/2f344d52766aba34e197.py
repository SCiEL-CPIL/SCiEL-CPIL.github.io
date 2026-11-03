import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Define blocks
    # m splits: 2, 2, 1
    # p splits: 2, 2, 2
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # We need to map the block indices to the global indices
    # Global U: (n*m, rank)
    # Global V: (m*p, rank)
    # Global W: (p*n, rank)
    
    # We will build the full matrices by placing blocks
    
    rank_total = 0
    block_info = []
    
    for i_m, m_size in enumerate(m_splits):
        for i_p, p_size in enumerate(p_splits):
            # Block is n x m_size x p_size
            # n is always 2
            if m_size == 2 and p_size == 2:
                # Strassen 2x2x2
                rank_b = 7
                U_b, V_b, W_b = strassen_222()
            elif m_size == 1 and p_size == 2:
                # Schoolbook 2x1x2
                rank_b = 4
                U_b, V_b, W_b = schoolbook(2, 1, 2)
            elif m_size == 2 and p_size == 1:
                # Schoolbook 2x2x1
                rank_b = 4
                U_b, V_b, W_b = schoolbook(2, 2, 1)
            else:
                # Fallback to schoolbook
                rank_b = 2 * m_size * p_size
                U_b, V_b, W_b = schoolbook(2, m_size, p_size)
            
            block_info.append((i_m, i_p, m_size, p_size, rank_b, U_b, V_b, W_b))
            rank_total += rank_b
            
    # Initialize global matrices
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    col_offset = 0
    j_offset = 0
    for i_m, m_size in enumerate(m_splits):
        k_offset = 0
        for i_p, p_size in enumerate(p_splits):
            # Find the block
            for b in block_info:
                if b[0] == i_m and b[1] == i_p:
                    rank_b = b[4]
                    U_b = b[5]
                    V_b = b[6]
                    W_b = b[7]
                    
                    # Place U_b
                    # U_b is (2 * m_size, rank_b)
                    # Global U rows: i in 0..1, j in j_offset..j_offset+m_size-1
                    # Flatten index: i * m + j
                    for i in range(2):
                        for j in range(m_size):
                            global_row = i * m + (j_offset + j)
                            U[global_row, col_offset:col_offset+rank_b] = U_b[i * m_size + j, :]
                    
                    # Place V_b
                    # V_b is (m_size * p_size, rank_b)
                    # Global V rows: j in j_offset..j_offset+m_size-1, k in k_offset..k_offset+p_size-1
                    # Flatten index: j * p + k
                    for j in range(m_size):
                        for k in range(p_size):
                            global_row = (j_offset + j) * p + (k_offset + k)
                            V[global_row, col_offset:col_offset+rank_b] = V_b[j * p_size + k, :]
                            
                    # Place W_b
                    # W_b is (p_size * 2, rank_b)
                    # Global W rows: k in k_offset..k_offset+p_size-1, i in 0..1
                    # Flatten index: k * n + i
                    for k in range(p_size):
                        for i in range(2):
                            global_row = (k_offset + k) * n + i
                            W[global_row, col_offset:col_offset+rank_b] = W_b[k * 2 + i, :]
                            
                    col_offset += rank_b
                    break
            k_offset += p_size
        j_offset += m_size
        
    # Convert to lists of lists as required by interface?
    # The interface says: "factor matrices of shapes ... with integer/half-integer entries"
    # It doesn't strictly say numpy arrays, but "integer/half-integer" suggests numbers.
    # The seed program returned lists of lists.
    # I should return lists of lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def strassen_222():
    # Strassen algorithm for 2x2x2
    # U, V, W shapes: (4, 7), (4, 7), (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to decompose the tensor.
    # Each M corresponds to a rank 1 term? No, M is a product of sums.
    # The decomposition is a sum of 7 rank 1 tensors.
    
    # Let's construct the matrices directly from the known Strassen decomposition.
    
    # M1: (A11+A22)(B11+B22)
    # U: A11=1, A22=1
    # V: B11=1, B22=1
    # W: C11+=1, C22+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11
    # U: A21=1, A22=1
    # V: B11=1
    # W: C21+=1, C22-=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A11(B21-B22)
    # U: A11=1
    # V: B21=1, B22=-1
    # W: C12+=1, C22+=1
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A22(B12-B11)
    # U: A22=1
    # V: B12=1, B11=-1
    # W: C11+=1, C21+=1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A11+A12)B22
    # U: A11=1, A12=1
    # V: B22=1
    # W: C11-=1, C12+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    # U: A21=1, A11=-1
    # V: B11=1, B12=1
    # W: C22+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    # U: A12=1, A22=-1
    # V: B21=1, B22=1
    # W: C11+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W
