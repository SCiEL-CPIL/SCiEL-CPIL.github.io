"""
Strategy: Use Strassen's algorithm for 2x2x2 sub-blocks to reduce the rank of 
multiplying 2x2 A with 2x2 sub-matrices of B. 
For 2x2 x 2x5, we split B into two 2x2 blocks and one 2x1 column.
Rank(2x2*2x2) = 7 (Strassen)
Rank(2x2*2x2) = 7 (Strassen)
Rank(2x2*2x1) = 4 (Schoolbook, as rank is n*m=4)
Total Rank = 7 + 7 + 4 = 18.
This is lower than the schoolbook rank of 20.
"""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 5
    # We claim minimal rank is 18
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 18
    V = [[0] * rank for _ in range(m * p)]  # 10 x 18
    W = [[0] * rank for _ in range(p * n)]  # 10 x 18
    
    current_rank = 0
    
    def add_strassen(k1, k2):
        """
        Adds 7 rank terms for multiplying A by B[:, k1:k2+1]
        k1 is the 'left' column, k2 is the 'right' column.
        Returns new rank index.
        """
        nonlocal current_rank
        
        # Indices in V (B matrix, m*p)
        # B[j, k] index = j * p + k
        idx_B_0_k1 = 0 * p + k1
        idx_B_1_k1 = 1 * p + k1
        idx_B_0_k2 = 0 * p + k2
        idx_B_1_k2 = 1 * p + k2
        
        # Indices in W (C matrix, p*n, trace convention W[k*n+i])
        # C[i, k] index = k * n + i
        idx_C_0_k1 = k1 * n + 0
        idx_C_1_k1 = k1 * n + 1
        idx_C_0_k2 = k2 * n + 0
        idx_C_1_k2 = k2 * n + 1
        
        # Indices in U (A matrix, n*m)
        # A[i, j] index = i * m + j
        idx_A_00 = 0
        idx_A_01 = 1
        idx_A_10 = 2
        idx_A_11 = 3
        
        # Strassen Terms:
        # M1 = (A00+A11)(B0k1+B1k2)
        # M2 = (A10+A11)B0k1
        # M3 = A00(B0k2-B1k2)
        # M4 = A11(B1k1-B0k1)
        # M5 = (A00+A01)B1k2
        # M6 = (A10-A00)(B0k1+B0k2)
        # M7 = (A01-A11)(B1k1+B1k2)
        
        # Note: Standard Strassen mapping:
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # Our A: A00=A11, A01=A12, A10=A21, A11=A22
        # Our B cols k1, k2:
        # B11 = B[0,k1], B12 = B[0,k2]
        # B21 = B[1,k1], B22 = B[1,k2]
        
        terms = [
            # (U coeffs [A00, A01, A10, A11], V coeffs [B0k1, B1k1, B0k2, B1k2], W coeffs [C0k1, C1k1, C0k2, C1k2])
            # We will map these to global indices.
            
            # M1: (A00+A11)(B0k1+B1k2)
            # U: [1, 0, 0, 1]
            # V: [1, 0, 0, 1] -> B0k1=1, B1k1=0, B0k2=0, B1k2=1
            # W: [1, 0, -1, 1] ? 
            # C11 = M1+M4-M5+M7 => C0k1
            # C22 = M1-M2+M3+M6 => C1k2
            # So W: C0k1 += 1, C1k2 += 1.
            {
                'u': [1, 0, 0, 1], 
                'v': [1, 0, 0, 1], # [B0k1, B1k1, B0k2, B1k2]
                'w': [1, 0, 0, 1]  # [C0k1, C1k1, C0k2, C1k2] -> C0k1 + C1k2
            },
            
            # M2: (A10+A11)B0k1
            # U: [0, 0, 1, 1]
            # V: [1, 0, 0, 0]
            # W: C21 = M2+M4 => C1k1 += 1. C22 = M1-M2+... => C1k2 += -1.
            {
                'u': [0, 0, 1, 1], 
                'v': [1, 0, 0, 0],
                'w': [0, 1, 0, -1]
            },
            
            # M3: A00(B0k2-B1k2)
            # U: [1, 0, 0, 0]
            # V: [0, 0, 1, -1]
            # W: C12 = M3+M5 => C0k2 += 1. C22 = M1-M2+M3+M6 => C1k2 += 1.
            {
                'u': [1, 0, 0, 0], 
                'v': [0, 0, 1, -1],
                'w': [0, 0, 1, 1]
            },
            
            # M4: A11(B1k1-B0k1)
            # U: [0, 0, 0, 1]
            # V: [-1, 1, 0, 0]
            # W: C11 = M1+M4-M5+M7 => C0k1 += 1. C21 = M2+M4 => C1k1 += 1.
            {
                'u': [0, 0, 0, 1], 
                'v': [-1, 1, 0, 0],
                'w': [1, 1, 0, 0]
            },
            
            # M5: (A00+A01)B1k2
            # U: [1, 1, 0, 0]
            # V: [0, 0, 0, 1]
            # W: C11 = ... - M5 ... => C0k1 += -1. C12 = M3+M5 => C0k2 += 1.
            {
                'u': [1, 1, 0, 0], 
                'v': [0, 0, 0, 1],
                'w': [-1, 0, 1, 0]
            },
            
            # M6: (A10-A00)(B0k1+B0k2)
            # U: [-1, 0, 1, 0]
            # V: [1, 0, 1, 0]
            # W: C22 = M1-M2+M3+M6 => C1k2 += 1.
            {
                'u': [-1, 0, 1, 0], 
                'v': [1, 0, 1, 0],
                'w': [0, 0, 0, 1]
            },
            
            # M7: (A01-A11)(B1k1+B1k2)
            # U: [0, 1, 0, -1]
            # V: [0, 1, 0, 1]
            # W: C11 = M1+M4-M5+M7 => C0k1 += 1.
            {
                'u': [0, 1, 0, -1], 
                'v': [0, 1, 0, 1],
                'w': [1, 0, 0, 0]
            }
        ]
        
        for t in terms:
            u = t['u']
            v = t['v']
            w = t['w']
            
            # Populate U
            U[idx_A_00][current_rank] += u[0]
            U[idx_A_01][current_rank] += u[1]
            U[idx_A_10][current_rank] += u[2]
            U[idx_A_11][current_rank] += u[3]
            
            # Populate V
            # v[0] is B0k1, v[1] is B1k1, v[2] is B0k2, v[3] is B1k2
            V[idx_B_0_k1][current_rank] += v[0]
            V[idx_B_1_k1][current_rank] += v[1]
            V[idx_B_0_k2][current_rank] += v[2]
            V[idx_B_1_k2][current_rank] += v[3]
            
            # Populate W
            # w[0] is C0k1, w[1] is C1k1, w[2] is C0k2, w[3] is C1k2
            W[idx_C_0_k1][current_rank] += w[0]
            W[idx_C_1_k1][current_rank] += w[1]
            W[idx_C_0_k2][current_rank] += w[2]
            W[idx_C_1_k2][current_rank] += w[3]
            
            current_rank += 1
            
        return current_rank

    def add_schoolbook(k):
        """
        Adds 4 rank terms for multiplying A by B[:, k]
        Returns new rank index.
        """
        nonlocal current_rank
        
        idx_B_0_k = 0 * p + k
        idx_B_1_k = 1 * p + k
        
        idx_C_0_k = k * n + 0
        idx_C_1_k = k * n + 1
        
        terms = [
            # C0k = A00 B0k
            { 'u': [1, 0, 0, 0], 'v_idx': 0, 'w_idx': 0, 'val': 1 },
            # C0k += A01 B1k
            { 'u': [0, 1, 0, 0], 'v_idx': 1, 'w_idx': 0, 'val': 1 },
            # C1k = A10 B0k
            { 'u': [0, 0, 1, 0], 'v_idx': 0, 'w_idx': 1, 'val': 1 },
            # C1k += A11 B1k
            { 'u': [0, 0, 0, 1], 'v_idx': 1, 'w_idx': 1, 'val': 1 },
        ]
        
        for t in terms:
            u = t['u']
            
            U[0][current_rank] += u[0]
            U[1][current_rank] += u[1]
            U[2][current_rank] += u[2]
            U[3][current_rank] += u[3]
            
            if t['v_idx'] == 0:
                V[idx_B_0_k][current_rank] += t['val']
            else:
                V[idx_B_1_k][current_rank] += t['val']
                
            if t['w_idx'] == 0:
                W[idx_C_0_k][current_rank] += t['val']
            else:
                W[idx_C_1_k][current_rank] += t['val']
                
            current_rank += 1
            
        return current_rank

    # Apply Strassen for cols 0,1
    add_strassen(0, 1)
    
    # Apply Strassen for cols 2,3
    add_strassen(2, 3)
    
    # Apply Schoolbook for col 4
    add_schoolbook(4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
