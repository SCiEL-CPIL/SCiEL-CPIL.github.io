import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for 5x5x5
    # Using a 4x4x4 block (Strassen) + cross terms
    
    # Total rank estimate: 49 (Strassen 4x4) + 16 + 16 + 4 + 4 + 4 + 1 = 94
    # Let's allocate space for a slightly larger rank to be safe, or exactly 94.
    # We'll use a list of tuples (U_idx, V_idx, W_idx, value) to build the columns.
    # U_idx is i*m + j
    # V_idx is j*p + k
    # W_idx is k*n + i
    
    terms = []
    
    # 1. Strassen 4x4x4 for A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # We need a function to generate Strassen terms for N=4
    def strassen_4x4():
        # Returns list of (U_indices, V_indices, W_indices) for 4x4x4
        # U indices are (i,j) for i,j in 0..3
        # V indices are (j,k) for j,k in 0..3
        # W indices are (k,i) for k,i in 0..3
        
        # Strassen 2x2x2 terms:
        # m1 = (A00 + A11) * (B00 + B11)
        # m2 = (A10 + A11) * B00
        # m3 = A00 * (B01 - B11)
        # m4 = A11 * (B10 - B00)
        # m5 = (A00 + A01) * B11
        # m6 = (A10 - A00) * (B00 + B01)
        # m7 = (A01 - A11) * (B10 + B11)
        
        # We need to map 2x2 indices to 4x4 indices.
        # 4x4 is composed of four 2x2 blocks.
        # This recursive approach is complex to generate the explicit 49 terms manually.
        # Alternative: Use the known fact that rank(4x4x4) <= 49.
        # For simplicity and correctness within time limit, I will use the schoolbook for 4x4x4 (rank 64) 
        # and see if the total rank 64+16+16+4+4+4+1 = 109 is better than 125.
        # 109 is better than 125.
        # Can I get 49? 
        # Implementing a recursive Strassen term generator is safer if done right.
        
        # Let's try to generate the 49 terms using recursion.
        # Base case: 2x2x2 Strassen terms.
        # The 2x2x2 terms are:
        # 1: U={(0,0),(1,1)}, V={(0,0),(1,1)}, W={(0,0),(1,1)}? No.
        # The terms are linear combinations.
        # m1: U: A00, A11. V: B00, B11. W: C00, C11 (with signs).
        
        # This is getting too complex to hardcode correctly in a short time.
        # Let's stick to Schoolbook for 4x4x4 (Rank 64).
        # Total Rank = 64 + 16 + 16 + 4 + 4 + 4 + 1 = 109.
        # This is a significant improvement over 125.
        
        terms_44 = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    # U index: i, j (relative to 4x4 block, which is top-left of 5x5)
                    # V index: j, k
                    # W index: k, i
                    terms_44.append(((i, j), (j, k), (k, i), 1))
        return terms_44
    
    terms.extend(strassen_4x4())
    
    # 2. A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # A_i4 * B_4j -> C_ij
    for i in range(4):
        for j in range(4):
            # U index: (i, 4)
            # V index: (4, j)
            # W index: (j, i)
            terms.append(((i, 4), (4, j), (j, i), 1))
            
    # 3. A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    # A_4j * B_jk -> C_4k
    for j in range(4):
        for k in range(4):
            # U index: (4, j)
            # V index: (j, k)
            # W index: (k, 4)
            terms.append(((4, j), (j, k), (k, 4), 1))
            
    # 4. A[4, 4] * B[4, 0:4] -> C[4, 0:4]
    # A_44 * B_4k -> C_4k
    for k in range(4):
        # U index: (4, 4)
        # V index: (4, k)
        # W index: (k, 4)
        terms.append(((4, 4), (4, k), (k, 4), 1))
        
    # 5. A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    # A_i4 * B_44 -> C_i4
    for i in range(4):
        # U index: (i, 4)
        # V index: (4, 4)
        # W index: (4, i)
        terms.append(((i, 4), (4, 4), (4, i), 1))
        
    # 6. A[4, 0:4] * B[0:4, 4] -> C[4, 4]
    # A_4j * B_j4 -> C_44
    for j in range(4):
        # U index: (4, j)
        # V index: (j, 4)
        # W index: (4, 4)
        terms.append(((4, j), (j, 4), (4, 4), 1))
        
    # 7. A[4, 4] * B[4, 4] -> C[4, 4]
    # A_44 * B_44 -> C_44
    terms.append(((4, 4), (4, 4), (4, 4), 1))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx, val) in enumerate(terms):
        u_row = u_idx[0] * m + u_idx[1]
        v_row = v_idx[0] * p + v_idx[1]
        w_row = w_idx[0] * n + w_idx[1]
        
        U[u_row][r] = val
        V[v_row][r] = val
        W[w_row][r] = val
        
    return {"U": U, "V": V, "W": W}

