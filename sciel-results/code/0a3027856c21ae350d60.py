import numpy as np

def construct():
    n, m, p = 3, 4, 5
    
    # We will build the decomposition recursively.
    # A decomposition is a tuple (U, V, W, rank)
    # U: list of lists, shape (n*m, rank)
    # V: list of lists, shape (m*p, rank)
    # W: list of lists, shape (p*n, rank)
    
    def base_decomp(n, m, p):
        """
        Returns schoolbook decomposition for M_{n,m,p}.
        """
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    def strassen_decomp():
        """
        Returns rank 7 decomposition for M_{2,2,2}.
        """
        n, m, p = 2, 2, 2
        rank = 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen's 7 multiplications
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # Indices:
        # A: i in {0,1}, j in {0,1}. idx = i*2 + j
        # B: j in {0,1}, k in {0,1}. idx = j*2 + k
        # C: k in {0,1}, i in {0,1}. idx = k*2 + i
        
        # U entries (A factors)
        # M1: A11, A22 -> U[0][0]=1, U[3][0]=1
        # M2: A21, A22 -> U[2][1]=1, U[3][1]=1
        # M3: A11 -> U[0][2]=1
        # M4: A22 -> U[3][3]=1
        # M5: A11, A12 -> U[0][4]=1, U[1][4]=1
        # M6: A21, A11 -> U[2][5]=1, U[0][5]=-1
        # M7: A12, A22 -> U[1][6]=1, U[3][6]=-1
        
        U[0][0] = 1; U[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        U[0][2] = 1
        U[3][3] = 1
        U[0][4] = 1; U[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        U[1][6] = 1; U[3][6] = -1
        
        # V entries (B factors)
        # M1: B11, B22 -> V[0][0]=1, V[3][0]=1
        # M2: B11 -> V[0][1]=1
        # M3: B12, B22 -> V[1][2]=1, V[3][2]=-1
        # M4: B21, B11 -> V[2][3]=1, V[0][3]=-1
        # M5: B22 -> V[3][4]=1
        # M6: B11, B12 -> V[0][5]=1, V[1][5]=1
        # M7: B21, B22 -> V[2][6]=1, V[3][6]=1
        
        V[0][0] = 1; V[3][0] = 1
        V[0][1] = 1
        V[1][2] = 1; V[3][2] = -1
        V[2][3] = 1; V[0][3] = -1
        V[3][4] = 1
        V[0][5] = 1; V[1][5] = 1
        V[2][6] = 1; V[3][6] = 1
        
        # W entries (C factors)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # W[k*n + i, r]
        # C11: k=0, i=0 -> idx 0
        # C12: k=0, i=1 -> idx 1
        # C21: k=1, i=0 -> idx 2
        # C22: k=1, i=1 -> idx 3
        
        # W[0, :] for C11
        W[0][0] = 1  # M1
        W[0][3] = 1  # M4
        W[0][4] = -1 # M5
        W[0][6] = 1  # M7
        
        # W[1, :] for C12
        W[1][2] = 1  # M3
        W[1][4] = 1  # M5
        
        # W[2, :] for C21
        W[2][1] = 1  # M2
        W[2][3] = 1  # M4
        
        # W[3, :] for C22
        W[3][0] = 1  # M1
        W[3][1] = -1 # M2
        W[3][2] = 1  # M3
        W[3][5] = 1  # M6
        
        return U, V, W, rank

    def combine_decomps(decomp_list, n, m, p):
        """
        Combines a list of decompositions for sub-problems that partition the output.
        This assumes the sub-problems are disjoint in their output indices or can be summed directly.
        Actually, this is for summing decompositions of the SAME tensor.
        If T = T1 + T2, and we have decomps for T1 and T2, we can concatenate them.
        """
        total_rank = sum(d[3] for d in decomp_list)
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        offset = 0
        for U_d, V_d, W_d, r_d in decomp_list:
            for i in range(n * m):
                for j in range(r_d):
                    U[i][offset + j] = U_d[i][j]
            for i in range(m * p):
                for j in range(r_d):
                    V[i][offset + j] = V_d[i][j]
            for i in range(p * n):
                for j in range(r_d):
                    W[i][offset + j] = W_d[i][j]
            offset += r_d
            
        return U, V, W, total_rank

    # We need to map sub-decompositions to global indices.
    # A sub-decomposition for M_{n', m', p'} with local indices i', j', k'
    # maps to global i, j, k.
    
    def embed_decomp(U_d, V_d, W_d, rank, n, m, p, i_offset, j_offset, k_offset):
        """
        Embeds a sub-decomposition into the global space.
        i_offset, j_offset, k_offset are the starting indices for the sub-block.
        """
        n_sub = len(W_d) // n # This is tricky. W_d is p_sub * n_sub.
        # Actually, we know the dimensions of the sub-problem from the context of where it's called.
        # Let's pass n_sub, m_sub, p_sub.
        pass

    # Let's do it manually for the specific structure.
    
    # 1. Get M_{2,2,2} (Strassen)
    U_s, V_s, W_s, r_s = strassen_decomp()
    
    # 2. Get M_{1,2,2} (Schoolbook rank 4)
    U_122, V_122, W_122, r_122 = base_decomp(1, 2, 2)
    
    # 3. Get M_{2,2,1} (Schoolbook rank 4)
    U_221, V_221, W_221, r_221 = base_decomp(2, 2, 1)
    
    # 4. Get M_{1,2,3} (Schoolbook rank 6)
    U_123, V_123, W_123, r_123 = base_decomp(1, 2, 3)
    
    # Now build M_{2,2,3} (Rank 11)
    # Split B cols: 0,1 and 2.
    # Part 1: M_{2,2,2} on B[:, 0:2]
    # Part 2: M_{2,2,1} on B[:, 2]
    
    U_223 = [[0] * 11 for _ in range(4)]
    V_223 = [[0] * 11 for _ in range(6)] # m=2, p=3 -> 2*3=6
    W_223 = [[0] * 11 for _ in range(6)] # p=3, n=2 -> 3*2=6
    
    # Embed M_{2,2,2} into V cols 0,1. W cols 0,1 (k=0,1).
    # V indices: j*p + k. j in {0,1}, k in {0,1}.
    # Local V indices: j*2 + k.
    # Global V indices: j*3 + k.
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V_223[j*3 + k][r] = V_s[j*2 + k][r]
                
    # Embed M_{2,2,1} into V col 2. W col 2 (k=2).
    # Local V indices: j*1 + 0 = j.
    # Global V indices: j*3 + 2.
    for j in range(2):
        for r in range(4):
            V_223[j*3 + 2][7 + r] = V_221[j][r]
            
    # U is same for both parts (A is full 2x2).
    for i in range(4):
        for r in range(7):
            U_223[i][r] = U_s[i][r]
        for r in range(4):
            U_223[i][7 + r] = U_221[i][r]
            
    # W:
    # Part 1: k in {0,1}. Local W indices: k*2 + i.
    # Global W indices: k*2 + i (since n=2).
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W_223[k*2 + i][r] = W_s[k*2 + i][r]
                
    # Part 2: k=2. Local W indices: 0*2 + i = i.
    # Global W indices: 2*2 + i = 4 + i.
    for i in range(2):
        for r in range(4):
            W_223[4 + i][7 + r] = W_221[i][r]
            
    # 5. Build M_{3,2,2} (Rank 11)
    # Split A rows: 0,1 and 2.
    # Part 1: M_{2,2,2} on A[0:2, :]
    # Part 2: M_{1,2,2} on A[2, :]
    
    U_322 = [[0] * 11 for _ in range(6)] # n=3, m=2 -> 6
    V_322 = [[0] * 11 for _ in range(4)] # m=2, p=2 -> 4
    W_322 = [[0] * 11 for _ in range(6)] # p=2, n=3 -> 6
    
    # Part 1: A rows 0,1. U indices i*2+j. i in {0,1}.
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U_322[i*2 + j][r] = U_s[i*2 + j][r]
                
    # Part 2: A row 2. U index 2*2+j = 4+j.
    for j in range(2):
        for r in range(4):
            U_322[4 + j][7 + r] = U_122[j][r]
            
    # V is same for both (B is full 2x2).
    for i in range(4):
        for r in range(7):
            V_322[i][r] = V_s[i][r]
        for r in range(4):
            V_322[i][7 + r] = V_122[i][r]
            
    # W:
    # Part 1: i in {0,1}. Local W indices k*2+i.
    # Global W indices k*3+i.
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W_322[k*3 + i][r] = W_s[k*2 + i][r]
                
    # Part 2: i=2. Local W indices k*1+i = k.
    # Global W indices k*3 + 2.
    for k in range(2):
        for r in range(4):
            W_322[k*3 + 2][7 + r] = W_122[k][r]
            
    # 6. Build M_{3,2,3} (Rank 17)
    # Split A rows: 0,1 and 2.
    # Part 1: M_{2,2,3} on A[0:2, :]
    # Part 2: M_{1,2,3} on A[2, :]
    
    U_323 = [[0] * 17 for _ in range(6)]
    V_323 = [[0] * 17 for _ in range(6)]
    W_323 = [[0] * 17 for _ in range(9)] # p=3, n=3 -> 9
    
    # Part 1
    for i in range(4):
        for r in range(11):
            U_323[i][r] = U_223[i][r]
    for i in range(6):
        for r in range(11):
            V_323[i][r] = V_223[i][r]
    # W Part 1: i in {0,1}.
    for k in range(3):
        for i in range(2):
            for r in range(11):
                W_323[k*3 + i][r] = W_223[k*2 + i][r]
                
    # Part 2
    for j in range(2):
        for r in range(6):
            U_323[4 + j][11 + r] = U_123[j][r]
    for i in range(6):
        for r in range(6):
            V_323[i][11 + r] = V_123[i][r]
    # W Part 2: i=2.
    for k in range(3):
        for r in range(6):
            W_323[k*3 + 2][11 + r] = W_123[k][r]
            
    # 7. Build M_{3,2,5} (Rank 28)
    # Split B cols: 0,1 and 2,3,4.
    # Part 1: M_{3,2,2} on B[:, 0:2]
    # Part 2: M_{3,2,3} on B[:, 2:5]
    
    U_325 = [[0] * 28 for _ in range(6)]
    V_325 = [[0] * 28 for _ in range(10)] # m=2, p=5 -> 10
    W_325 = [[0] * 28 for _ in range(15)] # p=5, n=3 -> 15
    
    # Part 1
    for i in range(6):
        for r in range(11):
            U_325[i][r] = U_322[i][r]
    # V Part 1: k in {0,1}.
    for j in range(2):
        for k in range(2):
            for r in range(11):
                V_325[j*5 + k][r] = V_322[j*2 + k][r]
    # W Part 1: k in {0,1}.
    for k in range(2):
        for i in range(3):
            for r in range(11):
                W_325[k*3 + i][r] = W_322[k*3 + i][r]
                
    # Part 2
    for i in range(6):
        for r in range(17):
            U_325[i][11 + r] = U_323[i][r]
    # V Part 2: k in {2,3,4}.
    for j in range(2):
        for k in range(3):
            for r in range(17):
                V_325[j*5 + (k+2)][11 + r] = V_323[j*3 + k][r]
    # W Part 2: k in {2,3,4}.
    for k in range(3):
        for i in range(3):
            for r in range(17):
                W_325[(k+2)*3 + i][11 + r] = W_323[k*3 + i][r]
                
    # 8. Build M_{3,4,5} (Rank 56)
    # Split A cols: 0,1 and 2,3.
    # Part 1: M_{3,2,5} on A[:, 0:2]
    # Part 2: M_{3,2,5} on A[:, 2:4]
    
    U_final = [[0] * 56 for _ in range(12)] # n=3, m=4 -> 12
    V_final = [[0] * 56 for _ in range(20)] # m=4, p=5 -> 20
    W_final = [[0] * 56 for _ in range(15)] # p=5, n=3 -> 15
    
    # Part 1
    for i in range(3):
        for j in range(2):
            for r in range(28):
                U_final[i*4 + j][r] = U_325[i*2 + j][r]
    # V Part 1: j in {0,1}.
    for j in range(2):
        for k in range(5):
            for r in range(28):
                V_final[j*5 + k][r] = V_325[j*5 + k][r]
    # W Part 1
    for k in range(5):
        for i in range(3):
            for r in range(28):
                W_final[k*3 + i][r] = W_325[k*3 + i][r]
                
    # Part 2
    for i in range(3):
        for j in range(2):
            for r in range(28):
                U_final[i*4 + (j+2)][28 + r] = U_325[i*2 + j][r]
    # V Part 2: j in {2,3}.
    for j in range(2):
        for k in range(5):
            for r in range(28):
                V_final[(j+2)*5 + k][28 + r] = V_325[j*5 + k][r]
    # W Part 2
    for k in range(5):
        for i in range(3):
            for r in range(28):
                W_final[k*3 + i][28 + r] = W_325[k*3 + i][r]
                
    return {"U": U_final, "V": V_final, "W": W_final}
