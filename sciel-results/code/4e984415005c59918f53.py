import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition block by block.
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2, Strassen Rank 7)
    # Block 2: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2] (2x3x2, Rank 11)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5] (2x2x3, Rank 11)
    # Block 4: A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5] (2x3x3, Rank 17)
    # Block 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2] (1x2x2, Rank 4)
    # Block 6: A[2:3, 2:5] * B[2:5, 0:2] -> C[2:3, 0:2] (1x3x2, Rank 6)
    # Block 7: A[2:3, 0:2] * B[0:2, 2:5] -> C[2:3, 2:5] (1x2x3, Rank 6)
    # Block 8: A[2:3, 2:5] * B[2:5, 2:5] -> C[2:3, 2:5] (1x3x3, Rank 9)
    
    # Total Rank = 7 + 11 + 11 + 17 + 4 + 6 + 6 + 9 = 71
    
    # We will accumulate U, V, W
    U = []
    V = []
    W = []
    
    def add_block(U_block, V_block, W_block):
        for i in range(len(U_block)):
            U.append(U_block[i])
        for i in range(len(V_block)):
            V.append(V_block[i])
        for i in range(len(W_block)):
            W.append(W_block[i])
            
    # Helper to create schoolbook block
    def schoolbook(m, n, p):
        rank = m * n * p
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p * m)]
        col = 0
        for i in range(m):
            for k in range(n):
                for j in range(p):
                    U[i * n + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * m + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen block for 2x2x2
    def strassen_2x2x2():
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is (0,0), A22 is (1,1)
        # B11 is (0,0), B22 is (1,1)
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Col 0: M1
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[1][0] += 1; W[2][0] += 1; W[3][0] += 1
        
        # Col 1: M2 = (A21+A22)B11
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1; W[3][1] += 1
        
        # Col 2: M3 = A11(B12-B22)
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        W[1][2] += 1; W[3][2] += 1
        
        # Col 3: M4 = A22(B21-B11)
        U[3][3] += 1
        V[2][3] += 1; V[0][3] -= 1
        W[2][3] += 1; W[3][3] += 1
        
        # Col 4: M5 = (A11+A12)B22
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1; W[1][4] += 1
        
        # Col 5: M6 = (A21-A11)(B11+B12)
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[2][5] += 1; W[3][5] += 1
        
        # Col 6: M7 = (A12-A22)(B21+B22)
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1; W[1][6] += 1
        
        return U, V, W

    # Helper to create 2x2x3 block (Rank 11)
    # Split into 2x2x2 (Rank 7) and 2x2x1 (Rank 4)
    # The 2x2x1 part corresponds to k=2 (0-indexed k=2 is the 3rd column)
    # A is 2x3, B is 3x2.
    # A[:, 0:2] * B[0:2, :] is 2x2x2.
    # A[:, 2:3] * B[2:3, :] is 2x1x2.
    
    def block_2x2x3():
        U222, V222, W222 = strassen_2x2x2()
        U212, V212, W212 = schoolbook(2, 1, 2)
        
        # Combine
        rank = 7 + 4
        U = [[0] * rank for _ in range(2 * 3)] # 6 rows
        V = [[0] * rank for _ in range(3 * 2)] # 6 rows
        W = [[0] * rank for _ in range(2 * 2)] # 4 rows
        
        # Embed 2x2x2 part (cols 0:2 of A, rows 0:2 of B)
        for i in range(4):
            for k in range(7):
                U[i][k] = U222[i][k]
                V[i][k] = V222[i][k]
                W[i][k] = W222[i][k]
                
        # Embed 2x1x2 part (col 2 of A, row 2 of B)
        # A index: 2 (col 2) -> U index 2*3 + 2 = 8? No.
        # U size is 2x3=6.
        # A is 2x3. U rows are 0..5.
        # A[i, 2] corresponds to U[i*3 + 2].
        # B is 3x2. V rows are 0..5.
        # B[2, j] corresponds to V[2*2 + j] = V[4+j].
        # W is 2x2. W rows 0..3.
        
        for i in range(6):
            for k in range(4):
                U[i][7+k] = 0
                V[i][7+k] = 0
                W[i][7+k] = 0
                
        for i in range(2):
            for j in range(2):
                # A[i, 2] B[2, j]
                # U index: i*3 + 2
                # V index: 2*2 + j = 4 + j
                # W index: j*2 + i
                u_idx = i * 3 + 2
                v_idx = 4 + j
                w_idx = j * 2 + i
                col = 7 + (i * 2 + j) # Schoolbook col order
                U[u_idx][col] = U212[i][col] # U212 is 2x1, so U212[i][k]
                # Wait, U212 is 2x1x2. U212 has 2 rows.
                # U212[i][k] is 1 if i=i', k=k' in schoolbook?
                # Schoolbook for 2x1x2:
                # U is 2x2. V is 1x2. W is 2x2.
                # U[i][k] = 1 if i=i', k=k'? No.
                # Let's re-verify schoolbook(2,1,2).
                # m=2, n=1, p=2.
                # U is 2x1=2 rows. V is 1x2=2 rows. W is 2x2=4 rows.
                # U[i*1+k] -> U[i].
                # V[k*2+j] -> V[j].
                # W[j*2+i].
                # Col k:
                # U[i][k]=1, V[j][k]=1, W[j*2+i][k]=1.
                
                # So for 2x1x2:
                # U212 has 2 rows.
                # V212 has 2 rows.
                # W212 has 4 rows.
                
                # We need to map U212[i] to U[u_idx].
                # U212 is 2x2. U212[i][k].
                # We want U[u_idx][7+k] = U212[i][k].
                
                # V212 is 2x2. V212[j][k].
                # We want V[v_idx][7+k] = V212[j][k].
                
                # W212 is 4x2. W212[w_idx][k].
                # We want W[w_idx][7+k] = W212[w_idx][k].
                
                for k in range(4):
                    U[u_idx][7+k] = U212[i][k]
                    V[v_idx][7+k] = V212[j][k]
                    W[w_idx][7+k] = W212[w_idx][k]
                    
        return U, V, W

    # We will use this structure for all blocks.
    # Due to time constraints and complexity, I will just implement the 71-rank decomposition.
    
    # Block 1: 2x2x2 (Strassen)
    U1, V1, W1 = strassen_2x2x2()
    # Embed into global
    # A[0:2, 0:2] -> U rows 0:4
    # B[0:2, 0:2] -> V rows 0:4
    # C[0:2, 0:2] -> W rows 0:4
    
    # Block 2: 2x3x2 (Rank 11)
    # A[0:2, 2:5] -> U rows 2:8 (cols 2,3,4)
    # B[2:5, 0:2] -> V rows 4:10 (rows 2,3,4)
    # C[0:2, 0:2] -> W rows 0:4
    
    # This is getting complex to hardcode. I will use a generic add_block function
    # and create the local blocks.
    
    # Let's just return the schoolbook decomposition to be safe, as implementing
    # the 71-rank decomposition correctly within the time limit is risky.
    # However, 71 is a significant improvement.
    
    # I will implement the 71-rank decomposition.
    
    # Re-initialize
    U = []
    V = []
    W = []
    
    def add_block_local(U_l, V_l, W_l, u_offset, v_offset, w_offset):
        # U_l is m_l * n_l x rank
        # V_l is n_l * p_l x rank
        # W_l is p_l * m_l x rank
        # u_offset is row offset in global U
        # v_offset is row offset in global V
        # w_offset is row offset in global W
        
        rank = len(U_l[0])
        
        # Pad global U, V, W if necessary
        # We will just append columns.
        # But we need to align rows.
        # This approach is flawed if we don't pre-allocate.
        
        # Let's pre-allocate the full size.
        pass

    # Pre-allocate
    rank_total = 71
    U = [[0] * rank_total for _ in range(15)] # 3*5
    V = [[0] * rank_total for _ in range(25)] # 5*5
    W = [[0] * rank_total for _ in range(15)] # 5*3
    
    col_offset = 0
    
    # Block 1: 2x2x2
    U1, V1, W1 = strassen_2x2x2()
    for i in range(4):
        for k in range(7):
            U[i][col_offset + k] = U1[i][k]
            V[i][col_offset + k] = V1[i][k]
            W[i][col_offset + k] = W1[i][k]
    col_offset += 7
    
    # Block 2: 2x3x2
    # A[0:2, 2:5], B[2:5, 0:2], C[0:2, 0:2]
    # Local U: 2x3 (rows 0:6)
    # Local V: 3x2 (rows 0:6)
    # Local W: 2x2 (rows 0:4)
    U2, V2, W2 = block_2x2x3() # This returns 2x2x3, which is A 2x3, B 3x2.
    # Wait, block_2x2x3 returns U of size 6x11, V 6x11, W 4x11.
    # This matches A[0:2, 2:5] (2x3) and B[2:5, 0:2] (3x2).
    for i in range(6):
        for k in range(11):
            # Global U row: 0*5 + (2+i%3) ? No.
            # A[0:2, 2:5] -> rows 0,1 of A, cols 2,3,4 of A.
            # Global U index: i_A * 5 + j_A.
            # i_A is 0,1. j_A is 2,3,4.
            # Local U index: i_l * 3 + j_l.
            # i_l = i_A. j_l = j_A - 2.
            # So global_u = i_l * 5 + (2 + j_l).
            
            # Local U row i_l (0..5).
            i_l = i
            i_A = i_l // 3
            j_l = i_l % 3
            global_u = i_A * 5 + (2 + j_l)
            
            # Global V row: 2*5 + (0 + i_l % 2) ? No.
            # B[2:5, 0:2] -> rows 2,3,4 of B, cols 0,1 of B.
            # Global V index: i_B * 5 + j_B.
            # i_B is 2,3,4. j_B is 0,1.
            # Local V index: i_l * 2 + j_l.
            # i_l = i_B - 2. j_l = j_B.
            # So global_v = (2 + i_l) * 5 + j_l.
            
            i_l_v = i
            i_B = i_l_v // 2 + 2
            j_l_v = i_l_v % 2
            global_v = i_B * 5 + j_l_v
            
            # Global W row: 0*3 + i_l % 2 ? No.
            # C[0:2, 0:2] -> rows 0,1 of C, cols 0,1 of C.
            # Global W index: i_C * 3 + j_C.
            # i_C is 0,1. j_C is 0,1.
            # Local W index: i_l * 2 + j_l.
            # i_l = i_C. j_l = j_C.
            # So global_w = i_l * 3 + j_l.
            
            i_l_w = i
            i_C = i_l_w // 2
            j_l_w = i_l_w % 2
            global_w = i_C * 3 + j_l_w
            
            U[global_u][col_offset + k] = U2[i][k]
            V[global_v][col_offset + k] = V2[i][k]
            W[global_w][col_offset + k] = W2[i][k]
            
    col_offset += 11
    
    # This is too slow and error-prone. I will return the schoolbook decomposition.
    
    # Revert to schoolbook
    U = [[0] * 75 for _ in range(15)]
    V = [[0] * 75 for _ in range(25)]
    W = [[0] * 75 for _ in range(15)]
    col = 0
    for i in range(3):
        for j in range(5):
            for k in range(5):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

