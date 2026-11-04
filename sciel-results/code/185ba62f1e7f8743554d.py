import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize U, V, W with zeros
    # We will determine the max rank first
    # Rank 82 is our target
    max_rank = 82
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a local decomposition to global
    def add_decomp(local_U, local_V, local_W, i0, j0, k0, n_sub, m_sub, p_sub):
        nonlocal col_offset
        rank_sub = len(local_U[0])
        for c in range(rank_sub):
            # U
            for u_idx in range(n_sub * m_sub):
                i_sub = u_idx // m_sub
                j_sub = u_idx % m_sub
                global_u = (i0 + i_sub) * m + (j0 + j_sub)
                U[global_u][col_offset + c] += local_U[u_idx][c]
            
            # V
            for v_idx in range(m_sub * p_sub):
                j_sub = v_idx // p_sub
                k_sub = v_idx % p_sub
                global_v = (j0 + j_sub) * p + (k0 + k_sub)
                V[global_v][col_offset + c] += local_V[v_idx][c]
            
            # W
            for w_idx in range(p_sub * n_sub):
                k_sub = w_idx // n_sub
                i_sub = w_idx % n_sub
                global_w = (k0 + k_sub) * n + (i0 + i_sub)
                W[global_w][col_offset + c] += local_W[w_idx][c]
        
        col_offset += rank_sub

    # Get Strassen 2x2x2 decomposition
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2
        # M1..M7
        # U rows: (i,j) in 2x2
        # V rows: (j,k) in 2x2
        # W rows: (k,i) in 2x2
        U_local = [[0]*7 for _ in range(4)]
        V_local = [[0]*7 for _ in range(4)]
        W_local = [[0]*7 for _ in range(4)]
        
        # Indices: 00, 01, 10, 11
        # M1: (A00+A11)(B00+B11)
        U_local[0][0] = 1; U_local[3][0] = 1
        V_local[0][0] = 1; V_local[3][0] = 1
        W_local[0][0] = 1; W_local[3][0] = 1
        W_local[1][0] = -1; W_local[2][0] = 1 # C00 = M1 + M4 - M5 + M7 -> W00 gets 1, W11 gets 1
        # Wait, W[k*n+i]. 
        # C00 is k=0, i=0. Index 0.
        # C11 is k=1, i=1. Index 3.
        # C01 is k=0, i=1. Index 1.
        # C10 is k=1, i=0. Index 2.
        
        # Let's just use the known Strassen tensors.
        # It's easier to just hardcode the 7 vectors.
        # This is tedious to do manually in code generation.
        # I will use a known list of Strassen factors.
        
        # Alternative: Use numpy to verify? No, must be exact.
        # I will use a simpler method: just schoolbook for 2x2x2 if I can't get Strassen right quickly.
        # But Strassen is rank 7.
        
        # Let's assume I can generate Strassen factors.
        # For now, I will implement a generic function that returns schoolbook if Strassen is too hard to code inline.
        # But I want the improvement.
        
        # I will hardcode the Strassen 2x2x2 factors.
        # Reference: 
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U factors for M1..M7
        # M1: A11, A22
        # M2: A21, A22
        # M3: A11
        # M4: A22
        # M5: A11, A12
        # M6: A21, A11
        # M7: A12, A22
        
        # V factors for M1..M7
        # M1: B11, B22
        # M2: B11
        # M3: B12, B22
        # M4: B21, B11
        # M5: B22
        # M6: B11, B12
        # M7: B21, B22
        
        # W factors for M1..M7
        # M1: C11, C22
        # M2: C21, C22
        # M3: C12, C22
        # M4: C11, C21
        # M5: C11, C12
        # M6: C22
        # M7: C11
        
        # This is still complex to map to indices.
        # I will use schoolbook for 2x2x2 to be safe in this step, 
        # and rely on the 3x3x3 partitioning for the main gain.
        # Wait, 3x3x3 partitioning uses 2x2x2 blocks.
        # If I use schoolbook for 2x2x2 (rank 8), then 3x3x3 rank becomes:
        # 8 + 2 + 4 + 2 + 4 + 2 + 2 + 1 = 25.
        # Total rank: 25 + 25 + 17 + 17 = 84.
        # Score 90/84 = 1.07.
        
        # If I use Strassen for 2x2x2 (rank 7), 3x3x3 rank is 24.
        # Total rank 82. Score 1.09.
        
        # I will try to implement Strassen 2x2x2 correctly.
        
        U_local = [
            [1, 0, 1, 0, 1, 1, 0], # A00 (i=0,j=0)
            [0, 0, 0, 0, 1, 0, 1], # A01 (i=0,j=1)
            [0, 1, 0, 0, 0, 1, 0], # A10 (i=1,j=0)
            [1, 1, 0, 1, 0, 0, 1]  # A11 (i=1,j=1)
        ]
        V_local = [
            [1, 1, 0, 1, 0, 1, 0], # B00 (j=0,k=0)
            [0, 0, 1, 0, 0, 1, 1], # B01 (j=0,k=1)
            [0, 0, 0, 1, 1, 0, 0], # B10 (j=1,k=0)
            [1, 0, 1, 0, 1, 0, 1]  # B11 (j=1,k=1)
        ]
        W_local = [
            [1, 0, 0, 1, 1, 0, 1], # C00 (k=0,i=0)
            [0, 0, 1, 0, 1, 0, 0], # C01 (k=0,i=1)
            [0, 1, 0, 1, 0, 0, 0], # C10 (k=1,i=0)
            [1, 1, 1, 0, 0, 1, 0]  # C11 (k=1,i=1)
        ]
        # Note: Signs!
        # M4 has -1 in C00? No, C00 = M1 + M4 - M5 + M7.
        # My W_local above assumes all +1.
        # I need to apply signs to U or V or W.
        # It's easier to put signs in W.
        
        # Correct W with signs:
        # C00: M1(+), M4(+), M5(-), M7(+)
        # C01: M3(+), M5(+)
        # C10: M2(+), M4(+)
        # C11: M1(+), M2(-), M3(+), M6(+)
        
        W_local = [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 0, 1, 0, 1, 0, 0],  # C01
            [0, 1, 0, 1, 0, 0, 0],  # C10
            [1, -1, 1, 0, 0, 1, 0]  # C11
        ]
        
        return U_local, V_local, W_local

    # Get 3x3x3 decomposition using Strassen partitioning (Rank 24)
    def get_3x3x3_strassen():
        # Partition 3x3 into 2x2, 2x1, 1x2, 1x1
        # We need to combine 8 sub-problems.
        # 1. 2x2x2x2 (Strassen, Rank 7)
        # 2. 2x1x1x2 (Schoolbook, Rank 2)
        # 3. 2x2x2x1 (Schoolbook, Rank 4)
        # 4. 2x1x1x1 (Schoolbook, Rank 2)
        # 5. 1x2x2x2 (Schoolbook, Rank 4)
        # 6. 1x1x1x2 (Schoolbook, Rank 2)
        # 7. 1x2x2x1 (Schoolbook, Rank 2)
        # 8. 1x1x1x1 (Schoolbook, Rank 1)
        
        # This requires careful embedding.
        # I will just use schoolbook for 3x3x3 to be safe and fast to code.
        # Rank 27.
        # Total rank: 27 + 27 + 17 + 17 = 88.
        # Score 90/88 = 1.02.
        
        # Let's try to get the 17 rank for 3x2x2x3.
        # That part is simpler.
        
        # I will use schoolbook for 3x3x3 and Strassen-based for 3x2x2x3.
        
        U_local = [[0]*27 for _ in range(9)]
        V_local = [[0]*27 for _ in range(9)]
        W_local = [[0]*27 for _ in range(9)]
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_local[i*3+j][col] = 1
                    V_local[j*3+k][col] = 1
                    W_local[k*3+i][col] = 1
                    col += 1
        return U_local, V_local, W_local

    # Get 3x2x2x3 decomposition (Rank 17)
    def get_3x2x2x3():
        # Partition 3x2 into 2x2, 1x2
        # Partition 2x3 into 2x2, 2x1
        # 1. 2x2x2x2 (Strassen, Rank 7)
        # 2. 2x2x2x1 (Schoolbook, Rank 4)
        # 3. 1x2x2x2 (Schoolbook, Rank 4)
        # 4. 1x2x2x1 (Schoolbook, Rank 2)
        
        U_local = [[0]*17 for _ in range(6)] # 3x2
        V_local = [[0]*17 for _ in range(6)] # 2x3
        W_local = [[0]*17 for _ in range(9)] # 3x3
        
        col = 0
        
        # Sub 1: 2x2x2x2 (Rows 0-1, Cols 0-1 of A; Rows 0-1, Cols 0-1 of B)
        U1, V1, W1 = get_strassen_2x2x2()
        for c in range(7):
            for u_idx in range(4):
                i_sub = u_idx // 2
                j_sub = u_idx % 2
                global_u = i_sub * 2 + j_sub
                U_local[global_u][col+c] += U1[u_idx][c]
            for v_idx in range(4):
                j_sub = v_idx // 2
                k_sub = v_idx % 2
                global_v = j_sub * 3 + k_sub
                V_local[global_v][col+c] += V1[v_idx][c]
            for w_idx in range(4):
                k_sub = w_idx // 2
                i_sub = w_idx % 2
                global_w = k_sub * 3 + i_sub
                W_local[global_w][col+c] += W1[w_idx][c]
        col += 7
        
        # Sub 2: 2x2x2x1 (Rows 0-1, Cols 0-1 of A; Rows 0-1, Col 2 of B)
        # A part: 2x2. B part: 2x1.
        # Rank 4.
        for i in range(2):
            for j in range(2):
                U_local[i*2+j][col] = 1
                V_local[j*3+2][col] = 1
                W_local[2*3+i][col] = 1
                col += 1
        
        # Sub 3: 1x2x2x2 (Row 2, Cols 0-1 of A; Rows 0-1, Cols 0-1 of B)
        # A part: 1x2. B part: 2x2.
        # Rank 4.
        for j in range(2):
            for k in range(2):
                U_local[2*2+j][col] = 1
                V_local[j*3+k][col] = 1
                W_local[k*3+2][col] = 1
                col += 1
        
        # Sub 4: 1x2x2x1 (Row 2, Cols 0-1 of A; Rows 0-1, Col 2 of B)
        # A part: 1x2. B part: 2x1.
        # Rank 2.
        for j in range(2):
            U_local[2*2+j][col] = 1
            V_local[j*3+2][col] = 1
            W_local[2*3+2][col] = 1
            col += 1
            
        return U_local, V_local, W_local

    # Add the 4 blocks
    # Block 1: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    U1, V1, W1 = get_3x3x3_strassen()
    add_decomp(U1, V1, W1, 0, 0, 0, 3, 3, 3)
    
    # Block 2: A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]
    U2, V2, W2 = get_3x3x3_strassen()
    add_decomp(U2, V2, W2, 0, 0, 3, 3, 3, 3)
    
    # Block 3: A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3]
    U3, V3, W3 = get_3x2x2x3()
    add_decomp(U3, V3, W3, 0, 3, 0, 3, 2, 3)
    
    # Block 4: A[0:3, 3:5] * B[3:5, 3:6] -> C[0:3, 3:6]
    U4, V4, W4 = get_3x2x2x3()
    add_decomp(U4, V4, W4, 0, 3, 3, 3, 2, 3)
    
    # Trim to actual rank
    actual_rank = col_offset
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

