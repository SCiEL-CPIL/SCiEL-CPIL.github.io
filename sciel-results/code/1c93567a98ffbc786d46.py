import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a tensor decomposition for 2x2x2 (Rank 7)
    # Returns U (4x7), V (4x7), W (4x7)
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    def get_2x2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # P1 = (a+d)(e+h)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1 # C11, C22
        
        # P2 = (c+d)e
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1 # C21, C22
        
        # P3 = a(f-h)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1 # C12, C22
        
        # P4 = d(g-e)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1 # C11, C21
        
        # P5 = (a+b)h
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1 # C11, C12
        
        # P6 = (c-a)(e+f)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1 # C22
        
        # P7 = (b-d)(g+h)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1; W[1,6]=-1 # C11, C12
        
        return U, V, W

    # Helper to create a tensor decomposition for 2x2x3 (Rank 11)
    # A is 2x2, B is 2x3
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3,4,5
    # C indices: 0,1,2,3,4,5
    def get_2x2x3():
        # Use 2x2x2 for first 2 cols, 2x2x1 for last col
        U222, V222, W222 = get_2x2x2()
        
        # 2x2x1: C = A * [g, j]^T
        # C1 = a*g + b*j
        # C2 = c*g + d*j
        # Rank 4
        U221 = np.zeros((4, 4), dtype=float)
        V221 = np.zeros((4, 4), dtype=float) # B has 2 cols, but we treat as 2x1? 
        # Wait, B is 2x3. Indices 0..5.
        # Last col is indices 4,5.
        # So V221 should be 2x4? No, V shape is (m*p, rank). Here m=2, p=1. Size 2.
        # But we are embedding into 2x3.
        
        # Let's build U, V, W for 2x2x1 directly.
        U1 = np.zeros((4, 4), dtype=float)
        V1 = np.zeros((2, 4), dtype=float)
        W1 = np.zeros((2, 4), dtype=float)
        
        # Term 1: a*g
        U1[0,0]=1; V1[0,0]=1; W1[0,0]=1
        # Term 2: b*j
        U1[1,1]=1; V1[1,1]=1; W1[0,1]=1
        # Term 3: c*g
        U1[2,2]=1; V1[0,2]=1; W1[1,2]=1
        # Term 4: d*j
        U1[3,3]=1; V1[1,3]=1; W1[1,3]=1
        
        # Combine
        U = np.hstack([U222, U1])
        # V222 is 4x7 (for 2x2 B). We need 6x11.
        # B indices for 2x2 part are 0,1,2,3.
        # B indices for 2x1 part are 4,5.
        V = np.zeros((6, 11), dtype=float)
        V[0:4, 0:7] = V222
        V[4:6, 7:11] = V1
        
        # W222 is 4x7 (for 2x2 C).
        # C indices for 2x2 part are 0,1,2,3.
        # C indices for 2x1 part are 4,5.
        W = np.zeros((6, 11), dtype=float)
        W[0:4, 0:7] = W222
        W[4:6, 7:11] = W1
        
        return U, V, W

    # Construct 4x4x4 (Rank 49)
    # A, B are 4x4. C is 4x4.
    # Blocks A11, A12, A21, A22 (each 2x2)
    # Blocks B11, B12, B21, B22 (each 2x2)
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U_444 = np.zeros((16, 49), dtype=float)
    V_444 = np.zeros((16, 49), dtype=float)
    W_444 = np.zeros((16, 49), dtype=float)
    
    # A indices: 0-15. Block 0: 0-3, Block 1: 4-7, Block 2: 8-11, Block 3: 12-15.
    # B indices: 0-15. Block 0: 0-3, Block 1: 4-7, Block 2: 8-11, Block 3: 12-15.
    # C indices: 0-15. Block 0: 0-3, Block 1: 4-7, Block 2: 8-11, Block 3: 12-15.
    
    U222, V222, W222 = get_2x2x2()
    
    # For each of the 7 Strassen multiplications, we add a 2x2x2 decomposition.
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    def add_block_decomp(U, V, W, A_blocks, B_blocks, C_blocks, offset, signs_A, signs_B, signs_C):
        # signs are dicts mapping local index to coefficient
        for l in range(4):
            for c in range(7):
                # U
                if l in signs_A:
                    # A_blocks is list of (start_idx)
                    # We need to add signs_A[l] * U222[l, c] to U[A_blocks + l, offset + c]
                    pass # This logic is complex to generalize simply. 
        # Better to hardcode the 7 blocks.
        
    # Let's do it manually for clarity and correctness.
    
    # M1: (A00 + A11) * (B00 + B11) -> C00 += M1, C11 += M1
    # A00 is indices 0-3, A11 is 12-15.
    # B00 is 0-3, B11 is 12-15.
    # C00 is 0-3, C11 is 12-15.
    
    for c in range(7):
        # U
        U_444[0:4, c] += U222[:, c]
        U_444[12:16, c] += U222[:, c]
        
        # V
        V_444[0:4, c] += V222[:, c]
        V_444[12:16, c] += V222[:, c]
        
        # W
        W_444[0:4, c] += W222[:, c]
        W_444[12:16, c] += W222[:, c]
        
    # M2: (A10 + A11) * B00 -> C10 += M2, C11 -= M2
    # A10 is 8-11, A11 is 12-15.
    # B00 is 0-3.
    # C10 is 8-11, C11 is 12-15.
    
    for c in range(7):
        U_444[8:12, 7+c] += U222[:, c]
        U_444[12:16, 7+c] += U222[:, c]
        
        V_444[0:4, 7+c] += V222[:, c]
        
        W_444[8:12, 7+c] += W222[:, c]
        W_444[12:16, 7+c] -= W222[:, c]
        
    # M3: A00 * (B01 - B11) -> C01 += M3, C11 += M3
    # A00 is 0-3.
    # B01 is 4-7, B11 is 12-15.
    # C01 is 4-7, C11 is 12-15.
    
    for c in range(7):
        U_444[0:4, 14+c] += U222[:, c]
        
        V_444[4:8, 14+c] += V222[:, c]
        V_444[12:16, 14+c] -= V222[:, c]
        
        W_444[4:8, 14+c] += W222[:, c]
        W_444[12:16, 14+c] += W222[:, c]
        
    # M4: A11 * (B10 - B00) -> C00 += M4, C10 += M4
    # A11 is 12-15.
    # B10 is 8-11, B00 is 0-3.
    # C00 is 0-3, C10 is 8-11.
    
    for c in range(7):
        U_444[12:16, 21+c] += U222[:, c]
        
        V_444[8:12, 21+c] += V222[:, c]
        V_444[0:4, 21+c] -= V222[:, c]
        
        W_444[0:4, 21+c] += W222[:, c]
        W_444[8:12, 21+c] += W222[:, c]
        
    # M5: (A00 + A01) * B11 -> C00 -= M5, C01 += M5
    # A00 is 0-3, A01 is 4-7.
    # B11 is 12-15.
    # C00 is 0-3, C01 is 4-7.
    
    for c in range(7):
        U_444[0:4, 28+c] += U222[:, c]
        U_444[4:8, 28+c] += U222[:, c]
        
        V_444[12:16, 28+c] += V222[:, c]
        
        W_444[0:4, 28+c] -= W222[:, c]
        W_444[4:8, 28+c] += W222[:, c]
        
    # M6: (A10 - A00) * (B00 + B01) -> C11 += M6
    # A10 is 8-11, A00 is 0-3.
    # B00 is 0-3, B01 is 4-7.
    # C11 is 12-15.
    
    for c in range(7):
        U_444[8:12, 35+c] += U222[:, c]
        U_444[0:4, 35+c] -= U222[:, c]
        
        V_444[0:4, 35+c] += V222[:, c]
        V_444[4:8, 35+c] += V222[:, c]
        
        W_444[12:16, 35+c] += W222[:, c]
        
    # M7: (A01 - A11) * (B10 + B11) -> C00 += M7, C01 -= M7
    # A01 is 4-7, A11 is 12-15.
    # B10 is 8-11, B11 is 12-15.
    # C00 is 0-3, C01 is 4-7.
    
    for c in range(7):
        U_444[4:8, 42+c] += U222[:, c]
        U_444[12:16, 42+c] -= U222[:, c]
        
        V_444[8:12, 42+c] += V222[:, c]
        V_444[12:16, 42+c] += V222[:, c]
        
        W_444[0:4, 42+c] += W222[:, c]
        W_444[4:8, 42+c] -= W222[:, c]
        
    # Construct 4x4x3 (Rank 44)
    # A is 4x4, B is 4x3.
    # B indices: 0-11 (4 rows, 3 cols).
    # C indices: 0-11 (4 rows, 3 cols).
    
    U_443 = np.zeros((16, 44), dtype=float)
    V_443 = np.zeros((12, 44), dtype=float)
    W_443 = np.zeros((12, 44), dtype=float)
    
    U223, V223, W223 = get_2x2x3()
    
    # C_top = A00 B_top + A01 B_bot
    # A00 is 0-3, A01 is 4-7.
    # B_top is 0-5 (rows 0-1 of B), B_bot is 6-11 (rows 2-3 of B).
    # C_top is 0-5 (rows 0-1 of C).
    
    # Term 1: A00 * B_top
    for c in range(11):
        U_443[0:4, c] += U223[:, c]
        V_443[0:6, c] += V223[:, c]
        W_443[0:6, c] += W223[:, c]
        
    # Term 2: A01 * B_bot
    for c in range(11):
        U_443[4:8, 11+c] += U223[:, c]
        V_443[6:12, 11+c] += V223[:, c]
        W_443[0:6, 11+c] += W223[:, c]
        
    # C_bot = A10 B_top + A11 B_bot
    # A10 is 8-11, A11 is 12-15.
    # C_bot is 6-11 (rows 2-3 of C).
    
    # Term 3: A10 * B_top
    for c in range(11):
        U_443[8:12, 22+c] += U223[:, c]
        V_443[0:6, 22+c] += V223[:, c]
        W_443[6:12, 22+c] += W223[:, c]
        
    # Term 4: A11 * B_bot
    for c in range(11):
        U_443[12:16, 33+c] += U223[:, c]
        V_443[6:12, 33+c] += V223[:, c]
        W_443[6:12, 33+c] += W223[:, c]
        
    # Combine
    # Total Rank 93
    U = np.hstack([U_444, U_443])
    V = np.hstack([V_444, V_443])
    W = np.hstack([W_444, W_443])
    
    return {"U": U, "V": V, "W": W}
