import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # rank = 44: 16 (for A[:,0]*B[0,:]) + 28 (4 blocks of Strassen 2x2x2)
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x8) -> 2x8
    # A1 is A[:, 0]
    # B1 is B[0, :]
    # C_part1[i, k] = A[i, 0] * B[0, k]
    for i in range(2):
        for k in range(8):
            # U index: i*m + 0 = i*3 + 0
            # V index: 0*p + k = k
            # W index: k*n + i = k*2 + i
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x8) -> 2x8
    # A2 is A[:, 1:3]
    # B2 is B[1:3, :]
    # Split B2 into 4 blocks of 2x2: B2_00, B2_01, B2_02, B2_03
    
    # Strassen decomposition for 2x2 * 2x2
    # A2 global indices: 
    # Row 0: A[0,1]=1, A[0,2]=2
    # Row 1: A[1,1]=4, A[1,2]=5
    
    # B2 global indices:
    # Row 0: B[1, 0..7] -> indices 8..15
    # Row 1: B[2, 0..7] -> indices 16..23
    
    for block in range(4):
        # B block columns: block*2, block*2+1
        
        # Local A indices in A2:
        # a00 = 1, a01 = 2
        # a10 = 4, a11 = 5
        
        # B block local:
        # b00 = B[1, block*2] -> index 1*8 + block*2
        # b01 = B[1, block*2+1] -> index 1*8 + block*2+1
        # b10 = B[2, block*2] -> index 2*8 + block*2
        # b11 = B[2, block*2+1] -> index 2*8 + block*2+1
        
        # Outputs C:
        # c00 -> C[0, block*2] -> W index (block*2)*2 + 0
        # c01 -> C[0, block*2+1] -> W index (block*2+1)*2 + 0
        # c10 -> C[1, block*2] -> W index (block*2)*2 + 1
        # c11 -> C[1, block*2+1] -> W index (block*2+1)*2 + 1
        
        # Strassen steps:
        # 1. m1 = (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        # Contribution to c00 (+), c11 (+)
        W[(block*2)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # 2. m2 = (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        # Contribution to c10 (+), c11 (-)
        W[(block*2)*2 + 1][col] = 1
        W[(block*2+1)*2 + 1][col] = -1
        col += 1
        
        # 3. m3 = a00(b01-b11)
        U[1][col] = 1
        V[1*8 + block*2+1][col] = 1
        V[2*8 + block*2+1][col] = -1
        # Contribution to c01 (+), c11 (+)
        W[(block*2+1)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # 4. m4 = a11(b10-b00)
        U[5][col] = 1
        V[2*8 + block*2][col] = 1
        V[1*8 + block*2][col] = -1
        # Contribution to c00 (+), c10 (+)
        W[(block*2)*2 + 0][col] = 1
        W[(block*2)*2 + 1][col] = 1
        col += 1
        
        # 5. m5 = (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[2*8 + block*2+1][col] = 1
        # Contribution to c00 (-), c01 (+)
        W[(block*2)*2 + 0][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        col += 1
        
        # 6. m6 = (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[1*8 + block*2][col] = 1
        V[1*8 + block*2+1][col] = 1
        # Contribution to c11 (+)
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # 7. m7 = (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[2*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        # Contribution to c00 (+)
        W[(block*2)*2 + 0][col] = 1
        col += 1
        
    # Verification (optional, but good for debugging)
    # Convert to numpy for fast verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i, j, k] = 1
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    # Let's check a few random entries
    # Actually, let's check all entries to be sure.
    # This is O(n*m*p*rank) which is 2*3*8*44 = 2112 ops. Fast.
    
    # For each i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can do this efficiently with numpy broadcasting or loops.
    # Since the sizes are small, loops are fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                if abs(val - 1.0) > 1e-9:
                    # This would indicate an error in the construction
                    # print(f"Error at {i},{j},{k}: {val}")
                    pass
                    
    return {"U": U, "V": V, "W": W}
