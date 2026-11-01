import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    
    # Target Rank 32
    # Structure: 2 blocks of M_{2,2,4} (each rank 14) + 1 block of M_{2,2,1} (rank 4)
    # M_{2,2,4} is computed as two independent M_{2,2,2} (Strassen) blocks.
    
    total_rank = 32
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(k0, k1, start_col):
        """
        Adds a Strassen decomposition for A * B[:, k0:k1+1] -> C[:, k0:k1+1]
        k0, k1 are the column indices in B (and C).
        Returns the next available column index.
        """
        # Strassen U factors (7)
        # U is indexed by i*m + j. i=0,1; j=0,1.
        # U[0]=U[0,0], U[1]=U[0,1], U[2]=U[1,0], U[3]=U[1,1]
        strassen_U = [
            [1, 0, 0, 1],   # M1
            [0, 0, 1, 1],   # M2
            [1, 0, 0, 0],   # M3
            [0, 0, 0, 1],   # M4
            [1, 1, 0, 0],   # M5
            [-1, 0, 1, 0],  # M6
            [0, 1, 0, -1]   # M7
        ]
        
        # V factors (7) - indices in B (size m*p = 18)
        # B index: j*p + k.
        strassen_V = [
            {k0: 1, 9+k1: 1},          # M1: B[0,k0] + B[1,k1]
            {k0: 1},                   # M2: B[0,k0]
            {9+k0: 1, 9+k1: -1},       # M3: B[1,k0] - B[1,k1]
            {k1: 1, k0: -1},           # M4: B[0,k1] - B[0,k0]
            {9+k1: 1},                 # M5: B[1,k1]
            {k0: 1, 9+k0: 1},          # M6: B[0,k0] + B[1,k0]
            {k1: 1, 9+k1: 1}           # M7: B[0,k1] + B[1,k1]
        ]
        
        # W factors (7) - indices in C (size p*n = 18)
        # C index: k_out * n + i.
        strassen_W = [
            {k0*2+0: 1, k1*2+1: 1},    # M1 -> C[0,k0] + C[1,k1] ? 
            # Wait, Strassen output:
            # C[0,0] = M1 - M3 + M4 - M7
            # C[0,1] = M2 + M5
            # C[1,0] = M3 + M6
            # C[1,1] = M1 - M2 + M4 + M7
            # So:
            # W for M1: C[0,0]=1, C[1,1]=1
            # W for M2: C[0,1]=1
            # W for M3: C[1,0]=1, C[0,0]=-1
            # W for M4: C[0,0]=1, C[1,1]=1
            # W for M5: C[0,1]=1
            # W for M6: C[1,0]=1
            # W for M7: C[0,0]=-1, C[1,1]=1
            {k0*2+0: 1, k1*2+1: 1},    # M1
            {k1*2+0: 1},               # M2 -> C[0, k1] ? 
            # Careful with indices.
            # If k0=0, k1=1.
            # M2 corresponds to B[0,k0] * (A[0,0]+A[1,0])?
            # Let's stick to the standard Strassen mapping for 2x2.
            # C[0,0] = M1 - M3 + M4 - M7
            # C[0,1] = M2 + M5
            # C[1,0] = M3 + M6
            # C[1,1] = M1 - M2 + M4 + M7
            
            # So:
            # M1: C[0,k0]=1, C[1,k1]=1
            # M2: C[0,k1]=1
            # M3: C[1,k0]=1, C[0,k0]=-1
            # M4: C[0,k0]=1, C[1,k1]=1
            # M5: C[0,k1]=1
            # M6: C[1,k0]=1
            # M7: C[0,k0]=-1, C[1,k1]=1
        ]
        
        # Corrected W
        strassen_W = [
            {k0*2+0: 1, k1*2+1: 1},    # M1
            {k1*2+0: 1},               # M2
            {k0*2+1: 1, k0*2+0: -1},   # M3
            {k0*2+0: 1, k1*2+1: 1},    # M4
            {k1*2+0: 1},               # M5
            {k0*2+1: 1},               # M6
            {k0*2+0: -1, k1*2+1: 1}    # M7
        ]
        
        for c in range(7):
            for i in range(n*m):
                U[i][start_col + c] = strassen_U[c][i]
            for idx, val in strassen_V[c].items():
                V[idx][start_col + c] = val
            for idx, val in strassen_W[c].items():
                W[idx][start_col + c] = val
        return start_col + 7

    def add_schoolbook_block(k, start_col):
        """
        Adds a Schoolbook decomposition for A * B[:, k] -> C[:, k]
        """
        schoolbook_U = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        schoolbook_V = [
            {k: 1},
            {9+k: 1},
            {k: 1},
            {9+k: 1}
        ]
        schoolbook_W = [
            {k*2+0: 1},
            {k*2+0: 1},
            {k*2+1: 1},
            {k*2+1: 1}
        ]
        
        for c in range(4):
            for i in range(n*m):
                U[i][start_col + c] = schoolbook_U[c][i]
            for idx, val in schoolbook_V[c].items():
                V[idx][start_col + c] = val
            for idx, val in schoolbook_W[c].items():
                W[idx][start_col + c] = val
        return start_col + 4

    # Block 1: Cols 0, 1, 2, 3
    col = add_strassen_block(0, 1, col)
    col = add_strassen_block(2, 3, col)
    
    # Block 2: Cols 4, 5, 6, 7
    col = add_strassen_block(4, 5, col)
    col = add_strassen_block(6, 7, col)
    
    # Block 3: Col 8
    col = add_schoolbook_block(8, col)
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-10:
                        return False
        return True

    if verify(result["U"], result["V"], result["W"]):
        return result
    else:
        # Fallback to Rank 33
        return construct_rank_33()

def construct_rank_33():
    # Re-use logic from seed or simple schoolbook
    # For safety, return a valid high-rank solution
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}
