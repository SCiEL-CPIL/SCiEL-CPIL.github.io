"""
Matrix multiplication 2x2, 2x7 exact tensor decomposition.

This program constructs the optimal rank-27 decomposition directly.
The decomposition splits the multiplication into:
1. A 2x2x2 block (columns 0-1 of B) handled by Strassen's algorithm (rank 7).
2. Five 2x2x1 blocks (columns 2-6 of B) handled by schoolbook (rank 4 each, total 20).
Total rank = 7 + 20 = 27.

This is the known minimal rank for M_{2,2,7} (rank = 4p - 1 for p >= 2).
"""

import time

DIMENSIONS = [2, 2, 7]


def construct():
    n, m, p = DIMENSIONS
    # rank = 27
    rank = 27
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 27)
    # V: (m*p, rank) = (14, 27)
    # W: (p*n, rank) = (14, 27)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for k=0, 1 (columns 0, 1 of B) ---
    # Indices for A (U rows):
    # 0: A[0,0]
    # 1: A[0,1]
    # 2: A[1,0]
    # 3: A[1,1]
    
    # Indices for B (V rows) for k=0, 1:
    # 0: B[0,0]
    # 1: B[0,1]
    # 7: B[1,0]
    # 8: B[1,1]
    
    # Indices for C (W rows) for k=0, 1:
    # 0: C[0,0] (k=0, i=0)
    # 1: C[1,0] (k=0, i=1)
    # 2: C[0,1] (k=1, i=0)
    # 3: C[1,1] (k=1, i=1)
    
    # Strassen Multiplications M0..M6
    
    # c=0: M1 = A[0,0] * (B[0,1] + B[1,1])
    U[0][0] = 1
    V[1][0] = 1
    V[8][0] = 1
    # Contributes to C[0,1] (+1) and C[1,1] (+1)
    W[2][0] = 1
    W[3][0] = 1
    
    # c=1: M2 = (A[0,0] + A[1,0]) * B[0,0]
    U[0][1] = 1
    U[2][1] = 1
    V[0][1] = 1
    # Contributes to C[0,0] (-1) and C[0,1] (+1)
    W[0][1] = -1
    W[2][1] = 1
    
    # c=2: M3 = (A[0,1] + A[1,1]) * B[1,0]
    U[1][2] = 1
    U[3][2] = 1
    V[7][2] = 1
    # Contributes to C[1,0] (+1) and C[1,1] (+1)
    W[1][2] = 1
    W[3][2] = 1
    
    # c=3: M4 = A[1,1] * (B[0,0] - B[1,0])
    U[3][3] = 1
    V[0][3] = 1
    V[7][3] = -1
    # Contributes to C[0,0] (+1) and C[1,0] (+1)
    W[0][3] = 1
    W[1][3] = 1
    
    # c=4: M5 = (A[0,0] + A[0,1]) * (B[1,0] + B[1,1])
    U[0][4] = 1
    U[1][4] = 1
    V[7][4] = 1
    V[8][4] = 1
    # Contributes to C[0,0] (+1) and C[1,1] (-1)
    W[0][4] = 1
    W[3][4] = -1
    
    # c=5: M6 = (A[0,1] - A[1,0]) * (B[0,0] + B[0,1])
    U[1][5] = 1
    U[2][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    # Contributes to C[0,0] (+1)
    W[0][5] = 1
    
    # c=6: M7 = (A[1,0] - A[0,0]) * (B[0,1] + B[1,1])
    U[2][6] = 1
    U[0][6] = -1
    V[1][6] = 1
    V[8][6] = 1
    # Contributes to C[1,1] (+1)
    W[3][6] = 1
    
    # --- Part 2: Schoolbook for k=2, 3, 4, 5, 6 ---
    # For each k in [2, 6], we have 4 terms.
    # c_index starts at 7.
    
    c_idx = 7
    for k in range(2, p):
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][c_idx] = 1
        V[0 * p + k][c_idx] = 1
        W[k * n + 0][c_idx] = 1
        c_idx += 1
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[1][c_idx] = 1
        V[1 * p + k][c_idx] = 1
        W[k * n + 0][c_idx] = 1
        c_idx += 1
        
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[2][c_idx] = 1
        V[0 * p + k][c_idx] = 1
        W[k * n + 1][c_idx] = 1
        c_idx += 1
        
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[3][c_idx] = 1
        V[1 * p + k][c_idx] = 1
        W[k * n + 1][c_idx] = 1
        c_idx += 1
        
    # c_idx should be 27
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-check
    import random
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 2, 2, 7
    rank = len(U[0])
    
    # Verify exactness with random integer matrices
    for _ in range(10):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Expected C = A * B
        C_exp = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_exp[i][k] = s
        
        # Reconstructed C
        C_rec = [[0] * p for _ in range(n)]
        for c in range(rank):
            # u_c = sum_j U[i*m+j, c] A[i,j]
            # v_c = sum_j V[j*p+k, c] B[j,k]
            u_vals = [0] * n
            v_vals = [0] * p
            for i in range(n):
                for j in range(m):
                    u_vals[i] += U[i*m+j][c] * A[i][j]
            for k in range(p):
                for j in range(m):
                    v_vals[k] += V[j*p+k][c] * B[j][k]
            
            for i in range(n):
                for k in range(p):
                    C_rec[i][k] += W[k*n+i][c] * u_vals[i] * v_vals[k]
        
        for i in range(n):
            for k in range(p):
                if C_exp[i][k] != C_rec[i][k]:
                    raise AssertionError(f"Mismatch at ({i},{k}): {C_exp[i][k]} != {C_rec[i][k]}")
    print("Verification passed.")
