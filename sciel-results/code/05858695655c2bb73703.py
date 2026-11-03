import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Split p=7 into p1=3 and p2=4
    # Use Strassen for 3x4x3 (rank 7*4 = 28? No, Strassen is for square matrices)
    # Actually, for 3x4x3, we can use a block approach:
    # A is 3x4, B is 4x3, C is 3x3
    # We can write A = [A1 A2] where A1 is 3x2, A2 is 3x2
    # B = [B1; B2] where B1 is 2x3, B2 is 2x3
    # C = A1*B1 + A2*B2
    # Each 3x2x3 multiplication can be done in rank 6 (schoolbook) or less
    # Actually, let's just use schoolbook for both parts to be safe, but optimize the 3x3x3 part
    
    # Better approach: Use Strassen for the 3x3x3 core
    # Split A (3x4) into [A_left (3x3), A_right (3x1)]
    # Split B (4x7) into [B_top (3x7); B_bottom (1x7)]
    # C = A_left * B_top + A_right * B_bottom
    # A_left * B_top is 3x3x7: use Strassen-like approach
    # A_right * B_bottom is 3x1x7: rank 3 (each column independent)
    
    # Let's use a cleaner approach:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split j into j1 (0,1,2) and j2 (3)
    # C[i,k] = sum_{j1=0}^{2} A[i,j1] * B[j1,k] + A[i,3] * B[3,k]
    
    # For the first term (3x3x7), we can use Strassen decomposition
    # Strassen for 3x3x3 has rank 23, but we need 3x3x7
    # We can split k into k1 (0,1,2) and k2 (3,4,5,6)
    # C[i,k1] = sum_{j1} A[i,j1] * B[j1,k1]  (3x3x3, rank 23)
    # C[i,k2] = sum_{j1} A[i,j1] * B[j1,k2]  (3x3x4)
    
    # For 3x3x4, split k2 into pairs or use schoolbook (36)
    # Total: 23 + 36 + 3 = 62
    
    # Actually, let's just implement Strassen for 3x3x3 and schoolbook for the rest
    # Strassen 3x3x3 rank = 23
    
    def strassen_3x3x3():
        """Return U, V, W for 3x3x3 Strassen decomposition, rank 23"""
        # Strassen's algorithm for 3x3x3
        # This is a known decomposition with rank 23
        # I'll use a simplified version or fall back to schoolbook if too complex
        
        # For simplicity, use schoolbook for now (rank 27)
        # But let's try to implement a better one
        
        # Actually, the minimal rank for 3x3x3 is 23 (Strassen)
        # Let me hardcode a known decomposition
        
        # Strassen 3x3x3 decomposition (from literature)
        # This is complex, so I'll use a different approach:
        # Use the fact that rank(3x3x3) <= 23
        
        # For now, use schoolbook (rank 27) to ensure correctness
        rank = 27
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def schoolbook(n, m, p):
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def combine_decomps(d1, d2):
        U1, V1, W1, r1 = d1
        U2, V2, W2, r2 = d2
        U = U1 + [[0]*r2 for _ in range(len(U1))]
        for i in range(len(U2)):
            U.append(U2[i] + [0]*r1)
        V = V1 + [[0]*r2 for _ in range(len(V1))]
        for i in range(len(V2)):
            V.append(V2[i] + [0]*r1)
        W = W1 + [[0]*r2 for _ in range(len(W1))]
        for i in range(len(W2)):
            W.append(W2[i] + [0]*r1)
        return U, V, W, r1 + r2
    
    # Decompose 3x4x7:
    # Split B's 4 rows into first 3 and last 1
    # C = A[:,:3] * B[:3,:] + A[:,3:4] * B[3:4,:]
    
    # Part 1: 3x3x7 (A is 3x3, B is 3x7)
    # Split B's 7 columns into 3 and 4
    # C1 = A * B[:,:3] (3x3x3) + A * B[:,3:7] (3x3x4)
    
    # Part 2: 3x1x7 (A is 3x1, B is 1x7)
    
    # Let's construct these:
    
    # Part 2: 3x1x7, rank 3
    d_part2 = schoolbook(3, 1, 7)
    
    # Part 1a: 3x3x3, use schoolbook for now (rank 27)
    d_part1a = schoolbook(3, 3, 3)
    
    # Part 1b: 3x3x4, schoolbook (rank 36)
    d_part1b = schoolbook(3, 3, 4)
    
    # Combine all parts
    d1 = combine_decomps(d_part1a, d_part1b)
    U, V, W, rank = combine_decomps(d1, d_part2)
    
    # Total rank: 27 + 36 + 3 = 66
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
