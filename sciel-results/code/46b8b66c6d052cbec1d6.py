"""
Rank-35 decomposition for 2x2 by 2x10 matrix multiplication.

Strategy: Use Strassen's rank-7 algorithm for each of the 5 column-pairs (blocks 0-4).
The tensor decomposition for 2x2 matrix multiplication has a minimal rank of 7.
Since the 5 blocks are independent in the general tensor sense, the total rank is 5 * 7 = 35.
Hybrid approaches with 'schoolbook' blocks (rank 8) or hypothetical lower ranks were analyzed and found to be suboptimal or invalid.
"""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition: 5 Strassen blocks (rank 7 each)
    rank = 35
    
    # U shape (n*m, rank) = (4, 35)
    # V shape (m*p, rank) = (20, 35)
    # W shape (p*n, rank) = (20, 35)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen terms for a single 2x2 block and replicate for each block
    # A indices (fixed): a=0, b=1, c=2, d=3
    # For block t: B columns 2t, 2t+1; C columns 2t, 2t+1
    # B indices: e=B[0,2t]=2t, f=B[0,2t+1]=2t+1, g=B[1,2t]=10+2t, h=B[1,2t+1]=10+2t+1
    # C indices: C[0,2t]=4t, C[0,2t+1]=4t+2, C[1,2t]=4t+1, C[1,2t+1]=4t+3
    
    for t in range(5):
        base = t * 7
        e_idx = 2*t
        f_idx = 2*t+1
        g_idx = 10+2*t
        h_idx = 10+2*t+1
        
        c00_idx = 4*t      # C[0, 2t]
        c01_idx = 4*t+2    # C[0, 2t+1]
        c10_idx = 4*t+1    # C[1, 2t]
        c11_idx = 4*t+3    # C[1, 2t+1]
        
        # Term 0: M1 = (a+d)(e+h)
        col = base
        U[0][col] = 1; U[3][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[c00_idx][col] = 1; W[c11_idx][col] = 1
        
        # Term 1: M2 = (c+d)e
        col = base + 1
        U[2][col] = 1; U[3][col] = 1
        V[e_idx][col] = 1
        W[c10_idx][col] = 1; W[c11_idx][col] = -1
        
        # Term 2: M3 = a(f-h)
        col = base + 2
        U[0][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[c01_idx][col] = 1; W[c11_idx][col] = 1
        
        # Term 3: M4 = d(g-e)
        col = base + 3
        U[3][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[c00_idx][col] = 1; W[c10_idx][col] = 1
        
        # Term 4: M5 = (a+b)h
        col = base + 4
        U[0][col] = 1; U[1][col] = 1
        V[h_idx][col] = 1
        W[c00_idx][col] = -1; W[c01_idx][col] = 1
        
        # Term 5: M6 = (c-a)(e+f)
        col = base + 5
        U[2][col] = 1; U[0][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[c11_idx][col] = 1
        
        # Term 6: M7 = (b-d)(g+h)
        col = base + 6
        U[1][col] = 1; U[3][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[c00_idx][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
