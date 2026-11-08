"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Construct a rank-minimal decomposition of 3x3 matrix multiplication."""
    deadline = time.time() + 115
    n, m, p = 3, 3, 3
    
    # Schoolbook fallback (rank 27)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find a rank-23 decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k in {0,1,2}
    
    # Use a known rank-23 decomposition for 3x3 matmul.
    # The decomposition uses 23 terms with integer/half-integer entries.
    
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the rank-23 decomposition explicitly.
    # Based on the known result that 3x3 matmul has rank 23.
    
    # I'll use the following construction:
    # The 23 terms are defined as follows.
    
    # Terms 0-8: For each i in 0..2, j in 0..2:
    #   u[i,j,c] = 1, v[j,k,c] = 1 for all k, w[k,i,c] = 1/3
    # This gives: sum over k of (1 * 1 * 1/3) = 1 for each (i,j) when summing over k... 
    # But we need it for each (i,j,k) individually.
    
    # Let me try a different construction.
    
    # Reset
    for i in range(n * m):
        for c in range(rank):
            U[i][c] = 0
    for i in range(m * p):
        for c in range(rank):
            V[i][c] = 0
    for i in range(p * n):
        for c in range(rank):
            W[i][c] = 0
    
    # Use the following rank-23 decomposition:
    # I'll define 23 terms where each term has specific non-zero entries.
    
    # Term c=0: u[0,0]=1, v[0,0]=1, w[0,0]=1
    U[0*3+0][0] = 1; V[0*3+0][0] = 1; W[0*3+0][0] = 1
    # Term c=1: u[0,1]=1, v[1,0]=1, w[0,0]=1
    U[0*3+1][1] = 1; V[1*3+0][1] = 1; W[0*3+0][1] = 1
    # Term c=2: u[0,2]=1, v[2,0]=1, w[0,0]=1
    U[0*3+2][2] = 1; V[2*3+0][2] = 1; W[0*3+0][2] = 1
    # Term c=3: u[1,0]=1, v[0,1]=1, w[1,1]=1
    U[1*3+0][3] = 1; V[0*3+1][3] = 1; W[1*3+1][3] = 1
    # Term c=4: u[1,1]=1, v[1,1]=1, w[1,1]=1
    U[1*3+1][4] = 1; V[1*3+1][4] = 1; W[1*3+1][4] = 1
    # Term c=5: u[1,2]=1, v[2,1]=1, w[1,1]=1
    U[1*3+2][5] = 1; V[2*3+1][5] = 1; W[1*3+1][5] = 1
    # Term c=6: u[2,0]=1, v[0,2]=1, w[2,2]=1
    U[2*3+0][6] = 1; V[0*3+2][6] = 1; W[2*3+2][6] = 1
    # Term c=7: u[2,1]=1, v[1,2]=1, w[2,2]=1
    U[2*3+1][7] = 1; V[1*3+2][7] = 1; W[2*3+2][7] = 1
    # Term c=8: u[2,2]=1, v[2,2]=1, w[2,2]=1
    U[2*3+2][8] = 1; V[2*3+2][8] = 1; W[2*3+2][8] = 1
    
    # This gives 9 terms but only covers 9 of the 27 (i,j,k) triples.
    # I need 14 more terms to cover the rest.
    
    # This approach is not working well. Let me just return the fallback.
    
    return fallback

# EVOLVE-BLOCK-END
