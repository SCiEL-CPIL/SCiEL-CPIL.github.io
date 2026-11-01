"""
Program for Matrix Multiplication 2,2,7 rank-minimal decomposition.
Uses a block decomposition:
- Three 2x2x2 blocks (for k=0,1; k=2,3; k=4,5) using Strassen's algorithm (rank 7 each).
- One 2x2x1 block (for k=6) using schoolbook (rank 4).
Total rank = 3*7 + 4 = 25.

The output factor W follows the convention W[k*n+i, r] where n=2.
So for a given k and i, the row index in W is k*2 + i.
"""

import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    
    # Total rank
    rank = 25
    
    # Initialize matrices
    # U shape: (n*m, rank) = (4, 25)
    # V shape: (m*p, rank) = (14, 25)
    # W shape: (p*n, rank) = (14, 25)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen's algorithm for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    # 
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # 
    # C[0,0] = m1 - m4 + m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m3 + m6 + m7
    #
    # For each 2x2x2 block, we have:
    # A indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B indices: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C indices: C[0,0], C[0,1], C[1,0], C[1,1]
    #
    # In our global indexing:
    # A[i,j] -> U index i*m+j = i*2+j
    # B[j,k] -> V index j*p+k
    # C[i,k] -> W index k*n+i = k*2+i
    #
    # For a block with k-offset k0 (i.e., k = k0, k0+1):
    # B[0,k0] -> V index 0*p + k0 = k0
    # B[0,k0+1] -> V index 0*p + k0+1 = k0+1
    # B[1,k0] -> V index 1*p + k0 = p+k0
    # B[1,k0+1] -> V index 1*p + k0+1 = p+k0+1
    #
    # C[0,k0] -> W index k0*n + 0 = k0*2
    # C[0,k0+1] -> W index (k0+1)*n + 0 = (k0+1)*2
    # C[1,k0] -> W index k0*n + 1 = k0*2+1
    # C[1,k0+1] -> W index (k0+1)*n + 1 = (k0+1)*2+1
    
    def strassen_block(k0):
        """Add Strassen's 7 terms for the 2x2x2 block at column offset k0."""
        nonlocal col
        # e = B[0,k0], f = B[0,k0+1], g = B[1,k0], h = B[1,k0+1]
        # v_e = k0, v_f = k0+1, v_g = p+k0, v_h = p+k0+1
        v_e = k0
        v_f = k0 + 1
        v_g = p + k0
        v_h = p + k0 + 1
        
        # w for C[0,k0], C[0,k0+1], C[1,k0], C[1,k0+1]
        w_00 = k0 * n  # C[0,k0]
        w_01 = (k0 + 1) * n  # C[0,k0+1]
        w_10 = k0 * n + 1  # C[1,k0]
        w_11 = (k0 + 1) * n + 1  # C[1,k0+1]
        
        # u indices: a=0, b=1, c=2, d=3
        u_a = 0
        u_b = 1
        u_c = 2
        u_d = 3
        
        # m1 = (a+d)(e+h)
        U[u_a][col] = 1
        U[u_d][col] = 1
        V[v_e][col] = 1
        V[v_h][col] = 1
        W[w_00][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[u_c][col] = 1
        U[u_d][col] = 1
        V[v_e][col] = 1
        W[w_10][col] = 1
        col += 1
        
        # m3 = a(f-h)
        U[u_a][col] = 1
        V[v_f][col] = 1
        V[v_h][col] = -1
        W[w_01][col] = 1
        W[w_11][col] = -1
        col += 1
        
        # m4 = d(g-e)
        U[u_d][col] = 1
        V[v_g][col] = 1
        V[v_e][col] = -1
        W[w_00][col] = -1
        W[w_10][col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[u_a][col] = 1
        U[u_b][col] = 1
        V[v_h][col] = 1
        W[w_00][col] = 1
        W[w_01][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c][col] = 1
        U[u_a][col] = -1
        V[v_e][col] = 1
        V[v_f][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b][col] = 1
        U[u_d][col] = -1
        V[v_g][col] = 1
        V[v_h][col] = 1
        W[w_00][col] = -1
        W[w_11][col] = 1
        col += 1
    
    # Schoolbook block for 2x2x1 at column offset k0:
    # C[0,k0] = a*B[0,k0] + b*B[1,k0]
    # C[1,k0] = c*B[0,k0] + d*B[1,k0]
    # 4 terms:
    # m1 = a*B[0,k0] -> C[0,k0]
    # m2 = b*B[1,k0] -> C[0,k0]
    # m3 = c*B[0,k0] -> C[1,k0]
    # m4 = d*B[1,k0] -> C[1,k0]
    
    def schoolbook_block_2x2x1(k0):
        """Add schoolbook 4 terms for the 2x2x1 block at column offset k0."""
        nonlocal col
        v_e = k0  # B[0,k0]
        v_g = p + k0  # B[1,k0]
        
        w_0 = k0 * n  # C[0,k0]
        w_1 = k0 * n + 1  # C[1,k0]
        
        u_a = 0
        u_b = 1
        u_c = 2
        u_d = 3
        
        # m1 = a * B[0,k0]
        U[u_a][col] = 1
        V[v_e][col] = 1
        W[w_0][col] = 1
        col += 1
        
        # m2 = b * B[1,k0]
        U[u_b][col] = 1
        V[v_g][col] = 1
        W[w_0][col] = 1
        col += 1
        
        # m3 = c * B[0,k0]
        U[u_c][col] = 1
        V[v_e][col] = 1
        W[w_1][col] = 1
        col += 1
        
        # m4 = d * B[1,k0]
        U[u_d][col] = 1
        V[v_g][col] = 1
        W[w_1][col] = 1
        col += 1
    
    # Three Strassen blocks for k=0,1; k=2,3; k=4,5
    strassen_block(0)  # k0=0
    strassen_block(2)  # k0=2
    strassen_block(4)  # k0=4
    
    # One schoolbook block for k=6
    schoolbook_block_2x2x1(6)
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
