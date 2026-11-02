"""
Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy: Model the 2×2×10 multiplication tensor as a direct sum of five 2×2×2 blocks.
The B matrix is 2×10, which can be viewed as five independent 2×2 blocks (since 10 = 5*2).
For each block, we multiply a 2×2 matrix A by a 2×2 sub-block B_k of B.
Each 2×2×2 multiplication has rank 7 via Strassen's algorithm.
Total rank = 5 * 7 = 35.

The tensor T for multiplying A (n×m) by B (m×p) to get C (n×p) is:
T[i, j, k] = 1 if we compute C[i,k] += A[i,j] * B[j,k]

With n=2, m=2, p=10:
- A is 2×2, B is 2×10, C is 2×10.
- We can partition B into 5 blocks of size 2×2: B[:, 2k:2k+2] for k=0..4.
- Each block multiplication A * B_k gives a 2×2 contribution to C[:, 2k:2k+2].

Strassen's algorithm for 2×2×2:
Given A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B:
C = [[ae+bg, af+bh], [ce+dg, cf+dh]]

Strassen computes 7 products:
p1 = (a+d)(e+h)
p2 = (c+d)e
p3 = a(f-h)
p4 = d(g-e)
p5 = (a+b)h
p6 = (c-a)(e+f)
p7 = (b-d)(g+h)

Then:
C[0,0] = p1 + p4 - p5 + p7
C[0,1] = p3 + p5
C[1,0] = p2 + p4
C[1,1] = p1 - p2 + p3 + p6

The factor matrices for one 2×2×2 block (rank 7):
U has shape (n*m, 7) = (4, 7) where row index = i*m+j = i*2+j
V has shape (m*p, 7) = (4, 7) where row index = j*p+k = j*2+k (for a 2×2 block, p=2)
W has shape (p*n, 7) = (4, 7) where row index = k*n+i = k*2+i (for a 2×2 block, p=2, n=2)

For the full problem:
U has shape (n*m, rank) = (4, 35)
V has shape (m*p, rank) = (20, 35)
W has shape (p*n, rank) = (20, 35)

For block k (k=0..4), the columns in U, V, W corresponding to block k are:
- U columns: same for all blocks (A is shared), but we need to assign them to specific column ranges.
  Actually, U[i*m+j, col] where col = k*7 + s for s=0..6.
- V[j*p+k', col] where k' = 2k + local_k (local_k = 0 or 1), col = k*7 + s.
- W[k'*n+i, col] where k' = 2k + local_k, col = k*7 + s.

Let me define the Strassen factors for a single 2×2×2 block:

U_block (4×7):
Row 0 (i=0,j=0, a): [1, 0, 1, 0, 1, -1, 1]  -- wait, let me be more careful.

Strassen products:
p1 = (a+d)(e+h)
p2 = (c+d)e
p3 = a(f-h)
p4 = d(g-e)
p5 = (a+b)h
p6 = (c-a)(e+f)
p7 = (b-d)(g+h)

U factors (for A entries a,b,c,d at positions (0,0),(0,1),(1,0),(1,1)):
p1: a+d => U[0]=1, U[1]=0, U[2]=0, U[3]=1
p2: c+d => U[0]=0, U[1]=0, U[2]=1, U[3]=1
p3: a   => U[0]=1, U[1]=0, U[2]=0, U[3]=0
p4: d   => U[0]=0, U[1]=0, U[2]=0, U[3]=1
p5: a+b => U[0]=1, U[1]=1, U[2]=0, U[3]=0
p6: c-a => U[0]=-1, U[1]=0, U[2]=1, U[3]=0
p7: b-d => U[0]=0, U[1]=1, U[2]=0, U[3]=-1

V factors (for B entries e,f,g,h at positions (0,0),(0,1),(1,0),(1,1)):
p1: e+h => V[0]=1, V[1]=0, V[2]=0, V[3]=1
p2: e   => V[0]=1, V[1]=0, V[2]=0, V[3]=0
p3: f-h => V[0]=0, V[1]=1, V[2]=0, V[3]=-1
p4: g-e => V[0]=-1, V[1]=0, V[2]=1, V[3]=0
p5: h   => V[0]=0, V[1]=0, V[2]=0, V[3]=1
p6: e+f => V[0]=1, V[1]=1, V[2]=0, V[3]=0
p7: g+h => V[0]=0, V[1]=0, V[2]=1, V[3]=1

W factors (for C entries at positions (0,0),(0,1),(1,0),(1,1)):
C[0,0] = p1 + p4 - p5 + p7
C[0,1] = p3 + p5
C[1,0] = p2 + p4
C[1,1] = p1 - p2 + p3 + p6

W[0,0] (C[0,0]): [1, 0, 0, 1, -1, 0, 1]
W[0,1] (C[0,1]): [0, 0, 1, 0, 1, 0, 0]
W[1,0] (C[1,0]): [0, 1, 0, 1, 0, 0, 0]
W[1,1] (C[1,1]): [1, -1, 1, 0, 0, 1, 0]
"""

import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Strassen factors for 2x2x2 block
    # U_block: 4x7, rows indexed by i*m+j = i*2+j
    U_block = [
        [1,  0,  1,  0,  1, -1,  1],  # i=0,j=0 (a)
        [0,  0,  0,  0,  1,  0,  1],  # i=0,j=1 (b)
        [0,  1,  0,  0,  0,  1,  0],  # i=1,j=0 (c)
        [1,  1,  0,  1,  0,  0, -1],  # i=1,j=1 (d)
    ]
    
    # V_block: 4x7, rows indexed by j*2+k (for 2x2 block, p_block=2)
    V_block = [
        [1,  1,  0, -1,  0,  1,  0],  # j=0,k=0 (e)
        [0,  0,  1,  0,  0,  1,  0],  # j=0,k=1 (f)
        [0,  0,  0,  1,  0,  0,  1],  # j=1,k=0 (g)
        [1,  0, -1, 0,  1,  0,  1],  # j=1,k=1 (h)
    ]
    
    # W_block: 4x7, rows indexed by k*2+i (for 2x2 block, p_block=2, n=2)
    W_block = [
        [1,  0,  0,  1, -1,  0,  1],  # k=0,i=0 (C[0,0])
        [0,  0,  1,  0,  1,  0,  0],  # k=0,i=1 (C[0,1])
        [0,  1,  0,  1,  0,  0,  0],  # k=1,i=0 (C[1,0])
        [1, -1,  1,  0,  0,  1,  0],  # k=1,i=1 (C[1,1])
    ]
    
    num_blocks = p // 2  # 5 blocks
    rank = num_blocks * 7
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]      # 4 x 35
    V = [[0] * rank for _ in range(m * p)]      # 20 x 35
    W = [[0] * rank for _ in range(p * n)]      # 20 x 35
    
    for blk in range(num_blocks):
        for s in range(7):
            col = blk * 7 + s
            # U: same for all blocks
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_block[i * 2 + j][s]
            
            # V: row index = j * p + k, where k = 2*blk + local_k
            for j in range(2):
                for local_k in range(2):
                    k_global = 2 * blk + local_k
                    V[j * p + k_global][col] = V_block[j * 2 + local_k][s]
            
            # W: row index = k * n + i, where k = 2*blk + local_k
            for local_k in range(2):
                k_global = 2 * blk + local_k
                for i in range(2):
                    W[k_global * n + i][col] = W_block[local_k * 2 + i][s]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
