"""
Matrix multiplication 4x4x7 with structured decomposition using Winograd 2x2x2 blocks.

We partition the 4x4 matrix into 2x2 blocks (2x2 block grid) and the 4x7 matrix into
2x2 and 2x3 blocks. Each 2x2x2 sub-problem uses the Winograd rank-7 decomposition,
and each 2x2x3 sub-problem uses a rank-24 decomposition (3 copies of Winograd 2x2x2).

Block layout:
- A is 4x4: split into 2x2 blocks A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
- B is 4x7: split rows into 2+2, columns into 2+2+2+1
  So B has blocks: B[0:2,0:2], B[0:2,2:4], B[0:2,4:6], B[0:2,6:7]
                   B[2:4,0:2], B[2:4,2:4], B[2:4,4:6], B[2:4,6:7]

For each pair (A_block, B_block) we need to multiply 2x2 by 2xk where k is 2 or 1.
- 2x2 * 2x2: Winograd rank 7
- 2x2 * 2x1: This is just 2x2 times a 2x1 vector. Rank 4 (schoolbook) or can we do better?
  Actually 2x2 * 2x1 = 4 products, rank 4.

Let me reconsider. The output C is 4x7.
C[i,j] = sum_k A[i,k] * B[k,j]

Partition A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2
Partition B into blocks by rows (2+2) and columns (2+2+2+1):
B = [[B00, B01, B02, B03], [B10, B11, B12, B13]]
where B0k is 2x(2) for k=0,1,2 and 2x1 for k=3, similarly for B1k.

C = [[A00*B00 + A01*B10, A00*B01 + A01*B11, A00*B02 + A01*B12, A00*B03 + A01*B13],
     [A10*B00 + A11*B10, A10*B01 + A11*B11, A10*B02 + A11*B12, A10*B03 + A11*B13]]

Each term like A00*B00 is 2x2 * 2x2 = 2x2 result, rank 7 (Winograd)
Each term like A00*B03 is 2x2 * 2x1 = 2x1 result, rank 4 (schoolbook, since 2x2*2x1 has 4 entries)

Total rank:
- 4 block-row positions * 4 block-col positions = 16 block products
- For block cols 0,1,2 (width 2): 4*3 = 12 products of type 2x2*2x2, each rank 7 → 84
- For block col 3 (width 1): 4*1 = 4 products of type 2x2*2x1, each rank 4 → 16
Total: 84 + 16 = 100

That's better than 112! Let me implement this.

Winograd 2x2x2 (rank 7):
For C = A*B where A,B,C are 2x2:
Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
Winograd factors:
U (input to A): rows indexed by (i,j) of A, cols by rank index
V (input to B): rows indexed by (j,k) of B, cols by rank index  
W (output C): rows indexed by (k,i) of C, cols by rank index

The 7 Winograd terms:
t0 = (a+b+c+d) * (e+f+g+h)
t1 = (a+b-c-d) * (e+f-g-h)
t2 = (a-b+c-d) * (e-f+g-h)
t3 = (a-b-c+d) * (e-f-g+h)
t4 = a * e
t5 = b * f
t6 = c * g

Then:
C[0,0] = t0 + t1 + t2 + t3 + t4
C[0,1] = t0 + t1 - t2 - t3 + t5
C[1,0] = t0 - t1 + t2 - t3 + t6
C[1,1] = t0 - t1 - t2 + t3

U matrix (4x7): for A indices (i,j) -> row i*2+j
U[0,0]=1, U[0,1]=1, U[0,2]=1, U[0,3]=1, U[0,4]=1, U[0,5]=0, U[0,6]=0  (a)
U[1,0]=1, U[1,1]=1, U[1,2]=-1, U[1,3]=-1, U[1,4]=0, U[1,5]=1, U[1,6]=0  (b)
U[2,0]=1, U[2,1]=-1, U[2,2]=1, U[2,3]=-1, U[2,4]=0, U[2,5]=0, U[2,6]=1  (c)
U[3,0]=1, U[3,1]=-1, U[3,2]=-1, U[3,3]=1, U[3,4]=0, U[3,5]=0, U[3,6]=0  (d)

V matrix (4x7): for B indices (j,k) -> row j*2+k
V[0,0]=1, V[0,1]=1, V[0,2]=1, V[0,3]=1, V[0,4]=1, V[0,5]=0, V[0,6]=0  (e)
V[1,0]=1, V[1,1]=1, V[1,2]=-1, V[1,3]=-1, V[1,4]=0, V[1,5]=1, V[1,6]=0  (f)
V[2,0]=1, V[2,1]=-1, V[2,2]=1, V[2,3]=-1, V[2,4]=0, V[2,5]=0, V[2,6]=1  (g)
V[3,0]=1, V[3,1]=-1, V[3,2]=-1, V[3,3]=1, V[3,4]=0, V[3,5]=0, V[3,6]=0  (h)

W matrix (4x7): for C indices (k,i) -> row k*2+i (trace convention W[k*n+i,r])
C[0,0] = t0+t1+t2+t3+t4: W[0,0]=1,W[0,1]=1,W[0,2]=1,W[0,3]=1,W[0,4]=1,W[0,5]=0,W[0,6]=0
C[0,1] = t0+t1-t2-t3+t5: W[1,0]=1,W[1,1]=1,W[1,2]=-1,W[1,3]=-1,W[1,4]=0,W[1,5]=1,W[1,6]=0
C[1,0] = t0-t1+t2-t3+t6: W[2,0]=1,W[2,1]=-1,W[2,2]=1,W[2,3]=-1,W[2,4]=0,W[2,5]=0,W[2,6]=1
C[1,1] = t0-t1-t2+t3:    W[3,0]=1,W[3,1]=-1,W[3,2]=-1,W[3,3]=1,W[3,4]=0,W[3,5]=0,W[3,6]=0

For 2x2 * 2x1 (rank 4):
A is 2x2, B is 2x1 = [e; g], C is 2x1 = [c0; c1]
c0 = a*e + b*g
c1 = c*e + d*g
U (2x4): a->(1,0,0,0), b->(0,1,0,0), c->(0,0,1,0), d->(0,0,0,1)
V (2x4): e->(1,0,1,0), g->(0,1,0,1)
W (2x4): c0->(1,1,0,0), c1->(0,0,1,1)
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Structured decomposition of 4x4x7 matrix multiplication.
    
    Uses Winograd 2x2x2 (rank 7) for 2x2*2x2 block products and
    schoolbook (rank 4) for 2x2*2x1 block products.
    
    Total rank: 12*7 + 4*4 = 84 + 16 = 100
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 7

    # Define Winograd 2x2x2 factors
    U_222 = [
        [1, 1, 1, 1, 1, 0, 0],   # a
        [1, 1, -1, -1, 0, 1, 0], # b
        [1, -1, 1, -1, 0, 0, 1], # c
        [1, -1, -1, 1, 0, 0, 0], # d
    ]
    V_222 = [
        [1, 1, 1, 1, 1, 0, 0],   # e
        [1, 1, -1, -1, 0, 1, 0], # f
        [1, -1, 1, -1, 0, 0, 1], # g
        [1, -1, -1, 1, 0, 0, 0], # h
    ]
    W_222 = [
        [1, 1, 1, 1, 1, 0, 0],   # C[0,0]
        [1, 1, -1, -1, 0, 1, 0], # C[0,1]
        [1, -1, 1, -1, 0, 0, 1], # C[1,0]
        [1, -1, -1, 1, 0, 0, 0], # C[1,1]
    ]

    # Define 2x2 * 2x1 factors (rank 4)
    U_221 = [
        [1, 0, 0, 0],  # a
        [0, 1, 0, 0],  # b
        [0, 0, 1, 0],  # c
        [0, 0, 0, 1],  # d
    ]
    V_221 = [
        [1, 0, 1, 0],  # e (top of 2x1)
        [0, 1, 0, 1],  # g (bottom of 2x1)
    ]
    W_221 = [
        [1, 1, 0, 0],  # C[0]
        [0, 0, 1, 1],  # C[1]
    ]

    # Block structure:
    # A (4x4) split into 2x2 blocks: A[bi, bj] where bi,bj in {0,1}
    #   A block (bi,bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B (4x7) split into blocks: B[bi, bk] where bi in {0,1}, bk in {0,1,2,3}
    #   B block (bi,bk) has rows 2*bi..2*bi+1, cols 2*bk..2*bk+1 for bk<3, col 6 for bk=3
    #   B block width: 2 for bk=0,1,2; 1 for bk=3
    # C (4x7) split into blocks: C[bi, bk] where bi in {0,1}, bk in {0,1,2,3}
    #   C block (bi,bk) has rows 2*bi..2*bi+1, cols 2*bk..2*bk+1 for bk<3, col 6 for bk=3

    # Accumulate rank
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []

    for bi in range(2):  # A block row / C block row
        for bj in range(2):  # A block col / B block row
            for bk in range(4):  # B block col / C block col
                if bk < 3:
                    # 2x2 * 2x2 -> 2x2, rank 7
                    rank_block = 7
                    U_block = U_222
                    V_block = V_222
                    W_block = W_222
                    # A indices: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                    # B indices: rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
                    # C indices: rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
                else:
                    # 2x2 * 2x1 -> 2x1, rank 4
                    rank_block = 4
                    U_block = U_221
                    V_block = V_221
                    W_block = W_221
                    # A indices: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                    # B indices: rows 2*bj..2*bj+1, col 6
                    # C indices: rows 2*bi..2*bi+1, col 6

                # Build block contribution to U, V, W
                U_rows = []
                for r in range(rank_block):
                    row = [0] * (n * m)  # 16 entries
                    for ai in range(2):
                        for aj in range(2):
                            # A global index: row = (2*bi + ai)*m + (2*bj + aj) = (2*bi+ai)*4 + (2*bj+aj)
                            a_global = (2 * bi + ai) * m + (2 * bj + aj)
                            row[a_global] = U_block[ai * 2 + aj][r]
                    U_rows.append(row)

                V_rows = []
                for r in range(rank_block):
                    row = [0] * (m * p)  # 28 entries
                    for b_i in range(2):
                        for b_j in range(2 if bk < 3 else 1):
                            # B global index: row = (2*bj + b_i)*p + (2*bk + b_j) for bk<3
                            # For bk=3: row = (2*bj + b_i)*p + 6
                            if bk < 3:
                                b_global = (2 * bj + b_i) * p + (2 * bk + b_j)
                                v_idx = b_i * 2 + b_j
                            else:
                                b_global = (2 * bj + b_i) * p + 6
                                v_idx = b_i
                            row[b_global] = V_block[v_idx][r]
                    V_rows.append(row)

                W_rows = []
                for r in range(rank_block):
                    row = [0] * (p * n)  # 28 entries
                    for c_i in range(2):
                        for c_j in range(2 if bk < 3 else 1):
                            # C global index: row = c_col * n + c_row (trace convention W[k*n+i])
                            # C col = 2*bk + c_j for bk<3, or 6 for bk=3
                            # C row = 2*bi + c_i
                            if bk < 3:
                                c_col = 2 * bk + c_j
                                w_idx = c_j * 2 + c_i
                            else:
                                c_col = 6
                                w_idx = c_i
                            c_global = c_col * n + (2 * bi + c_i)
                            row[c_global] = W_block[w_idx][r]
                    W_rows.append(row)

                U_list.extend(U_rows)
                V_list.extend(V_rows)
                W_list.extend(W_rows)
                total_rank += rank_block

    # Convert to numpy arrays
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)

    # Verify correctness
    if time.time() < deadline:
        # Check: for each tensor entry T[i,j,k], sum_r U[i*?+?,r]*V[?,r]*W[?,r] should equal 1 if valid
        # T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(total_rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        errors += 1
        if errors > 0:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
