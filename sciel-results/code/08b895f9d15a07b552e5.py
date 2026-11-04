"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _strassen_222():
    """Return (U, V, W) for M_{2,2,2} with rank 7 (Strassen)."""
    # Strassen's 7-multiplication algorithm for 2x2 matrices
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Indices: i,j in {0,1} for first matrix (2x2), j,k in {0,1} for second (2x2)
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U[i*m+j, c] = coefficient of A[i,j] in M_c
    # V[j*p+k, c] = coefficient of B[j,k] in M_c
    # W[k*n+i, c] = coefficient of C[i,k] from M_c (trace convention: W[k*n+i, c])
    
    # M1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U[0,0]=1, U[3,0]=1
    # V: B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
    # W: C00 += M1, C11 += M1 -> W[0*2+0,0]=1, W[1*2+1,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    # U: A10=1, A11=1 -> U[2,1]=1, U[3,1]=1
    # V: B00=1 -> V[0,1]=1
    # W: C10 += M2, C11 += -M2 -> W[0*2+1,1]=1, W[1*2+1,1]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A00(B10-B11)
    # U: A00=1 -> U[0,2]=1
    # V: B10=1, B11=-1 -> V[2,2]=1, V[3,2]=-1
    # W: C01 += M3, C11 += M3 -> W[0*2+0,2]=1, W[1*2+1,2]=1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[3][2] = 1
    
    # M4: A11(B00-B01)
    # U: A11=1 -> U[3,3]=1
    # V: B00=1, B01=-1 -> V[0,3]=1, V[1,3]=-1
    # W: C00 += M4, C10 += M4 -> W[0*2+0,3]=1, W[0*2+1,3]=1
    U[3][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A00+A01)B11
    # U: A00=1, A01=1 -> U[0,4]=1, U[1,4]=1
    # V: B11=1 -> V[3,4]=1
    # W: C00 += -M5, C01 += M5 -> W[0*2+0,4]=-1, W[0*2+1,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 -> U[2,5]=1, U[0,5]=-1
    # V: B00=1, B01=1 -> V[0,5]=1, V[1,5]=1
    # W: C11 += M6 -> W[1*2+1,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 -> U[1,6]=1, U[3,6]=-1
    # V: B10=1, B11=1 -> V[2,6]=1, V[3,6]=1
    # W: C00 += M7 -> W[0*2+0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    # Use Kronecker product structure: decompose 4 = 2*2
    # Build M_{3,4,7} from smaller optimal tensors
    # Strategy: M_{3,4,7} = sum of blocks
    # Split: 3=2+1, 4=2+2, 7=4+3
    # M_{3,4,7} = M_{2,2,4} + M_{2,2,3} + M_{1,2,4} + M_{1,2,3} + M_{2,2,4} + M_{2,2,3} + M_{1,2,4} + M_{1,2,3}
    # Wait, let me be more careful.
    
    # A is 3x4, B is 4x7, C is 3x7
    # Split A rows into [0:2, 2:3], B cols into [0:4, 4:7], A cols into [0:2, 2:4], B rows into [0:2, 2:4]
    # C[0:2, 0:4] = A[0:2, 0:2] * B[0:2, 0:4] + A[0:2, 2:4] * B[2:4, 0:4]
    # C[0:2, 4:7] = A[0:2, 0:2] * B[0:2, 4:7] + A[0:2, 2:4] * B[2:4, 4:7]
    # C[2:3, 0:4] = A[2:3, 0:2] * B[0:2, 0:4] + A[2:3, 2:4] * B[2:4, 0:4]
    # C[2:3, 4:7] = A[2:3, 0:2] * B[0:2, 4:7] + A[2:3, 2:4] * B[2:4, 4:7]
    
    # So we need:
    # M_{2,2,4}: rank 8 (schoolbook) or better
    # M_{2,2,3}: rank 6 (schoolbook)
    # M_{1,2,4}: rank 8 (schoolbook)
    # M_{1,2,3}: rank 6 (schoolbook)
    # And 8 blocks total: 2*(M_{2,2,4} + M_{2,2,3} + M_{1,2,4} + M_{1,2,3})
    # = 2*(8+6+8+6) = 56
    
    # That's better than 84! Let me verify: 2*(8+6+8+6) = 2*28 = 56. Yes!
    
    # But can we do better with optimal decompositions?
    # M_{2,2,4}: known rank is 8 (same as schoolbook for this case)
    # M_{2,2,3}: known rank is 6 (schoolbook)
    # M_{1,2,4}: rank 8 (schoolbook, since 1*2*4=8)
    # M_{1,2,3}: rank 6 (schoolbook)
    
    # So 56 is our target. Let me construct it.
    
    n, m, p = 3, 4, 7
    rank = 56
    U = [[0]*rank for _ in range(n*m)]  # 12 x 56
    V = [[0]*rank for _ in range(m*p)]  # 28 x 56
    W = [[0]*rank for _ in range(p*n)]  # 21 x 56
    
    col = 0
    # Block (i_block, j_block, k_block) where:
    # i_block in {0,1}: rows 0:2 or 2:3 of A, cols of C
    # j_block in {0,1}: cols 0:2 or 2:4 of A, rows of B
    # k_block in {0,1}: cols 0:4 or 4:7 of B, cols of C
    
    # For each combination, we have a small matrix multiplication
    # i_block=0 (2 rows), i_block=1 (1 row)
    # j_block=0 (2 cols), j_block=1 (2 cols)
    # k_block=0 (4 cols), k_block=1 (3 cols)
    
    for i_b in range(2):
        for j_b in range(2):
            for k_b in range(2):
                ni = 2 if i_b == 0 else 1
                nj = 2
                nk = 4 if k_b == 0 else 3
                # This is M_{ni, nj, nk}
                # Schoolbook rank = ni * nj * nk
                block_rank = ni * nj * nk
                
                # Row offset in A: i_b * 2 (since first 2 rows for i_b=0, next 1 row for i_b=1)
                # Actually: i_b=0 -> rows 0,1; i_b=1 -> row 2
                i_off = 0 if i_b == 0 else 2
                # Col offset in A: j_b * 2
                j_off = j_b * 2
                # Row offset in B: j_b * 2
                b_row_off = j_b * 2
                # Col offset in B: k_b * 4
                k_off = 0 if k_b == 0 else 4
                
                for a_i in range(ni):
                    for a_j in range(nj):
                        for b_k in range(nk):
                            # U index: (i_off + a_i) * m + (j_off + a_j)
                            u_idx = (i_off + a_i) * m + (j_off + a_j)
                            # V index: (b_row_off + a_j) * p + (k_off + b_k)
                            v_idx = (b_row_off + a_j) * p + (k_off + b_k)
                            # W index: (k_off + b_k) * n + (i_off + a_i)
                            w_idx = (k_off + b_k) * n + (i_off + a_i)
                            
                            U[u_idx][col] = 1
                            V[v_idx][col] = 1
                            W[w_idx][col] = 1
                            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

