"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Rank 42 decomposition based on Strassen's algorithm
    # M_{2,4,6} = 3 * M_{2,4,2} = 3 * (2 * M_{2,2,2}) = 6 * 7 = 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        """
        Adds a Strassen 2x2x2 decomposition.
        a_idx: [a00, a01, a10, a11] indices into U
        b_idx: [b00, b01, b10, b11] indices into V
        w_idx: [w00, w01, w10, w11] indices into W
        start_col: starting column index
        """
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    # A is 2x4. Indices i*m+j.
    # A_L (cols 0,1): [0, 1, 4, 5]
    # A_R (cols 2,3): [2, 3, 6, 7]
    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    # B is 4x6. Indices j*p+k.
    # Col block 0 (k=0,1):
    # Top (j=0,1): B[0,0]=0, B[0,1]=1, B[1,0]=6, B[1,1]=7 -> [0, 1, 6, 7]
    # Bot (j=2,3): B[2,0]=12, B[2,1]=13, B[3,0]=18, B[3,1]=19 -> [12, 13, 18, 19]
    # W indices for C cols 0,1:
    # C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    # w_idx order: [c00, c01, c10, c11] -> [0, 2, 1, 3]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    # Col block 1 (k=2,3):
    # Top (j=0,1): B[0,2]=2, B[0,3]=3, B[1,2]=8, B[1,3]=9 -> [2, 3, 8, 9]
    # Bot (j=2,3): B[2,2]=14, B[2,3]=15, B[3,2]=20, B[3,3]=21 -> [14, 15, 20, 21]
    # W indices for C cols 2,3:
    # C[0,2]->W[4], C[0,3]->W[6], C[1,2]->W[5], C[1,3]->W[7]
    # w_idx: [4, 6, 5, 7]
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    # Col block 2 (k=4,5):
    # Top (j=0,1): B[0,4]=4, B[0,5]=5, B[1,4]=10, B[1,5]=11 -> [4, 5, 10, 11]
    # Bot (j=2,3): B[2,4]=16, B[2,5]=17, B[3,4]=22, B[3,5]=23 -> [16, 17, 22, 23]
    # W indices for C cols 4,5:
    # C[0,4]->W[8], C[0,5]->W[10], C[1,4]->W[9], C[1,5]->W[11]
    # w_idx: [8, 10, 9, 11]
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: schoolbook rank 48
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
