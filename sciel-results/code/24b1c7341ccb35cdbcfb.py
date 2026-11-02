"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank below schoolbook (72).
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140

    def elapsed():
        return time.time() - start_time

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p  # 72
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

    best_U, best_V, best_W, best_rank = schoolbook()

    # Strassen-based decomposition for 3x3 * 3x8
    # Split: A[0:2,0:2]*B[0:2,:] uses Strassen (rank 7 per 2x2x2, but B has 8 cols)
    # Actually, for 2x2 * 2x8, we can use Strassen on the 2x2 block structure.
    # The 2x2 * 2x8 multiplication: think of B as 4 column-pairs, each 2x2.
    # Strassen gives rank 7 per 2x2x2 block, so 4 blocks * 7 = 28 for the 2x2*2x8 part.
    # But we can do better: use Strassen once for the whole 2x2*2x8? No, Strassen is for square.
    # 
    # Better approach: decompose into 2x2x2 Strassen blocks for each pair of output columns.
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2]
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2]*B[2,2:4]
    # etc.
    #
    # Let's use a structured approach:
    # For the 2x2 * 2x8 part (A[0:2,0:2] * B[0:2,0:8]):
    #   Use Strassen: 7 multiplications per 2x2x2 sub-block, 4 sub-blocks = 28
    # For A[0:2,2] * B[2,0:8]: 2x1 * 1x8 = rank 16 (schoolbook)
    # For A[2,0:2] * B[0:2,0:8]: 1x2 * 2x8 = rank 16 (schoolbook)  
    # For A[2,2] * B[2,0:8]: 1x1 * 1x8 = rank 8 (schoolbook)
    # Total: 28 + 16 + 16 + 8 = 68
    #
    # But we can do even better by using Strassen more cleverly.
    # Let's implement the rank-68 Strassen-based decomposition.

    def strassen_2x2x2():
        """
        Strassen's 7-term decomposition for 2x2 * 2x2.
        Returns U (4x7), V (4x7), W (4x7) where:
        U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 products:
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a10)*b11
        # m6 = (a01-a10)*(b00+b10)
        # m7 = (a01-a11)*(b01+b11)
        
        # m1: (a00+a11)*(b00+b11) -> c00 += m1, c01 -= m3, c10 += m2, c11 += m4
        # Actually, the standard Strassen reconstruction:
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m3 + m6 + m7
        
        # m1
        U[0, 0] = 1; U[3, 0] = 1  # a00 + a11
        V[0, 0] = 1; V[3, 0] = 1  # b00 + b11
        W[0, 0] = 1; W[3, 0] = 1  # c00 += 1, c11 += 1
        
        # m2
        U[2, 1] = 1; U[3, 1] = 1  # a10 + a11
        V[0, 1] = 1               # b00
        W[2, 1] = 1               # c10 += 1
        
        # m3
        U[0, 2] = 1               # a00
        V[1, 2] = 1; V[3, 2] = -1 # b01 - b11
        W[1, 2] = 1               # c01 += 1
        
        # m4
        U[3, 3] = 1               # a11
        V[2, 3] = 1; V[0, 3] = -1 # b10 - b00
        W[3, 3] = 1; W[0, 3] = 1  # c11 += 1, c00 += 1
        
        # m5
        U[0, 4] = 1; U[2, 4] = 1  # a00 + a10
        V[3, 4] = 1               # b11
        W[0, 4] = -1; W[1, 4] = 1 # c00 -= 1, c01 += 1
        
        # m6
        U[1, 5] = 1; U[2, 5] = -1 # a01 - a10
        V[0, 5] = 1; V[2, 5] = 1  # b00 + b10
        W[3, 5] = 1               # c11 += 1
        
        # m7
        U[1, 6] = 1; U[3, 6] = -1 # a01 - a11
        V[1, 6] = 1; V[3, 6] = 1  # b01 + b11
        W[0, 6] = 1; W[3, 6] = 1  # c00 += 1, c11 += 1
        
        return U, V, W

    # Build the full 3x3 * 3x8 decomposition
    # Strategy:
    # 1. For each pair of output columns (0,1), (2,3), (4,5), (6,7):
    #    - Use Strassen for A[0:2,0:2] * B[0:2, col_pair] (rank 7 each)
    #    - Schoolbook for A[0:2,2] * B[2, col_pair] (rank 2*1*2=4 each)
    #    - Schoolbook for A[2,0:2] * B[0:2, col_pair] (rank 1*2*2=4 each)
    #    - Schoolbook for A[2,2] * B[2, col_pair] (rank 1*1*2=2 each)
    # Total: 4*(7+4+4+2) = 4*17 = 68
    
    U_full = np.zeros((9, 68))
    V_full = np.zeros((24, 68))
    W_full = np.zeros((24, 68))
    
    U_s, V_s, W_s = strassen_2x2x2()
    
    col_offset = 0
    for col_pair in range(4):
        # Columns 2*col_pair and 2*col_pair+1
        c0 = 2 * col_pair
        c1 = 2 * col_pair + 1
        
        # Strassen part: A[0:2,0:2] * B[0:2, c0:c1+1]
        for s in range(7):
            # U: A indices (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # In full U: row i*m+j where i=0,1 and j=0,1
            U_full[0, col_offset+s] = U_s[0, s]  # A[0,0]
            U_full[1, col_offset+s] = U_s[1, s]  # A[0,1]
            U_full[3, col_offset+s] = U_s[2, s]  # A[1,0]
            U_full[4, col_offset+s] = U_s[3, s]  # A[1,1]
            
            # V: B indices (0,c0)->0, (0,c1)->1, (1,c0)->2, (1,c1)->3
            # In full V: row j*p+k where j=0,1 and k=c0,c1
            V_full[0*8+c0, col_offset+s] = V_s[0, s]  # B[0,c0]
            V_full[0*8+c1, col_offset+s] = V_s[1, s]  # B[0,c1]
            V_full[1*8+c0, col_offset+s] = V_s[2, s]  # B[1,c0]
            V_full[1*8+c1, col_offset+s] = V_s[3, s]  # B[1,c1]
            
            # W: C indices (0,c0)->0, (0,c1)->1, (1,c0)->2, (1,c1)->3
            # In full W: row k*n+i where k=c0,c1 and i=0,1
            W_full[c0*3+0, col_offset+s] = W_s[0, s]  # C[0,c0]
            W_full[c0*3+1, col_offset+s] = W_s[1, s]  # C[1,c0]
            W_full[c1*3+0, col_offset+s] = W_s[2, s]  # C[0,c1]
            W_full[c1*3+1, col_offset+s] = W_s[3, s]  # C[1,c1]
        
        col_offset += 7
        
        # Schoolbook: A[0:2,2] * B[2, c0:c1+1]
        # This contributes to C[0:2, c0:c1+1]
        for i in range(2):  # rows 0,1 of C
            for k in range(2):  # cols c0, c1
                col = col_offset
                U_full[i*3+2, col] = 1  # A[i,2]
                V_full[2*8+(c0+k), col] = 1  # B[2, c0+k]
                W_full[(c0+k)*3+i, col] = 1  # C[i, c0+k]
                col_offset += 1
        
        # Schoolbook: A[2,0:2] * B[0:2, c0:c1+1]
        # This contributes to C[2, c0:c1+1]
        for k in range(2):  # cols c0, c1
            for j in range(2):  # rows 0,1 of B
                col = col_offset
                U_full[2*3+j, col] = 1  # A[2,j]
                V_full[j*8+(c0+k), col] = 1  # B[j, c0+k]
                W_full[(c0+k)*3+2, col] = 1  # C[2, c0+k]
                col_offset += 1
        
        # Schoolbook: A[2,2] * B[2, c0:c1+1]
        # This contributes to C[2, c0:c1+1]
        for k in range(2):  # cols c0, c1
            col = col_offset
            U_full[2*3+2, col] = 1  # A[2,2]
            V_full[2*8+(c0+k), col] = 1  # B[2, c0+k]
            W_full[(c0+k)*3+2, col] = 1  # C[2, c0+k]
            col_offset += 1
    
    # Verify
    if validate(U_full.tolist(), V_full.tolist(), W_full.tolist()):
        best_U, best_V, best_W, best_rank = U_full.tolist(), V_full.tolist(), W_full.tolist(), 68
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
