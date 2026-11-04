"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_222():
    """Return (U,V,W) for 2x2x2 matrix multiplication, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # M1 = A00*(B00+B11)
    # M2 = A10*(B00+B01)
    # M3 = A00*B11
    # M4 = A11*B00
    # M5 = (A00+A11)*(B10+B11)
    # M6 = (A10+A11)*B01
    # M7 = (A00+A01)*(B10+B01)
    # C00 = M1 - M3 + M4 + M5
    # C01 = M3 + M6 - M5 - M7
    # C10 = M2 + M3 - M4 - M6
    # C11 = M1 - M2 + M4 + M7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices (k*2+i): 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = A00*(B00+B11)
    U[0, 0] = 1  # A00
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00 +=
    W[1, 0] = 1  # C01 +=
    
    # M2 = A10*(B00+B01)
    U[2, 1] = 1  # A10
    V[0, 1] = 1  # B00
    V[1, 1] = 1  # B01
    W[2, 1] = -1  # C10 -=
    W[3, 1] = -1  # C11 -=
    
    # M3 = A00*B11
    U[0, 2] = 1  # A00
    V[3, 2] = 1  # B11
    W[0, 2] = -1  # C00 -=
    W[1, 2] = 1  # C01 +=
    W[2, 2] = 1  # C10 +=
    
    # M4 = A11*B00
    U[3, 3] = 1  # A11
    V[0, 3] = 1  # B00
    W[0, 3] = 1  # C00 +=
    W[2, 3] = -1  # C10 -=
    W[3, 3] = 1  # C11 +=
    
    # M5 = (A00+A11)*(B10+B11)
    U[0, 4] = 1  # A00
    U[3, 4] = 1  # A11
    V[2, 4] = 1  # B10
    V[3, 4] = 1  # B11
    W[0, 4] = 1  # C00 +=
    W[1, 4] = -1  # C01 -=
    
    # M6 = (A10+A11)*B01
    U[2, 5] = 1  # A10
    U[3, 5] = 1  # A11
    V[1, 5] = 1  # B01
    W[1, 5] = 1  # C01 +=
    W[2, 5] = -1  # C10 -=
    
    # M7 = (A00+A01)*(B10+B01)
    U[0, 6] = 1  # A00
    U[1, 6] = 1  # A01
    V[1, 6] = 1  # B01
    V[2, 6] = 1  # B10
    W[1, 6] = -1  # C01 -=
    W[3, 6] = 1  # C11 +=
    
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2, offset):
    """Concatenate two decompositions."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    U = np.zeros((U1.shape[0], r1 + r2))
    V = np.zeros((V1.shape[0], r1 + r2))
    W = np.zeros((W1.shape[0], r1 + r2))
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    U[:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    return U, V, W

def construct():
    """Construct a rank-108 decomposition for 4x5 * 5x6 multiplication.
    
    Strategy: Split the 5-dimensional middle into 1+4.
    - 4x1x6 part: rank 24 (schoolbook)
    - 4x4x6 part: split 6 into 2+2+2, each 4x4x2 has rank 28
    Total: 24 + 3*28 = 108
    """
    n, m, p = 4, 5, 6
    rank = 108
    
    # Initialize
    U = np.zeros((n * m, rank))  # (20, 108)
    V = np.zeros((m * p, rank))  # (30, 108)
    W = np.zeros((p * n, rank))  # (24, 108)
    
    # Part 1: 4x1x6 (first column of B, first column of A's middle)
    # A is 4x5, B is 5x6. 
    # C = A * B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split j into {0} and {1,2,3,4}
    # Part 1: j=0 only
    # A_part1 is 4x1, B_part1 is 1x6
    # C_part1 = A_part1 * B_part1 (4x6)
    # Rank = 4*6 = 24
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1  # A[i, 0]
            V[0 * p + k, col] = 1  # B[0, k]
            W[k * n + i, col] = 1  # C[k*n+i]
            col += 1
    
    # Part 2: 4x4x6 (j=1,2,3,4)
    # A_part2 is 4x4 (rows i, cols j=1..4 mapped to 0..3)
    # B_part2 is 4x6 (rows j=1..4 mapped to 0..3, cols k)
    # C_part2 = A_part2 * B_part2 (4x6)
    
    # Split 6 into 2+2+2
    # For each group of 2 k-columns, we have 4x4x2
    # 4x4x2 can be done with rank 28:
    #   Split 4x4 A into 2x2 blocks
    #   A = [[A00, A01], [A10, A11]] each 2x2
    #   B is 4x2, split into B_top (2x2) and B_bot (2x2)
    #   C_top = A_top * B_top (2x2) -> rank 7
    #   C_bot = A_bot * B_bot (2x2) -> rank 7
    #   Wait, that's only 14. But 4x4x2 should be higher.
    #   Actually, C is 4x2. 
    #   C = A * B where A is 4x4, B is 4x2.
    #   C_top (rows 0,1) = A_top * B + A_bot * 0? No.
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
    #   C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k]
    #   C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k] + A[2,3]B[3,k]
    #   C[3,k] = A[3,0]B[0,k] + A[3,1]B[1,k] + A[3,2]B[2,k] + A[3,3]B[3,k]
    #   
    #   So C_top = A_top * B where A_top is 2x4, B is 4x2
    #   C_bot = A_bot * B where A_bot is 2x4, B is 4x2
    #   
    #   2x4x2: split B's 4 rows into 2+2
    #   2x4x2 = (2x2x2) + (2x2x2) -> rank 7+7=14
    #   So 4x4x2 = 14 + 14 = 28. Correct.
    
    s222_U, s222_V, s222_W = _strassen_222()
    
    for k_block in range(3):
        k_offset = k_block * 2
        # For this 4x4x2 block:
        # A_part2 is 4x4, B_block is 4x2 (cols k_offset, k_offset+1)
        
        # C_top (2x2) = A_top (2x4) * B_block (4x2)
        #   = A_top_left (2x2) * B_top (2x2) + A_top_right (2x2) * B_bot (2x2)
        # C_bot (2x2) = A_bot (2x4) * B_block (4x2)
        #   = A_bot_left (2x2) * B_top (2x2) + A_bot_right (2x2) * B_bot (2x2)
        
        # 4 Strassen multiplications of 2x2x2
        
        # 1. C_top_left = A_top_left * B_top
        # A_top_left: rows i=0,1; cols j=1,2 (original) -> local j=0,1
        # B_top: rows j=1,2 (original) -> local j=0,1; cols k=k_offset, k_offset+1
        U1, V1, W1 = s222_U, s222_V, s222_W
        for c in range(7):
            for i in range(2):  # i=0,1
                for j in range(2):  # j=0,1 -> original j=1+j
                    if U1[i * 2 + j, c] != 0:
                        U[(i) * m + (1 + j), col + c] += U1[i * 2 + j, c]
            for j in range(2):  # j=0,1 -> original j=1+j
                for k in range(2):  # k=0,1 -> original k=k_offset+k
                    if V1[j * 2 + k, c] != 0:
                        V[(1 + j) * p + (k_offset + k), col + c] += V1[j * 2 + k, c]
            for i in range(2):  # i=0,1
                for k in range(2):  # k=0,1 -> original k=k_offset+k
                    if W1[k * 2 + i, c] != 0:
                        W[(k_offset + k) * n + i, col + c] += W1[k * 2 + i, c]
        col += 7
        
        # 2. C_top_right = A_top_right * B_bot
        # A_top_right: rows i=0,1; cols j=3,4 (original) -> local j=0,1
        # B_bot: rows j=3,4 (original) -> local j=0,1; cols k=k_offset, k_offset+1
        U1, V1, W1 = s222_U, s222_V, s222_W
        for c in range(7):
            for i in range(2):  # i=0,1
                for j in range(2):  # j=0,1 -> original j=3+j
                    if U1[i * 2 + j, c] != 0:
                        U[(i) * m + (3 + j), col + c] += U1[i * 2 + j, c]
            for j in range(2):  # j=0,1 -> original j=3+j
                for k in range(2):  # k=0,1 -> original k=k_offset+k
                    if V1[j * 2 + k, c] != 0:
                        V[(3 + j) * p + (k_offset + k), col + c] += V1[j * 2 + k, c]
            for i in range(2):  # i=0,1
                for k in range(2):  # k=0,1 -> original k=k_offset+k
                    if W1[k * 2 + i, c] != 0:
                        W[(k_offset + k) * n + i, col + c] += W1[k * 2 + i, c]
        col += 7
        
        # 3. C_bot_left = A_bot_left * B_top
        # A_bot_left: rows i=2,3; cols j=1,2 (original) -> local j=0,1
        # B_top: rows j=1,2 (original) -> local j=0,1; cols k=k_offset, k_offset+1
        U1, V1, W1 = s222_U, s222_V, s222_W
        for c in range(7):
            for i in range(2):  # i=0,1 -> original i=2+i
                for j in range(2):  # j=0,1 -> original j=1+j
                    if U1[i * 2 + j, c] != 0:
                        U[(2 + i) * m + (1 + j), col + c] += U1[i * 2 + j, c]
            for j in range(2):  # j=0,1 -> original j=1+j
                for k in range(2):  # k=0,1 -> original k=k_offset+k
                    if V1[j * 2 + k, c] != 0:
                        V[(1 + j) * p + (k_offset + k), col + c] += V1[j * 2 + k, c]
            for i in range(2):  # i=0,1 -> original i=2+i
                for k in range(2):  # k=0,1 -> original k=k_offset+k
                    if W1[k * 2 + i, c] != 0:
                        W[(k_offset + k) * n + (2 + i), col + c] += W1[k * 2 + i, c]
        col += 7
        
        # 4. C_bot_right = A_bot_right * B_bot
        # A_bot_right: rows i=2,3; cols j=3,4 (original) -> local j=0,1
        # B_bot: rows j=3,4 (original) -> local j=0,1; cols k=k_offset, k_offset+1
        U1, V1, W1 = s222_U, s222_V, s222_W
        for c in range(7):
            for i in range(2):  # i=0,1 -> original i=2+i
                for j in range(2):  # j=0,1 -> original j=3+j
                    if U1[i * 2 + j, c] != 0:
                        U[(2 + i) * m + (3 + j), col + c] += U1[i * 2 + j, c]
            for j in range(2):  # j=0,1 -> original j=3+j
                for k in range(2):  # k=0,1 -> original k=k_offset+k
                    if V1[j * 2 + k, c] != 0:
                        V[(3 + j) * p + (k_offset + k), col + c] += V1[j * 2 + k, c]
            for i in range(2):  # i=0,1 -> original i=2+i
                for k in range(2):  # k=0,1 -> original k=k_offset+k
                    if W1[k * 2 + i, c] != 0:
                        W[(k_offset + k) * n + (2 + i), col + c] += W1[k * 2 + i, c]
        col += 7
    
    # Convert to lists
    U_list = [[U[i, j] for j in range(rank)] for i in range(n * m)]
    V_list = [[V[i, j] for j in range(rank)] for i in range(m * p)]
    W_list = [[W[i, j] for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

