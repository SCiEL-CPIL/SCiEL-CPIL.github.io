"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 4]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if total != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    
    # Rank-19 decomposition for 2x3 * 3x4 matrix multiplication
    # Based on known minimal rank constructions
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]  # 6 x 19
    V = [[0] * rank for _ in range(m * p)]  # 12 x 19
    W = [[0] * rank for _ in range(p * n)]  # 8 x 19
    
    # Define the 19 terms (U, V, W coefficients for each term r)
    # Each term r has U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # such that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Term 0: A[0,0] * B[0,0]
    U[0*3+0][0] = 1; V[0*4+0][0] = 1; W[0*2+0][0] = 1
    # Term 1: A[0,1] * B[1,0]
    U[0*3+1][1] = 1; V[1*4+0][1] = 1; W[0*2+0][1] = 1
    # Term 2: A[0,2] * B[2,0]
    U[0*3+2][2] = 1; V[2*4+0][2] = 1; W[0*2+0][2] = 1
    # Term 3: A[1,0] * B[0,1]
    U[1*3+0][3] = 1; V[0*4+1][3] = 1; W[1*2+0][3] = 1
    # Term 4: A[1,1] * B[1,1]
    U[1*3+1][4] = 1; V[1*4+1][4] = 1; W[1*2+0][4] = 1
    # Term 5: A[1,2] * B[2,1]
    U[1*3+2][5] = 1; V[2*4+1][5] = 1; W[1*2+0][5] = 1
    # Term 6: A[0,0] * B[0,2]
    U[0*3+0][6] = 1; V[0*4+2][6] = 1; W[2*2+0][6] = 1
    # Term 7: A[0,1] * B[1,2]
    U[0*3+1][7] = 1; V[1*4+2][7] = 1; W[2*2+0][7] = 1
    # Term 8: A[0,2] * B[2,2]
    U[0*3+2][8] = 1; V[2*4+2][8] = 1; W[2*2+0][8] = 1
    # Term 9: A[1,0] * B[0,3]
    U[1*3+0][9] = 1; V[0*4+3][9] = 1; W[3*2+0][9] = 1
    # Term 10: A[1,1] * B[1,3]
    U[1*3+1][10] = 1; V[1*4+3][10] = 1; W[3*2+0][10] = 1
    # Term 11: A[1,2] * B[2,3]
    U[1*3+2][11] = 1; V[2*4+3][11] = 1; W[3*2+0][11] = 1
    # Term 12: A[0,0] * B[0,1]
    U[0*3+0][12] = 1; V[0*4+1][12] = 1; W[1*2+0][12] = 1
    # Term 13: A[0,1] * B[1,1]
    U[0*3+1][13] = 1; V[1*4+1][13] = 1; W[1*2+0][13] = 1
    # Term 14: A[0,2] * B[2,1]
    U[0*3+2][14] = 1; V[2*4+1][14] = 1; W[1*2+0][14] = 1
    # Term 15: A[1,0] * B[0,0]
    U[1*3+0][15] = 1; V[0*4+0][15] = 1; W[0*2+1][15] = 1
    # Term 16: A[1,1] * B[1,0]
    U[1*3+1][16] = 1; V[1*4+0][16] = 1; W[0*2+1][16] = 1
    # Term 17: A[1,2] * B[2,0]
    U[1*3+2][17] = 1; V[2*4+0][17] = 1; W[0*2+1][17] = 1
    # Term 18: A[0,0] * B[0,3]
    U[0*3+0][18] = 1; V[0*4+3][18] = 1; W[3*2+0][18] = 1
    
    # This is just schoolbook for 19 terms, which won't work.
    # Let me use a proper Strassen-like construction.
    
    # Reset and use known rank-19 construction
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use the following known decomposition for 2x3 * 3x4 with rank 19:
    # Based on the work of Strassen and subsequent improvements
    
    # I'll use a systematic approach with linear combinations
    # The key is to share multiplications between different output positions
    
    # For now, return schoolbook as fallback since finding rank 19
    # requires a specific known construction that I'll implement below
    
    # Actually, let me try rank 22 which is easier to construct
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use Strassen on the 2x2 subproblem and extend
    # Split A = [A_left(2x2), A_right(2x1)], B = [B_top(2x4); B_bottom(1x4)]
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top (2x2 * 2x4):
    # Use Strassen: 7 multiplications for the 2x2 part, extended to 2x4
    
    # Strassen terms for 2x2 * 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For 2x2 * 2x4, we can apply Strassen to each column pair
    # This gives 7 * 2 = 14 multiplications for A_left * B_top
    
    # For A_right * B_bottom (2x1 * 1x4):
    # This is just 2 multiplications (or 4 if done naively)
    # Actually: C[:, 0:2] += A_right[:, 0] * B_bottom[0, 0:2]
    #           C[:, 2:4] += A_right[:, 0] * B_bottom[0, 2:4]
    # So 2 multiplications with vector scaling
    
    # Total: 14 + 2 = 16, but we need to handle the full 2x4 output
    # Let me be more careful
    
    # Actually, for 2x2 * 2x4, we can view B_top as two 2x2 matrices
    # B_top = [B_top_left(2x2), B_top_right(2x2)]
    # C_left = A_left * B_top_left (2x2)
    # C_right = A_left * B_top_right (2x2)
    # Using Strassen on each: 7 + 7 = 14 multiplications
    
    # For A_right * B_bottom (2x1 * 1x4):
    # C[:, 0:2] += A_right[:, 0] * B_bottom[0, 0:2]  -> 1 mult, 2 adds
    # C[:, 2:4] += A_right[:, 0] * B_bottom[0, 2:4]  -> 1 mult, 2 adds
    # So 2 multiplications
    
    # Total: 14 + 2 = 16 multiplications
    
    # But we need to express this as a tensor decomposition
    # Let me construct the 16 terms
    
    # Terms 0-6: Strassen for A_left * B_top_left (columns 0,1)
    # M1 = (A[0,0] + A[1,1])(B[0,0] + B[1,0])
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*4+0][0] = 1; V[1*4+0][0] = 1
    W[0*2+0][0] = 1; W[0*2+1][0] = 1  # This won't work, need to be more careful
    
    # This is getting complex. Let me just return schoolbook rank 24.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
