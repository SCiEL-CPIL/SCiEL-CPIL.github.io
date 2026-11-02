"""Strassen-based decomposition for 4x4 * 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7).
    U indexed by (i,j) for A, V by (j,k) for B, W by (k,i) for C.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A11(B21-B22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22(B12-B11)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _rank9_2x2x3():
    """
    Rank-9 decomposition for 2x2 * 2x3 -> 2x3.
    Uses 7 Strassen products for the 2x2*2x2 part and 2 additional for the 2x1 part.
    
    Actually, let me think about this differently.
    We need C = A*B where A is 2x2, B is 2x3.
    
    Let B = [b1 b2 b3] where b1,b2,b3 are 2x1 columns.
    C = [A*b1, A*b2, A*b3]
    
    For the first two columns, use Strassen (7 products).
    For the third column, we need A*b3 (2x2 * 2x1 -> 2x1).
    
    A*b3:
    C1_3 = A11*b3_1 + A12*b3_2
    C2_3 = A21*b3_1 + A22*b3_2
    
    Schoolbook needs 4 products. But can we do it in 2?
    
    M8 = (A11+A21) * b3_1
    M9 = (A12+A22) * b3_2
    
    C1_3 = ? We can't get A11*b3_1 from M8 alone.
    
    Alternative: 3 products for the 2x1 part?
    M8 = A11*b3_1
    M9 = A12*b3_2  
    M10 = (A11+A12)*(b3_1+b3_2)
    
    C1_3 = M8 + M9 ... no, C1_3 = A11*b3_1 + A12*b3_2 = M8 + M9. That's 2 products!
    C2_3 = A21*b3_1 + A22*b3_2. Need 2 more.
    
    So minimum 4 for the 2x1 part with schoolbook.
    
    Total: 7 + 4 = 11. Not 9.
    
    Let me try a different approach. Maybe there's a 9-rank decomposition.
    
    Actually, I recall that the minimum rank for 2x2*2x3 might be 9 or 10.
    Let me just use 11 and accept rank 76, or find a better blocking.
    
    For now, return schoolbook 2x2x3 with rank 12.
    """
    n, m, p = 2, 2, 3
    rank = n * m * p  # 12
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    """
    Construct decomposition for 4x4 * 4x5 using Strassen for 2x2x2 blocks.
    Rank: 4 blocks * (7 + 12) = 76.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build block-by-block
    U_full = {}
    V_full = {}
    W_full = {}
    total_rank = 0
    
    for bi in range(2):  # row block of A and C (0,1)
        for bj in range(2):  # col block of A = row block of B (0,1)
            # C[bi][0] += A[bi][bj] * B[bj][0] (2x2 * 2x2 -> 2x2)
            U_b, V_b, W_b = _strassen_2x2x2()
            for a_i in range(2):
                for a_j in range(2):
                    i = bi*2 + a_i
                    j = bj*2 + a_j
                    for c in range(7):
                        val = U_b[a_i*2 + a_j][c]
                        if val != 0:
                            U_full.setdefault((i,j), []).append((val, total_rank + c))
            for b_j in range(2):
                for b_k in range(2):
                    j = bj*2 + b_j
                    k = b_k
                    for c in range(7):
                        val = V_b[b_j*2 + b_k][c]
                        if val != 0:
                            V_full.setdefault((j,k), []).append((val, total_rank + c))
            for c_k in range(2):
                for c_i in range(2):
                    k = c_k
                    i = bi*2 + c_i
                    for c in range(7):
                        val = W_b[c_k*2 + c_i][c]
                        if val != 0:
                            W_full.setdefault((k,i), []).append((val, total_rank + c))
            total_rank += 7
            
            # C[bi][1] += A[bi][bj] * B[bj][1] (2x2 * 2x3 -> 2x3)
            U_b, V_b, W_b = _rank9_2x2x3()
            for a_i in range(2):
                for a_j in range(2):
                    i = bi*2 + a_i
                    j = bj*2 + a_j
                    for c in range(12):
                        val = U_b[a_i*2 + a_j][c]
                        if val != 0:
                            U_full.setdefault((i,j), []).append((val, total_rank + c))
            for b_j in range(2):
                for b_k in range(3):
                    j = bj*2 + b_j
                    k = 2 + b_k
                    for c in range(12):
                        val = V_b[b_j*3 + b_k][c]
                        if val != 0:
                            V_full.setdefault((j,k), []).append((val, total_rank + c))
            for c_k in range(3):
                for c_i in range(2):
                    k = 2 + c_k
                    i = bi*2 + c_i
                    for c in range(12):
                        val = W_b[c_k*2 + c_i][c]
                        if val != 0:
                            W_full.setdefault((k,i), []).append((val, total_rank + c))
            total_rank += 12
    
    if time.time() > deadline:
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    U_mat = [[0]*total_rank for _ in range(n*m)]
    V_mat = [[0]*total_rank for _ in range(m*p)]
    W_mat = [[0]*total_rank for _ in range(p*n)]
    
    for (i,j), entries in U_full.items():
        for val, col in entries:
            U_mat[i*m + j][col] += val
    for (j,k), entries in V_full.items():
        for val, col in entries:
            V_mat[j*p + k][col] += val
    for (k,i), entries in W_full.items():
        for val, col in entries:
            W_mat[k*n + i][col] += val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
