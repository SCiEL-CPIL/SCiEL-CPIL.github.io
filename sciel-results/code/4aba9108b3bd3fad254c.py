"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    # Also check that non-target entries are zero (orthogonality)
    # Actually for matmul tensor, T[i,j,k]=1 for all i,j,k, so we just need the dot products to be 1.
    # Wait, the tensor is T_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    # The decomposition is T_{i,j,k} = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    # So we need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k.
    return max_err < tol

def _check_all_entries(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for x in row:
                x2 = 2.0 * float(x)
                if abs(x2 - round(x2)) > 1e-9:
                    return False
    return True

def _build_strassen_2x2():
    """
    Build the rank-7 decomposition for 2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7).
    U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]
    
    # Strassen's 7 products:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # c11 = m1 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m6
    
    # Term 0: m1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c00 and c11 get +1
    
    # Term 1: m2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # c10 gets +1, c11 gets -1
    
    # Term 2: m3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # c01 gets +1
    
    # Term 3: m4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1  # c10 gets +1
    
    # Term 4: m5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # c00 gets -1, c01 gets +1
    
    # Term 5: m6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # c11 gets +1
    
    # Term 6: m7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # c00 gets +1
    
    return U, V, W

def _build_3x4x4_from_strassen():
    """
    Build a decomposition for 3x4x4 matmul using Strassen's 2x2 decomposition.
    
    Strategy: 
    - The 4x4 block (m x p) can be viewed as a 2x2 block matrix of 2x2 blocks.
    - For each of the 3 rows (i), we multiply a 1x4 row vector by a 4x4 matrix.
    - This is equivalent to multiplying a 1x4 vector by a 4x4 matrix.
    - We can group the 4 columns into 2 groups of 2.
    - For each group, we use Strassen's decomposition.
    
    Actually, a better approach:
    - Think of the 4x4 matrix B as a 2x2 block matrix where each block is 2x2.
    - The row vector A (1x4) can be split into two 1x2 vectors.
    - The product A*B gives a 1x4 vector, which is two 1x2 vectors.
    - For each output position, we need to compute the dot product.
    
    Let's use a different structure:
    - Split the 4 columns (m) into 2 groups of 2: m1, m2
    - Split the 4 rows (p) into 2 groups of 2: p1, p2
    - The 4x4 matrix B is [[B11, B12], [B21, B22]] where each Bij is 2x2.
    - The row vector A is [A1, A2] where A1, A2 are 1x2 vectors.
    - The result C is [C1, C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22.
    
    For each i (row of A, 3 total), we have:
    C_i,1 = A_i,1 * B_11 + A_i,2 * B_21  (1x2 result)
    C_i,2 = A_i,1 * B_12 + A_i,2 * B_22  (1x2 result)
    
    Each of these is a (1x2) * (2x2) = (1x2) multiplication, which is just a dot product.
    This doesn't directly use Strassen.
    
    Alternative: Use the fact that 4 = 2*2 and apply Strassen to the 2x2 block structure.
    But Strassen is for 2x2 * 2x2 = 2x2, not for 1x4 * 4x4.
    
    Let me try a different approach: embed the 3x4x4 problem into a larger problem
    and use known decompositions.
    
    Actually, the simplest improvement is to use the rank-7 2x2 decomposition
    for the 4x4 part by treating it as a 2x2 block matrix and using the
    block Strassen algorithm.
    
    For 4x4 matmul (m=4, p=4), the block Strassen algorithm uses 7 recursive
    calls to 2x2 matmul. If each 2x2 matmul has rank 7, then the total rank
    for 4x4 is 7*7 = 49. But we also have the n=3 dimension.
    
    Wait, the problem is 3x4x4, not 4x4x4. So we have 3 rows of A, each of
    length 4, multiplied by a 4x4 matrix B.
    
    For each row i of A, we compute C_i = A_i * B, which is a 1x4 vector.
    This is a (1x4) * (4x4) multiplication.
    
    We can write this as:
    C_i = [A_i,1  A_i,2] * [[B11, B12], [B21, B22]]
        = [A_i,1*B11 + A_i,2*B21, A_i,1*B12 + A_i,2*B22]
    
    Each term is a (1x2) * (2x2) = (1x2) product.
    
    A (1x2) * (2x2) product can be computed with rank 4 (schoolbook) or possibly
    less. But 1x2 * 2x2 is just two dot products of length 2, so rank 4 is natural.
    
    Total rank for 3x4x4 would be 3 * 4 * 2 = 24? No, that's not right.
    
    Let me think more carefully. The tensor is T[i,j,k] = 1 for i in [3], j in [4], k in [4].
    
    A rank-r decomposition gives:
    T[i,j,k] = sum_{t=1}^r U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    For the schoolbook, r = 3*4*4 = 48.
    
    To reduce the rank, we need to find a structure that allows sharing.
    
    One idea: use the fact that 4 = 2+2 and apply a block decomposition.
    Let's split j into j1 in [2] and j2 in [2], and k into k1 in [2] and k2 in [2].
    
    T[i, j1, k1] = sum_t U[i,j1,t] V[j1,k1,t] W[k1,i,t]
    T[i, j1, k2] = sum_t U[i,j1,t] V[j1,k2,t] W[k2,i,t]
    T[i, j2, k1] = sum_t U[i,j2,t] V[j2,k1,t] W[k1,i,t]
    T[i, j2, k2] = sum_t U[i,j2,t] V[j2,k2,t] W[k2,i,t]
    
    This doesn't immediately help unless we can factor the dependence.
    
    Let's try a random search with a better strategy.
    """
    n, m, p = 3, 4, 4
    # Try to find a rank < 48 decomposition
    best_rank = n * m * p
    best_decomp = None
    
    # Schoolbook as fallback
    U_school = [[0]*best_rank for _ in range(n*m)]
    V_school = [[0]*best_rank for _ in range(m*p)]
    W_school = [[0]*best_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    return U_school, V_school, W_school

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try random search for lower rank
    values = [-1, -0.5, 0, 0.5, 1]
    
    for target_rank in range(36, 48):
        if time.time() > deadline - 10:
            break
        rng = random.Random(target_rank * 1000 + 42)
        found = False
        max_iters = 5
        for it in range(max_iters):
            U_try = [[rng.choice(values) for _ in range(target_rank)] for _ in range(n*m)]
            V_try = [[rng.choice(values) for _ in range(target_rank)] for _ in range(m*p)]
            W_try = [[rng.choice(values) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U_try, V_try, W_try, n, m, p):
                if _check_all_entries(U_try, V_try, W_try):
                    U, V, W = U_try, V_try, W_try
                    found = True
                    break
        if found:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
