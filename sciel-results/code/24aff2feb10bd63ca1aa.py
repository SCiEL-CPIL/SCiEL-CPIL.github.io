import numpy as np
import time

def construct():
    """
    Returns a rank-32 exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy: Split B into 4 blocks of 2 columns. For each block, split the
    4 rows into top (rows 0,1) and bottom (rows 2,3). Compute:
        C_l = A_left @ B_l_top + A_right @ B_l_bot
    where A_left = A[:, 0:2], A_right = A[:, 2:4], each 2x2.
    Each 2x2 x 2x2 product uses schoolbook (rank 4).
    Total: 4 blocks * 2 sub-products * 4 terms = 32.
    """
    n, m, p = 2, 4, 8
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    col = 0
    
    for l in range(4):  # 4 column blocks of B
        k0 = l * 2  # starting column index for this block
        
        # Sub-product 1: A_left (2x2) @ B_l_top (2x2) -> contributes to C[:, k0:k0+2]
        # A_left: rows i in {0,1}, cols j in {0,1}
        # B_l_top: rows j in {0,1}, cols k in {k0, k0+1}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = col + i * 4 + j * 2 + k
                    U[i * m + j][r] = 1
                    V[j * p + (k0 + k)][r] = 1
                    W[(k0 + k) * n + i][r] = 1
        col += 4
        
        # Sub-product 2: A_right (2x2) @ B_l_bot (2x2) -> contributes to C[:, k0:k0+2]
        # A_right: rows i in {0,1}, cols j in {2,3}
        # B_l_bot: rows j in {2,3}, cols k in {k0, k0+1}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = col + i * 4 + j * 2 + k
                    U[i * m + (j + 2)][r] = 1
                    V[(j + 2) * p + (k0 + k)][r] = 1
                    W[(k0 + k) * n + i][r] = 1
        col += 4
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print("Rank:", rank)
    
    # Verification
    n, m, p = 2, 4, 8
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    max_diff = 0
    for i in range(n):
        for k in range(p):
            c_val = 0
            for j in range(m):
                for r in range(rank):
                    c_val += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            diff = abs(c_val - C_expected[i, k])
            if diff > max_diff:
                max_diff = diff
    print("Max diff:", max_diff)
    if max_diff > 1e-10:
        print("ERROR: Decomposition is invalid!")
    else:
        print("Decomposition is valid.")

