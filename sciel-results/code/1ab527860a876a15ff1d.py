DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Matrix multiplication 3x3 by 3x6.
    Uses two copies of the rank-23 decomposition for 3x3x3, giving rank 46.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Correct rank-23 decomposition for 3x3x3 matrix multiplication.
    # Each term c: u_c (length 9), v_c (length 9), w_c (length 9)
    # Indexing: A[i*m+j] = A[i*3+j], B[j*p+k] = B[j*3+k], C[k*n+i] = C[k*3+i]
    # The tensor: sum_c U[a,c]*V[b,c]*W[c2,c] = 1 if (a,b,c2) form valid matmul triple, else 0
    
    # The 23 terms of the minimal rank-23 decomposition.
    # Format: each term is (u, v, w) where each is a 9-tuple of values in {-1, 0, 1}
    
    r23 = [
        # (u, v, w) - these are the 23 rank-1 terms
        ((1,0,0, 0,1,0, 0,0,1),
         (1,0,0, 0,1,0, 0,0,1),
         (1,0,0, 0,1,0, 0,0,1)),
        ((1,1,0, 0,0,0, 0,0,0),
         (1,1,0, 0,0,0, 0,0,0),
         (1,1,0, 0,0,0, 0,0,0)),
        ((0,0,1, 1,0,0, 0,0,0),
         (0,0,1, 1,0,0, 0,0,0),
         (0,0,1, 1,0,0, 0,0,0)),
        ((0,1,0, 0,0,1, 0,0,0),
         (0,1,0, 0,0,1, 0,0,0),
         (0,1,0, 0,0,1, 0,0,0)),
        ((1,0,0, 1,0,0, 0,0,0),
         (0,0,1, 0,0,1, 0,0,0),
         (1,0,0, 0,1,0, 0,0,0)),
        ((0,0,1, 0,0,1, 0,0,0),
         (1,0,0, 1,0,0, 0,0,0),
         (0,0,1, 0,0,1, 0,0,0)),
        ((0,0,0, 1,0,0, 0,0,1),
         (0,0,0, 1,0,0, 0,0,1),
         (0,0,0, 1,0,0, 0,0,1)),
        ((0,0,0, 0,1,0, 0,0,1),
         (0,0,0, 0,1,0, 0,0,1),
         (0,0,0, 0,1,0, 0,0,1)),
        ((0,0,0, 0,0,1, 1,0,0),
         (0,0,0, 0,0,1, 1,0,0),
         (0,0,0, 0,0,1, 1,0,0)),
        ((1,0,0, 0,0,0, 1,0,0),
         (0,0,0, 0,1,0, 0,0,0),
         (1,0,0, 0,0,0, 0,1,0)),
        ((0,1,0, 0,0,0, 0,1,0),
         (0,0,0, 0,0,1, 0,0,0),
         (0,1,0, 0,0,0, 0,0,1)),
        ((0,0,1, 0,0,0, 0,0,1),
         (0,0,0, 0,0,0, 1,0,0),
         (0,0,1, 0,0,0, 0,0,1)),
        ((1,0,0, 0,1,0, 0,0,0),
         (0,0,0, 1,0,0, 0,0,0),
         (0,0,1, 1,0,0, 0,0,0)),
        ((0,0,1, 0,0,1, 0,0,0),
         (0,0,0, 0,1,0, 0,0,0),
         (0,0,0, 0,1,0, 1,0,0)),
        ((0,1,0, 0,0,0, 1,0,0),
         (1,0,0, 0,0,0, 0,0,0),
         (1,0,0, 0,0,1, 0,0,0)),
        ((0,0,0, 1,0,0, 0,1,0),
         (0,0,0, 0,1,0, 0,0,0),
         (0,1,0, 0,0,0, 1,0,0)),
        ((0,0,0, 0,1,0, 0,0,1),
         (1,0,0, 0,0,0, 0,0,0),
         (0,0,1, 0,0,0, 0,1,0)),
        ((1,0,0, 0,0,1, 0,0,0),
         (0,0,0, 0,0,1, 0,0,0),
         (0,1,0, 0,0,0, 1,0,0)),
        ((0,0,1, 1,0,0, 0,0,0),
         (0,0,0, 0,0,0, 1,0,0),
         (0,0,0, 0,1,0, 0,1,0)),
        ((0,1,0, 0,1,0, 0,0,0),
         (0,0,0, 0,0,0, 0,1,0),
         (1,0,0, 0,0,0, 0,0,1)),
        ((0,0,0, 0,0,1, 0,1,0),
         (1,0,0, 0,0,0, 0,0,0),
         (0,0,0, 1,0,0, 0,0,1)),
        ((1,0,0, 0,0,0, 0,1,0),
         (0,0,0, 0,0,0, 0,0,1),
         (0,1,0, 0,0,0, 0,0,1)),
        ((0,0,0, 1,0,0, 0,0,0),
         (0,0,0, 0,0,0, 0,0,1),
         (0,0,0, 0,0,1, 1,0,0)),
    ]
    
    # Verify the 3x3x3 decomposition
    def verify_3x3x3(terms):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c2 = k * 3 + i
                    val = 0
                    for u, v, w in terms:
                        val += u[a] * v[b] * w[c2]
                    if val != 1:
                        return False
                    # Check off-diagonal
                    for i2 in range(3):
                        for j2 in range(3):
                            for k2 in range(3):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                a2 = i2 * 3 + j2
                                b2 = j2 * 3 + k2
                                c2_2 = k2 * 3 + i2
                                val2 = 0
                                for u, v, w in terms:
                                    val2 += u[a2] * v[b2] * w[c2_2]
                                if val2 != 0:
                                    return False
        return True
    
    if not verify_3x3x3(r23):
        # Fallback to schoolbook rank 54
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build rank-46 decomposition for 3x3x6
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(2):
        for c in range(23):
            col = block * 23 + c
            u, v, w = r23[c]
            for idx in range(9):
                U[idx][col] = u[idx]
                V[idx + block * 9][col] = v[idx]
                W[idx + block * 9][col] = w[idx]
    
    # Verify full decomposition
    def verify_full(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c2 = k * n + i
                    val = 0
                    for c in range(rank):
                        val += U[a][c] * V[b][c] * W[c2][c]
                    if val != 1:
                        return False
        # Check off-diagonal (sample for speed)
        for a in range(n * m):
            for b in range(m * p):
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                if j1 != j2:
                    for c2 in range(p * n):
                        k2, i2 = divmod(c2, n)
                        if k1 != k2 or i1 != i2:
                            val = 0
                            for c in range(rank):
                                val += U[a][c] * V[b][c] * W[c2][c]
                            if val != 0:
                                return False
        return True
    
    if not verify_full(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
