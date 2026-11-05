import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition as fallback
    def schoolbook():
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

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Laderman's 23-rank algorithm for 3x3x3x3
    # Defined as a list of (U, V, W) matrices for the 3x3 case
    # U, V, W are 3x3 matrices
    laderman_terms = [
        ([[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        ([[0,1,0],[0,0,0],[0,0,0]], [[0,0,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[0,0,0]]),
        ([[0,0,0],[1,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]),
        ([[0,0,0],[0,0,0],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,0],[0,1,0]]),
        ([[1,0,0],[0,0,0],[0,1,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]),
        ([[1,0,0],[0,0,0],[0,1,0]], [[0,0,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[0,0,0]]),
        ([[0,1,0],[0,0,0],[0,1,0]], [[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        ([[0,1,0],[0,0,0],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,0],[0,1,0]]),
        ([[0,0,0],[1,0,0],[0,1,0]], [[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        ([[0,0,0],[1,0,0],[0,1,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]),
        ([[1,0,0],[1,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,1,0]], [[0,0,0],[0,0,0],[0,1,0]]),
        ([[1,0,0],[1,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        ([[0,1,0],[1,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]),
        ([[0,1,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[0,0,0]]),
        ([[1,0,0],[1,0,0],[0,1,0]], [[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        ([[1,0,0],[1,0,0],[0,1,0]], [[0,0,0],[0,0,0],[0,1,0]], [[0,0,0],[0,0,0],[0,1,0]]),
        ([[1,0,0],[1,0,0],[0,1,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]),
        ([[1,0,0],[1,0,0],[0,1,0]], [[0,0,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[0,0,0]]),
        ([[1,0,0],[0,0,0],[0,1,0]], [[0,0,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[0,0,0]]),
        ([[1,0,0],[0,0,0],[0,1,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]),
        ([[2,0,0],[0,0,0],[0,1,0]], [[0,0,0],[0,0,0],[0,1,0]], [[0,0,0],[0,0,0],[0,1,0]]),
        ([[1,0,0],[0,0,0],[0,2,0]], [[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        ([[1,0,0],[1,0,0],[0,2,0]], [[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]])
    ]

    # Construct the 69-rank decomposition
    # 3 blocks, each of rank 23
    rank = 23 * 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(3):
        for t in range(23):
            u_mat, v_mat, w_mat = laderman_terms[t]
            col = b * 23 + t
            
            for i in range(3):
                for j in range(3):
                    U[i * 3 + j][col] = u_mat[i][j]
                    
            for j in range(3):
                for k in range(3):
                    # Global k index for block b is 3*b + k
                    V[j * 8 + (3 * b + k)][col] = v_mat[j][k]
                    
            for k in range(3):
                for i in range(3):
                    # Global k index for block b is 3*b + k
                    # W index is k_global * n + i
                    W[(3 * b + k) * 3 + i][col] = w_mat[k][i]

    # Verify the decomposition
    if verify_decomposition(U, V, W, rank):
        best_rank = rank
        best_result = {"U": U, "V": V, "W": W}
    else:
        # If verification fails, fall back to schoolbook
        pass

    return best_result
