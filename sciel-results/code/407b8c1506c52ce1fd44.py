import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 130
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try rank 42 decomposition
    # 14 multiplications per 3x3 block
    rank_14 = 14
    total_rank = 3 * rank_14
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define the 14 multiplications for 2x3x3
    # u: [a11, a12, a13, a21, a22, a23]
    # v: [b11, b12, b13, b21, b22, b23, b31, b32, b33]
    # w: [c11, c12, c13, c21, c22, c23]
    
    mults = [
        # 1
        ([1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0], [1/4, 0, 0, 1/4, 0, 0]),
        # 2
        ([1, 0, 0, -1, 0, 0], [1, 0, 0, -1, 0, 0, 0, 0, 0], [1/4, 0, 0, -1/4, 0, 0]),
        # 3
        ([1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [1/4, 0, 0, 1/4, 0, 0]),
        # 4
        ([1, 0, 0, 0, -1, 0], [1, 0, 0, 0, -1, 0, 0, 0, 0], [1/4, 0, 0, -1/4, 0, 0]),
        # 5
        ([0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1/2, 0, 0, 0, 0, 0]),
        # 6
        ([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [1/2, 0, 0, 0, 0, 0]),
        # 7
        ([0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1/2, 0, 0]),
        # 8
        ([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1/2, 0, 0]),
        # 9
        ([0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1/2, 0, 0, 0, 0]),
        # 10
        ([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1/2, 0, 0, 0]),
        # 11
        ([0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1/2, 0]),
        # 12
        ([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1/2, 0]),
        # 13
        ([0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1/4, 0, 0, 1/4, 0]),
        # 14
        ([0, 1, 0, 0, -1, 0], [0, 1, 0, 0, -1, 0, 0, 0, 0], [0, 1/4, 0, 0, -1/4, 0]),
    ]
    
    for block in range(3):
        for c_idx in range(14):
            col = block * 14 + c_idx
            u, v, w = mults[c_idx]
            # u: [a11, a12, a13, a21, a22, a23]
            # U indices: i*m+j
            # a11: i=0, j=0 -> 0
            # a12: i=0, j=1 -> 1
            # a13: i=0, j=2 -> 2
            # a21: i=1, j=0 -> 3
            # a22: i=1, j=1 -> 4
            # a23: i=1, j=2 -> 5
            U[0][col] += u[0]
            U[1][col] += u[1]
            U[2][col] += u[2]
            U[3][col] += u[3]
            U[4][col] += u[4]
            U[5][col] += u[5]
            
            # v: [b11, b12, b13, b21, b22, b23, b31, b32, b33]
            # V indices: j*p+k
            # b11: j=0, k=0 -> 0
            # b12: j=0, k=1 -> 1
            # b13: j=0, k=2 -> 2
            # b21: j=1, k=0 -> 9
            # b22: j=1, k=1 -> 10
            # b23: j=1, k=2 -> 11
            # b31: j=2, k=0 -> 18
            # b32: j=2, k=1 -> 19
            # b33: j=2, k=2 -> 20
            # Wait, the block index shifts the k index.
            # For block 0, k is 0,1,2.
            # For block 1, k is 3,4,5.
            # For block 2, k is 6,7,8.
            # So v[0] corresponds to b1, (3*block+0)
            # v[1] corresponds to b1, (3*block+1)
            # v[2] corresponds to b1, (3*block+2)
            # v[3] corresponds to b2, (3*block+0)
            # v[4] corresponds to b2, (3*block+1)
            # v[5] corresponds to b2, (3*block+2)
            # v[6] corresponds to b3, (3*block+0)
            # v[7] corresponds to b3, (3*block+1)
            # v[8] corresponds to b3, (3*block+2)
            
            V[0 * p + (3*block + 0)][col] += v[0]
            V[0 * p + (3*block + 1)][col] += v[1]
            V[0 * p + (3*block + 2)][col] += v[2]
            V[1 * p + (3*block + 0)][col] += v[3]
            V[1 * p + (3*block + 1)][col] += v[4]
            V[1 * p + (3*block + 2)][col] += v[5]
            V[2 * p + (3*block + 0)][col] += v[6]
            V[2 * p + (3*block + 1)][col] += v[7]
            V[2 * p + (3*block + 2)][col] += v[8]
            
            # w: [c11, c12, c13, c21, c22, c23]
            # W indices: k*n+i
            # c11: k=3*block+0, i=0 -> (3*block+0)*2+0
            # c12: k=3*block+1, i=0 -> (3*block+1)*2+0
            # c13: k=3*block+2, i=0 -> (3*block+2)*2+0
            # c21: k=3*block+0, i=1 -> (3*block+0)*2+1
            # c22: k=3*block+1, i=1 -> (3*block+1)*2+1
            # c23: k=3*block+2, i=1 -> (3*block+2)*2+1
            
            W[(3*block + 0) * n + 0][col] += w[0]
            W[(3*block + 1) * n + 0][col] += w[1]
            W[(3*block + 2) * n + 0][col] += w[2]
            W[(3*block + 0) * n + 1][col] += w[3]
            W[(3*block + 1) * n + 1][col] += w[4]
            W[(3*block + 2) * n + 1][col] += w[5]
            
    if _verify_decomposition(U, V, W, n, m, p):
        best = {"U": U, "V": V, "W": W}
        best_rank = total_rank

    return best
