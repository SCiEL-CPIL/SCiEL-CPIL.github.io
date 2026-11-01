import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # Rank 56 construction using Strassen on 2x2 blocks
    # 8 blocks of 2x2 multiplication
    # Each block has rank 7
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Indices for A (2x4): i in 0..1, j in 0..3
    # Indices for B (4x8): j in 0..3, k in 0..7
    # Indices for C (2x8): i in 0..1, k in 0..7
    # W index: k*n + i -> k*2 + i
    
    # u: block index for middle dimension (0, 1) -> rows 2u, 2u+1 of B, cols 2u, 2u+1 of A
    # v: block index for last dimension (0, 1, 2, 3) -> cols 2v, 2v+1 of B, cols 2v, 2v+1 of C
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A block indices
            # A[0, 2u], A[0, 2u+1], A[1, 2u], A[1, 2u+1]
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            
            # B block indices
            # B[2u, 2v], B[2u, 2v+1], B[2u+1, 2v], B[2u+1, 2v+1]
            b11_idx = 2 * u * 8 + 2 * v
            b12_idx = 2 * u * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            
            # C block indices (W indices)
            # C[0, 2v], C[0, 2v+1], C[1, 2v], C[1, 2v+1]
            # W index = k * n + i = k * 2 + i
            c11_idx = 2 * v * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = 2 * v * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # s=0: M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # s=1: M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # s=2: M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # s=3: M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # s=4: M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # s=5: M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # s=6: M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
            
    return {"U": U, "V": V, "W": W}
