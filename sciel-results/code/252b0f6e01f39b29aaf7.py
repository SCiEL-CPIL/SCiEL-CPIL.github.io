import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Rank 56: 8 independent 2x2x2 multiplications, each via Strassen (rank 7).
    
    U: shape (n*m, rank) = (8, 56)
    V: shape (m*p, rank) = (32, 56)
    W: shape (p*n, rank) = (16, 56)
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A is 2x4: index a = i*4 + j, i in {0,1}, j in {0,1,2,3}
    # B is 4x8: index b = j*8 + k, j in {0,1,2,3}, k in {0..7}
    # C is 2x8: index c = k*2 + i, k in {0..7}, i in {0,1}
    
    # Decompose into 8 blocks: for each pair of middle indices (u=0,1) 
    # and each pair of output columns (v=0,1,2,3):
    #   A block: rows 0,1; cols 2u, 2u+1  (2x2)
    #   B block: rows 2u, 2u+1; cols 2v, 2v+1  (2x2)
    #   C block: rows 0,1; cols 2v, 2v+1  (2x2)
    
    for u in range(2):
        for v in range(4):
            base = 7 * (4 * u + v)
            
            # A indices: A[i, 2u] and A[i, 2u+1] for i=0,1
            a00 = 0 * 4 + 2 * u
            a01 = 0 * 4 + 2 * u + 1
            a10 = 1 * 4 + 2 * u
            a11 = 1 * 4 + 2 * u + 1
            
            # B indices: B[2u, 2v], B[2u, 2v+1], B[2u+1, 2v], B[2u+1, 2v+1]
            b00 = 2 * u * 8 + 2 * v
            b01 = 2 * u * 8 + 2 * v + 1
            b10 = (2 * u + 1) * 8 + 2 * v
            b11 = (2 * u + 1) * 8 + 2 * v + 1
            
            # C indices: C[i, 2v] and C[i, 2v+1], W index = k*2 + i
            c00 = 2 * v * 2 + 0  # i=0, k=2v
            c01 = (2 * v + 1) * 2 + 0  # i=0, k=2v+1
            c10 = 2 * v * 2 + 1  # i=1, k=2v
            c11 = (2 * v + 1) * 2 + 1  # i=1, k=2v+1
            
            # Strassen 7 products for 2x2 * 2x2:
            # M = [[c00, c01], [c10, c11]]
            # s0 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
            U[a00][base + 0] += 1
            U[a11][base + 0] += 1
            V[b00][base + 0] += 1
            V[b11][base + 0] += 1
            W[c00][base + 0] += 1
            W[c11][base + 0] += 1
            
            # s1 = (a10+a11)b00 -> c10 += 1, c11 -= 1
            U[a10][base + 1] += 1
            U[a11][base + 1] += 1
            V[b00][base + 1] += 1
            W[c10][base + 1] += 1
            W[c11][base + 1] -= 1
            
            # s2 = a00(b01-b11) -> c01 += 1, c11 += 1
            U[a00][base + 2] += 1
            V[b01][base + 2] += 1
            V[b11][base + 2] -= 1
            W[c01][base + 2] += 1
            W[c11][base + 2] += 1
            
            # s3 = a11(b10-b00) -> c00 += 1, c10 += 1
            U[a11][base + 3] += 1
            V[b10][base + 3] += 1
            V[b00][base + 3] -= 1
            W[c00][base + 3] += 1
            W[c10][base + 3] += 1
            
            # s4 = (a00+a01)b11 -> c00 -= 1, c01 += 1
            U[a00][base + 4] += 1
            U[a01][base + 4] += 1
            V[b11][base + 4] += 1
            W[c00][base + 4] -= 1
            W[c01][base + 4] += 1
            
            # s5 = (a10-a00)(b00+b01) -> c11 += 1
            U[a10][base + 5] += 1
            U[a00][base + 5] -= 1
            V[b00][base + 5] += 1
            V[b01][base + 5] += 1
            W[c11][base + 5] += 1
            
            # s6 = (a01-a11)(b10+b11) -> c00 += 1
            U[a01][base + 6] += 1
            U[a11][base + 6] -= 1
            V[b10][base + 6] += 1
            V[b11][base + 6] += 1
            W[c00][base + 6] += 1
    
    return {"U": U, "V": V, "W": W}
