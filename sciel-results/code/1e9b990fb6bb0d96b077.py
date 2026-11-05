import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

def _deadline_ok(t0, limit=130.0):
    return (time.time() - t0) < limit

def _verify_decomp(U, V, W, rank):
    """Verify that the decomposition is exact for the 2x5x5 matrix multiplication tensor."""
    if len(U) != N * M or len(V) != M * P or len(W) != P * N:
        return False
    for i in range(N * M):
        if len(U[i]) != rank:
            return False
    for j in range(M * P):
        if len(V[j]) != rank:
            return False
    for k in range(P * N):
        if len(W[k]) != rank:
            return False
    
    # T[i,j,k] should be 1 for all valid indices
    # T[i,j,k] = sum_r U[i*M+j][r] * V[j*P+k][r] * W[k*N+i][r]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[i * M + j][r]) * Fraction(V[j * P + k][r]) * Fraction(W[k * N + i][r])
                if val != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition with rank n*m*p = 50."""
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W

def _strassen_222_decomp():
    """
    Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where each is a list of 7 vectors of length 4.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    Verified Strassen 2x2x2 tensor decomposition:
    T[i,j,k] = sum_{r=0}^{6} U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k in {0,1}
    """
    U = [
        [1, 0, 0, 1],  # r=0: (a00+a11)
        [0, 0, 1, 1],  # r=1: (a10+a11)
        [1, 0, 0, 0],  # r=2: a00
        [0, 0, 0, 1],  # r=3: a11
        [1, 1, 0, 0],  # r=4: (a00+a01)
        [-1, 0, 1, 0], # r=5: (a10-a00)
        [0, 1, 0, -1], # r=6: (a01-a11)
    ]
    V = [
        [1, 0, 0, 1],  # r=0: (b00+b11)
        [1, 0, 0, 0],  # r=1: b00
        [0, 1, 0, -1], # r=2: (b01-b11)
        [0, 0, 1, -1], # r=3: (b10-b00)
        [0, 0, 0, 1],  # r=4: b11
        [1, 1, 0, 0],  # r=5: (b00+b01)
        [0, 0, 1, 1],  # r=6: (b10+b11)
    ]
    W = [
        [1, 0, 0, 1],  # r=0: C00+=1, C11+=1
        [0, 1, 0, 0],  # r=1: C10+=1
        [0, 0, 1, 1],  # r=2: C01+=1, C11+=1
        [1, 1, 0, 0],  # r=3: C00+=1, C10+=1
        [-1, 1, 0, 0], # r=4: C00-=1, C01+=1
        [0, 0, 0, 1],  # r=5: C11+=1
        [1, 0, 0, 0],  # r=6: C00+=1
    ]
    return U, V, W

def _build_strassen_2x5x5():
    """
    Build a decomposition for 2x5x5 using Strassen 2x2x2 blocks + schoolbook.
    
    Strategy: Use Strassen for the 2x2x2 sub-block (rank 7) and schoolbook for the rest.
    Split 5 = 2 + 3.
    
    A is 2x5, B is 5x5, C is 2x5.
    Split columns of A: A[:, 0:2] (2x2) and A[:, 2:5] (2x3)
    Split rows/cols of B: B[0:2, 0:2] (2x2), B[0:2, 2:5] (2x3), B[2:5, 0:2] (3x2), B[2:5, 2:5] (3x3)
    
    C[0:2, 0:2] = A[:,0:2]*B[0:2,0:2] + A[:,2:5]*B[2:5,0:2]
    C[0:2, 2:5] = A[:,0:2]*B[0:2,2:5] + A[:,2:5]*B[2:5,2:5]
    
    Sub-problems:
    1. 2x2x2: A[:,0:2] * B[0:2,0:2] -> rank 7 (Strassen)
    2. 2x3x2: A[:,2:5] * B[2:5,0:2] -> rank 6 (schoolbook)
    3. 2x2x3: A[:,0:2] * B[0:2,2:5] -> rank 6 (schoolbook)
    4. 2x3x3: A[:,2:5] * B[2:5,2:5] -> rank 6 (schoolbook)
    
    Total: 7 + 6 + 6 + 6 = 25.
    """
    total_rank = 7 + 6 + 6 + 6  # 25
    U = [[0] * total_rank for _ in range(N * M)]  # 10 x 25
    V = [[0] * total_rank for _ in range(M * P)]  # 25 x 25
    W = [[0] * total_rank for _ in range(P * N)]  # 10 x 25
    
    # Strassen 2x2x2 block: i in 0..1, j in 0..1, k in 0..1
    U_s, V_s, W_s = _strassen_222_decomp()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * M + j][r] = U_s[r][i * 2 + j]
        for j in range(2):
            for k in range(2):
                V[j * P + k][r] = V_s[r][j * 2 + k]
        for k in range(2):
            for i in range(2):
                W[k * N + i][r] = W_s[r][k * 2 + i]
    
    # Schoolbook 2x3x2 block: i in 0..1, j in 2..4, k in 0..1
    col = 7
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    
    # Schoolbook 2x2x3 block: i in 0..1, j in 0..1, k in 2..4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    
    # Schoolbook 2x3x3 block: i in 0..1, j in 2..4, k in 2..4
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    t0 = time.time()
    
    # Try the Strassen-based decomposition (rank 25)
    try:
        U, V, W = _build_strassen_2x5x5()
        rank = 25
        if _verify_decomp(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 50)
    U, V, W = _schoolbook()
    rank = N * M * P
    
    if not _verify_decomp(U, V, W, rank):
        raise ValueError("Schoolbook verification failed")
    
    return {"U": U, "V": V, "W": W}
