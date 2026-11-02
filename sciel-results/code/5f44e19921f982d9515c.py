import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 10]

def _verify_decomposition(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Build tensor from U, V, W: T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
    # Wait, the convention is W[k*n+i, r] where k is row of B (which is col index in our indexing?)
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank), indexed by (k, i) where k is column of B (0..p-1) and i is row of A (0..n-1)
    # Actually for C = A*B where A is n x m and B is m x p:
    # C[i, l] = sum_j A[i,j] * B[j,l]
    # The tensor is T[i, j, j', l] = delta_{j,j'} so that C[i,l] = sum_{j,j'} A[i,j] B[j',l] T[i,j,j',l]
    # But that's not quite right either. Let me think again.
    
    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p) is:
    # T[i, j, k, l] where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # T[i,j,k,l] = 1 if j == k else 0
    # Then C[i,l] = sum_{j,k} A[i,j] * B[k,l] * T[i,j,k,l] = sum_j A[i,j] * B[j,l]
    
    # The rank-r decomposition: T[i,j,k,l] = sum_{r=0}^{R-1} U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
    # Wait, the problem says W has shape (p*n, rank) with convention W[k*n+i, r]
    # So W is indexed by (l, i) where l is column of B (0..p-1) and i is row of A (0..n-1)
    
    # Let me verify: for each (i, j, k, l), we need sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r] == (1 if j==k else 0)
    
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    expected = 1 if j == k else 0
                    actual = sum(U[i*m + j][r] * V[k*p + l][r] * W[l*n + i][r] for r in range(rank))
                    if abs(actual - expected) > 1e-9:
                        return False
    return True

def _construct_rank35():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    The rank-35 decomposition is based on 5 independent Strassen 2x2x2 blocks.
    U shape: (4, 35)
    V shape: (20, 35)
    W shape: (20, 35)
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    strassen_A = [
        [1, 0, 0, 1],  # P1: a+d
        [0, 0, 1, 1],  # P2: c+d
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: a+b
        [-1, 0, 1, 0], # P6: c-a
        [0, 1, 0, -1]  # P7: b-d
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1: e+h
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: f-h
        [-1, 0, 1, 0], # P4: g-e
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: e+f
        [0, 0, 1, 1]   # P7: g+h
    ]
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    
    w_contrib[5][3] = 1
    
    w_contrib[6][0] = 1
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            k0 = 2 * b
            k1 = 2 * b + 1
            
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
            
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    """Fallback: rank-40 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 40
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    if j == k:
                        U[i * m + j][idx] = 1.0
                        V[k * p + l][idx] = 1.0
                        W[l * n + i][idx] = 1.0
                        idx += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Uses Strassen-based decomposition (rank 35) with verification.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline with 20s safety margin
    
    # Try rank-35 Strassen decomposition
    result = _construct_rank35()
    
    if time.time() > deadline:
        return result
    
    # Verify the decomposition
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return _construct_schoolbook()
