"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time
import random
import sys

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for Strassen 2x2x2 decomposition, each shape (4, 7)."""
    # U[i*2+j, c] for i,j in 0..1, c in 0..6
    # V[j*2+k, c] for j,k in 0..1, c in 0..6
    # W[k*2+i, c] for k,i in 0..1, c in 0..6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[0*2+1, 0] = -1; W[1*2+0, 0] = 1; W[1*2+1, 0] = -1
    
    # M2 = (A21+A22)B11
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1; W[1*2+1, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+1, 2] = 1; W[1*2+1, 2] = 1
    
    # M4 = A22(B21-B11)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[1*2+0, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+1, 6] = 1
    
    return U, V, W

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1, r2 = U1.shape[1], U2.shape[1]
    U = np.zeros((U1.shape[0]*U2.shape[0], r1*r2))
    V = np.zeros((V1.shape[0]*V2.shape[0], r1*r2))
    W = np.zeros((W1.shape[0]*W2.shape[0], r1*r2))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1*r2 + c2
            U[:, c] = np.kron(U1[:, c1], U2[:, c2])
            V[:, c] = np.kron(V1[:, c1], V2[:, c2])
            W[:, c] = np.kron(W1[:, c1], W2[:, c2])
    
    return U, V, W

def _is_valid(U, V, W, r, n, m, p):
    """Check if the decomposition is exact for matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(r):
                    lhs += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(lhs - 1.0) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook decomposition as baseline
    r_school = n * m * p
    U_sb = np.zeros((n * m, r_school))
    V_sb = np.zeros((m * p, r_school))
    W_sb = np.zeros((p * n, r_school))
    
    indices = list(product(range(n), range(m), range(p)))
    for c, (i, j, k) in enumerate(indices):
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0
    
    best_result = (U_sb, V_sb, W_sb, r_school)
    
    # Kronecker product approach for 4x5x5
    # Split: 4x5x5 = 4x4x4 + 4x5x1 + 4x1x4
    # 4x4x4 via Strassen-Kronecker: rank 49
    # 4x5x1: rank 4 (covers i in 0..3, j in 0..4, k=4)
    # 4x1x4: rank 4 (covers i in 0..3, j=4, k in 0..3)
    # Total: 57
    
    if time.time() - start_time < TIME_LIMIT - 20:
        # Get Strassen 2x2x2
        U_s, V_s, W_s = _strassen_2x2x2()
        
        # Kronecker product for 4x4x4
        U_444, V_444, W_444 = _kron_decomp(U_s, V_s, W_s, U_s, V_s, W_s)
        
        # Now build the full 4x5x5 decomposition
        r_total = 49 + 4 + 4  # 57
        U_full = np.zeros((n*m, r_total))
        V_full = np.zeros((m*p, r_total))
        W_full = np.zeros((p*n, r_total))
        
        # Part 1: 4x4x4 (i in 0..3, j in 0..3, k in 0..3)
        for i in range(4):
            for j in range(4):
                U_full[i*5+j, 0:49] = U_444[i*4+j, 0:49]
        
        for j in range(4):
            for k in range(4):
                V_full[j*5+k, 0:49] = V_444[j*4+k, 0:49]
        
        for k in range(4):
            for i in range(4):
                W_full[k*4+i, 0:49] = W_444[k*4+i, 0:49]
        
        # Part 2: 4x5x1 (i in 0..3, j in 0..4, k=4)
        # For term i (i=0..3):
        # U_full[i*5+j, 49+i] = 1 for all j=0..4
        # V_full[j*5+4, 49+i] = 1 for all j=0..4
        # W_full[4*4+i, 49+i] = 1
        for i in range(4):
            for j in range(5):
                U_full[i*5+j, 49+i] = 1.0
            for j in range(5):
                V_full[j*5+4, 49+i] = 1.0
            W_full[4*4+i, 49+i] = 1.0
        
        # Part 3: 4x1x4 (i in 0..3, j=4, k in 0..3)
        # For term k (k=0..3):
        # U_full[i*5+4, 53+k] = 1 for all i=0..3
        # V_full[4*5+k, 53+k] = 1
        # W_full[k*4+i, 53+k] = 1 for all i=0..3
        for k in range(4):
            for i in range(4):
                U_full[i*5+4, 53+k] = 1.0
            V_full[4*5+k, 53+k] = 1.0
            for i in range(4):
                W_full[k*4+i, 53+k] = 1.0
        
        # Validate
        if _is_valid(U_full, V_full, W_full, r_total, n, m, p):
            best_result = (U_full, V_full, W_full, r_total)
    
    U_best, V_best, W_best, r_best = best_result
    return {"U": U_best[:, :r_best].tolist(), "V": V_best[:, :r_best].tolist(), "W": W_best[:, :r_best].tolist()}
# EVOLVE-BLOCK-END
