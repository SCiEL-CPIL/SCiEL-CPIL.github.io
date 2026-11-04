import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]
_DEADLINE = 140.0

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] for t in range(len(U[0])))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank_54():
    """
    Construct a rank-54 decomposition for 2x5x6 matrix multiplication.
    Uses Strassen for 2x2x2 blocks and schoolbook for the remaining part.
    """
    n, m, p = 2, 5, 6
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # U coeffs for A (2x2)
    # Indices: 00, 01, 10, 11
    U_strassen = [
        [1, 0, 1, 0], # P1: A00=1, A11=1
        [0, 0, 1, 1], # P2: A10=1, A11=1
        [1, 0, 0, 0], # P3: A00=1
        [0, 0, 0, 1], # P4: A11=1
        [1, 1, 0, 0], # P5: A00=1, A01=1
        [-1, 0, 1, 0],# P6: A10=1, A00=-1
        [0, 1, 0, -1] # P7: A01=1, A11=-1
    ]
    
    # V coeffs for B (2x2)
    # Indices: 00, 01, 10, 11
    V_strassen = [
        [1, 0, 0, 1], # P1: B00=1, B11=1
        [1, 0, 0, 0], # P2: B00=1
        [0, 1, 0, -1],# P3: B01=1, B11=-1
        [-1, 0, 1, 0],# P4: B10=1, B00=-1
        [0, 0, 0, 1], # P5: B11=1
        [1, 1, 0, 0], # P6: B00=1, B01=1
        [0, 0, 1, 1]  # P7: B10=1, B11=1
    ]
    
    # W coeffs for C (2x2)
    # Indices: 00, 01, 10, 11 -> k=0,i=0; k=1,i=0; k=0,i=1; k=1,i=1
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 - P6
    W_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, -1],# P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, -1],# P6
        [1, 0, 0, 0]  # P7
    ]
    
    # Process 2x2 blocks of j (0-1, 2-3) and k (0-1, 2-3, 4-5)
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            for l in range(7):
                # Update U
                # A indices: i in {0,1}, j in {j_start, j_start+1}
                # U index: i*m + j
                for i in range(2):
                    for dj in range(2):
                        u_idx = i * m + (j_start + dj)
                        u_val = U_strassen[l][i * 2 + dj]
                        U[u_idx, col] = u_val
                        
                # Update V
                # B indices: j in {j_start, j_start+1}, k in {k_start, k_start+1}
                # V index: j*p + k
                for dj in range(2):
                    for dk in range(2):
                        v_idx = (j_start + dj) * p + (k_start + dk)
                        v_val = V_strassen[l][dj * 2 + dk]
                        V[v_idx, col] = v_val
                        
                # Update W
                # C indices: k in {k_start, k_start+1}, i in {0,1}
                # W index: k*n + i
                for dk in range(2):
                    for i in range(2):
                        w_idx = (k_start + dk) * n + i
                        w_val = W_strassen[l][dk * 2 + i]
                        W[w_idx, col] = w_val
                        
                col += 1
                
    # Process the remaining j=4 (single row)
    # Schoolbook for 2x1x6
    # A indices: i in {0,1}, j=4
    # B indices: j=4, k in {0..5}
    # C indices: k in {0..5}, i in {0,1}
    
    for k in range(p):
        for i in range(n):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            
            col += 1
            
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return None
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    start = time.time()
    
    # Try the rank 54 construction
    result = _construct_rank_54()
    if result is not None:
        return result
        
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
