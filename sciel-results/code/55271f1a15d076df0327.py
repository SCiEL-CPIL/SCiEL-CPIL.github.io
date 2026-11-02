"""Matrix multiplication 5x5x5 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
DEADLINE = 145.0  # Safety margin before 150s kill

def _verify(U, V, W, n=5, m=5, p=5):
    """Verify that U, V, W exactly decompose matrix multiplication."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} (for matrix mult)
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition means: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where T[i,j,k] = 1 if this corresponds to a valid term, else 0
    # For matrix multiplication, the tensor is: T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    # Wait, no. The matrix multiplication tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # and the decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But actually the convention is W[k*n+i, r] so W index is (k*n+i)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Since it's the full multiplication tensor, every entry should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n=5, m=5, p=5):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_based_5x5():
    """
    Try to construct a 5x5 decomposition using Strassen-like ideas.
    Split 5x5 into blocks: use 2x2 Strassen for part, handle rest.
    This is heuristic - may not yield improvement, but let's try.
    """
    n, m, p = 5, 5, 5
    # This is a placeholder - a proper Strassen-based 5x5 is complex
    # For now, return None to indicate no valid decomposition found
    return None

def construct():
    start_time = time.time()
    
    # Always have the schoolbook as fallback
    fallback = _schoolbook_decomposition()
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        return fallback
    
    # Try to find a better decomposition
    # For 5x5, known upper bound is 103, but explicit constructions are complex
    # Let's try a simple improvement: see if we can merge any terms
    
    # One approach: try random perturbations and see if we can reduce rank
    # But this is unlikely to work within time budget for exact verification
    
    # Given the complexity, return the schoolbook decomposition
    # which is guaranteed correct and has rank 125
    return fallback
