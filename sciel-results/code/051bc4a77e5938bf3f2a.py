"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Actually the tensor T[i,j,k] = 1 for all i,j,k in matrix multiplication
    # Wait, no. Let me reconsider.
    
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all valid i,j,k
    # Decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_rank55(n, m, p):
    """Try a rank-55 construction for 2×5×5×6."""
    # For 2×m × m×p, try rank = 2*m*p - m
    rank = 2 * m * p - m  # 60 - 5 = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # For each j and k, create two terms but merge cleverly
    # Standard: for (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # To save m terms: for each j, the two rows U[0*m+j] and U[1*m+j] 
    # can share structure.
    
    # Construction: For j in range(m), k in range(p):
    #   term 1: U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
    #   term 2: U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1
    # That's 2*m*p = 60 terms.
    
    # Alternative: use the fact that we can write
    # For each j: define a combined term that handles both i=0 and i=1
    # This doesn't directly reduce rank in a simple way.
    
    # Let me just try the standard 60 and return it if 55 doesn't work.
    # Actually, let me just return None to indicate failure.
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # First try schoolbook (always works)
    result = _schoolbook(n, m, p)
    
    # Verify schoolbook
    assert _verify(result["U"], result["V"], result["W"], n, m, p)
    
    # Try to find better decomposition
    # For now, return schoolbook
    return result
# EVOLVE-BLOCK-END
