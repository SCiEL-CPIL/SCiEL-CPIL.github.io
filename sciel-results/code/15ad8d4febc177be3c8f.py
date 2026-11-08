"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.
    
    The matrix multiplication tensor T[i,j,k] = 1 if output index corresponds to (i,j,k)
    where i is row of A (0..n-1), j is column of A / row of B (0..m-1), k is column of B (0..p-1).
    
    We seek U (n*m x r), V (m*p x r), W (p*n x r) such that:
    For all i, j, k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Strategy: Use the known rank-50 schoolbook decomposition and try to find a lower rank
    via random projection and verification. Also try structured low-rank decompositions.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Build the tensor T[i,j,k] = 1 for all valid (i,j,k)
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    # Schoolbook rank = n*m*p = 50
    schoolbook_rank = n * m * p
    
    # Generate schoolbook decomposition
    def schoolbook_decomp():
        r = schoolbook_rank
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, r
    
    # Verify a decomposition
    def verify(U, V, W, r):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    # Try to find a lower rank decomposition using random linear combinations
    # Idea: combine multiple schoolbook terms into one rank-1 term via linear combinations
    
    # The key insight: if we can find r rank-1 tensors whose sum gives T, we have rank r.
    # Each rank-1 term is u ⊗ v ⊗ w.
    
    # Try small ranks from a lower bound up
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Also rank >= max(n, m, p) * max(n, m, p) / max(n, m, p) ... various bounds
    # Strassen-like bounds: for 2x5x5, known best is around 40-45
    
    best_U, best_V, best_W, best_r = schoolbook_decomp()
    
    deadline = time.time() + 140  # Safety margin
    
    # Try to find lower rank by random search
    # Strategy: start with schoolbook and try to merge terms
    # Or: random initialization and gradient-free search
    
    # For each candidate rank r, try random initialization
    for r in range(25, best_r + 1):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with integer values
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            W = np.random.randint(-2, 3, size=(p*n, r))
            
            if verify(U, V, W, r):
                best_U, best_V, best_W, best_r = U, V, W, r
                break
        
        if best_r <= r:
            break
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
