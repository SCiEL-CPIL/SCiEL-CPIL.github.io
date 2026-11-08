"""Matrix multiplication 2x3 by 3x4: find rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find minimal rank decomposition
    # Start with schoolbook and try to reduce
    
    # The multiplication tensor: for each (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since every (i,j,k) triple contributes to the product)
    
    # Actually, the tensor equation is:
    # For the bilinear map (A,B) -> C where C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: the coefficient of A[i,j]*B[j,k] in C[i,k] is 1
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try ranks from low to high
    best_result = None
    
    for rank in range(10, 25):
        try:
            result = _find_decomposition(n, m, p, rank)
            if result is not None:
                return result
        except Exception:
            continue
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

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

def _find_decomposition(n, m, p, rank):
    """Try to find a decomposition of given rank using structured search."""
    # Use the fact that entries can be 0, 1/2, 1
    # Set up: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Try a structured approach: fix W to have simple entries
    # and solve for U, V
    
    # Alternative: use the Strassen-like approach or known decompositions
    # For now, try random search with verification
    
    import random
    random.seed(42)
    
    deadline = time.time() + 10  # 10 second budget per rank
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate random U, V, W with entries in {0, 0.5, 1}
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for r in range(rank):
            # Each column of U, V, W
            for idx in range(n*m):
                U[idx, r] = random.choice([0, 0.5, 1])
            for idx in range(m*p):
                V[idx, r] = random.choice([0, 0.5, 1])
            for idx in range(p*n):
                W[idx, r] = random.choice([0, 0.5, 1])
        
        # Verify
        if _verify(n, m, p, U, V, W):
            # Convert to list of lists with Fraction for exactness
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _verify(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-END
