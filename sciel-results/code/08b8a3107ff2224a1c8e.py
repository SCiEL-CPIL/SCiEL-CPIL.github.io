"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Must equal 1 if i,j,k correspond to valid indices (always valid here), else 0.
    Actually for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor T[i,j,k] should be 1 (the coefficient in the product).
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, time_limit=10):
    """Try to find a rank-r decomposition using random initialization and gradient-free search.
    This is a very rough approach; for this small problem we might use algebraic methods.
    """
    n, m, p = DIMENSIONS
    # For now, just return None if we can't find one quickly
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook()
    
    # Try to find better decompositions
    # Known lower bounds: For 4x4 by 4x7, the rank is at least max(n*m, m*p, n*p) = max(16, 28, 28) = 28
    # Upper bound from schoolbook: 4*4*7 = 112
    
    # Try some known constructions
    # For matrix multiplication, there are algorithms like Strassen's that can reduce rank
    # But for specific dimensions, we need to check if a rank-r decomposition exists
    
    # Let's try a simple approach: use the fact that we can sometimes factor the multiplication
    # using block structure
    
    # For 4x4 by 4x7:
    # A is 4x4, B is 4x7
    # We can think of this as multiplying two matrices where one is square and the other is rectangular
    
    # One approach: use the standard algorithm but look for linear dependencies
    # The tensor has n*m*p = 112 entries, and we want to express it as sum of r rank-1 tensors
    
    # Let's try rank 28 (the lower bound) - this is unlikely but worth checking
    # Actually, for matrix multiplication, the rank is known to be at least the maximum of the three dimensions' products
    # For 4x4*4x7, lower bound is max(4*4, 4*7, 4*7) = 28
    
    # Try to construct a rank-28 decomposition if possible
    # This would be optimal if achievable
    
    # For now, let's just return the schoolbook decomposition
    # A more sophisticated approach would use numerical optimization or algebraic geometry
    
    return result
# EVOLVE-BLOCK-END
