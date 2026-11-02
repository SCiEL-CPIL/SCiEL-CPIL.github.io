"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each entry C[i,k], we need:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] over all j
    # Actually, the tensor decomposition means:
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T_{i,j,k} = 1 if this is the multiplication tensor
    # The multiplication tensor for A*B has T_{i,j,k} = 1 for all valid (i,j,k)
    # Wait, no. The matrix multiplication tensor T is defined such that:
    # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # As a tensor, T_{i,j,k} = delta (it's 1 for the "multiplication" operation)
    # Actually the tensor is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the structure is
    # T_{i,j,k} = 1 if we're computing the (i,k) entry using A_{i,j} and B_{j,k}
    
    # The correct check: reconstruct the multiplication tensor
    # T_{i,j,k} should equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because the multiplication tensor has a 1 at each (i,j,k) position
    
    # Actually no. The matrix multiplication tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # This is the "all ones" tensor of shape (n, m, p)
    
    # A rank-r decomposition means:
    # T_{i,j,k} = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify this
    reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                reconstructed[i, j, k] = val
    
    # The target tensor is all ones
    target = np.ones((n, m, p))
    return np.allclose(reconstructed, target, atol=tol)

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return np.allclose(2*x, np.round(2*x), atol=tol)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: try ranks from best_rank-1 downward, use random search with half-integer constraint
    
    # For efficiency, try a few lower ranks with limited iterations
    for rank in range(best_rank - 1, max(best_rank - 10, 0), -1):
        if time.time() > deadline:
            break
        
        max_iters = 500
        found = False
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization with half-integer values
            # Use small set of half-integers: {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
            vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                found = True
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
        
        if found:
            continue  # Try even lower rank
    
    return best
# EVOLVE-BLOCK-END
