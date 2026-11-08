"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 if i,j,k are valid indices (they always are for matmul)
    # Actually for matmul: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T[i,j,k] = delta (it's 1 for all i,j,k in the matmul tensor)
    # Wait, the matmul tensor has T[i,j,k] = 1 for all valid i,j,k
    # Let me verify: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline):
    """Try to find a rank-rank decomposition using structured search."""
    # For small ranks, use random restarts with local search
    best_result = None
    np.random.seed(42)
    
    # Start from schoolbook and try to merge terms
    # Or use random initialization with gradient-free optimization
    
    # For rank < n*m*p, we need a smarter approach
    # Use the fact that entries should be integers or half-integers
    
    # Try structured patterns: exploit that for 2x3x3, we might have symmetry
    # The tensor has n=2, m=3, p=3
    
    # Known result: rank of 2x3x3 matmul tensor
    # Let's try to find it by searching over small integer/half-integer values
    
    if rank >= n * m * p:
        # Schoolbook works
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # For lower rank, use random search with integer/half-integer entries
    # Values in {-1, -0.5, 0, 0.5, 1, 1.5, 2}
    possible_vals = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        if best_result is not None:
            break
        
        # Random initialization with small integers/half-integers
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c in range(rank):
            # Assign random values
            for i in range(n * m):
                U[i, c] = np.random.choice(possible_vals)
            for j in range(m * p):
                V[j, c] = np.random.choice(possible_vals)
            for k in range(p * n):
                W[k, c] = np.random.choice(possible_vals)
        
        # Verify
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        valid = False
                        break
        
        if valid:
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Try ranks from low to high
    for rank in range(1, n * m * p + 1):
        if time.time() > deadline:
            break
        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
