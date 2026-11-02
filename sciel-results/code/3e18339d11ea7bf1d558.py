"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check if U, V, W form a valid decomposition."""
    n, m, p = DIMENSIONS
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Check entries are integers or half-integers
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if not isinstance(val, (int, float)):
                    return False
                # Check if integer or half-integer (multiple of 0.5)
                if abs(val * 2 - round(val * 2)) > 1e-10:
                    return False
    
    # Verify the tensor decomposition:
    # The matrix multiplication tensor T[i,j,k] = delta (where A[i,j]*B[j,k])
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a valid term, else 0
    # Actually, for matrix mult, the tensor is: T[i,j,k] = 1 for all i,j,k (it's the identity tensor for this product)
    # Wait, let me reconsider. The matrix multiplication tensor for (n,m,p) has entries:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # The product C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k (each term contributes 1)
    # 
    # The decomposition requires: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-10:
                    return False
    return True

def _random_decomposition(rank, max_attempts=100):
    """Try to find a random valid decomposition of given rank."""
    n, m, p = DIMENSIONS
    # Use values from {-1, -0.5, 0, 0.5, 1}
    possible_vals = [-1, -0.5, 0, 0.5, 1]
    
    for attempt in range(max_attempts):
        U = [[random.choice(possible_vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(possible_vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(possible_vals) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = len(best['U'][0])
    
    # Try to find lower rank decompositions
    # Start from rank just below schoolbook and go down
    for rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 10:  # Safety margin
            break
        
        # Try random search for this rank
        result = _random_decomposition(rank, max_attempts=50)
        if result is not None:
            best = result
            best_rank = rank
            # If we found one, try to go lower
            continue
    
    return best
# EVOLVE-BLOCK-END
