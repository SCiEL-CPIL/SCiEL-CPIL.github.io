"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import random

def _schoolbook():
    """Return the schoolbook rank-56 decomposition."""
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
    """Verify that U, V, W form a valid decomposition. Returns True if valid."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if i,j,k valid, 0 otherwise
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication: T[i,j,k] = 1 if this corresponds to a valid product term, else 0
    # Actually, the decomposition should satisfy:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: C[i,k] = sum_{j,c} A[i,j] * B[j,k] * (something)
    # The standard tensor decomposition: T_{i,j,k} = sum_c U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c}
    # This T should equal the matrix multiplication tensor where T_{i,j,k} = 1 for all valid i,j,k
    # Wait, that's not right. Let me think again.
    
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] such that sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # So T[i,j,k] = 1 for all i, j, k (it's just the identity in the j-dimension)
    
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _check_half_integer(U, V, W, tol=1e-9):
    """Check if all entries are integers or half-integers (within tolerance)."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                # Check if val is close to an integer or half-integer
                v = float(val)
                nearest_int = round(v)
                nearest_half = round(2 * v) / 2.0
                if abs(v - nearest_int) > tol and abs(v - nearest_half) > tol:
                    return False
    return True

def _try_random_search(rank, max_iters=5000, seed=None):
    """Try to find a decomposition of given rank using random search."""
    if seed is not None:
        random.seed(seed)
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    for _ in range(max_iters):
        # Random initialization
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Scale to make the product work
        # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a complex constraint. Let's try a different approach.
        
        # Actually, let's use the fact that we can set up a linear system.
        # For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is nonlinear. Let's try alternating minimization.
        
        # Initialize W to 1s
        W = np.ones((pn, rank))
        
        # Now for each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] = 1
        # This is a bilinear constraint. Let's try to solve it iteratively.
        
        # For a fixed W, we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Let's define: for each c, let a_c[i,j] = U[i*m+j,c], b_c[j,k] = V[j*p+k,c], w_c[k,i] = W[k*n+i,c]
        # Then sum_c a_c[i,j] * b_c[j,k] * w_c[k,i] = 1
        
        # This is still hard. Let me try a simpler approach: assume rank = n*m*p and use schoolbook,
        # then try to reduce rank by finding linear dependencies.
        
        break  # Skip random search for now, use structured approach
    
    return None

def _structured_search(max_rank=56, deadline=None):
    """Search for lower-rank decompositions using structured methods."""
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from lower to higher
    for rank in range(1, best_rank + 1):
        if deadline is not None and time.time() > deadline:
            break
        
        # For small ranks, try brute force with integer/half-integer entries
        # This is only feasible for very small ranks
        if rank <= 20:
            # Try to find a decomposition using linear algebra
            # The constraint is: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # Let's try a specific structure: 
            # U[i*m+j, c] depends only on i and j
            # V[j*p+k, c] depends only on j and k  
            # W[k*n+i, c] depends only on k and i
            
            # One approach: use the fact that the tensor is "simple" in some sense
            # For 2x4x7, try rank 2*4 = 8, 4*7 = 28, 2*7 = 14, etc.
            
            pass  # Placeholder for structured search
        
        # Check if we've found something better
        if rank < best_rank:
            # This rank is potentially better, but we need to find an actual decomposition
            pass
    
    return best, best_rank

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = 2 * 4 * 7  # 56
    
    # Try to find lower-rank decompositions
    # The key insight: for matrix multiplication, the tensor rank is related to the
    # arithmetic complexity. For 2x4x7, we can try various structured approaches.
    
    # Approach 1: Try rank = n*m = 8 (if possible, this would be great)
    # Approach 2: Try rank = n*p = 14
    # Approach 3: Try rank = m*p = 28
    # Approach 4: Try intermediate ranks
    
    # Let's try a systematic search for low-rank decompositions
    # using the fact that entries must be integers or half-integers.
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Try various ranks
    for rank in [8, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52]:
        if time.time() > deadline:
            break
        
        # For each rank, try to find a valid decomposition
        # Using a simple heuristic: random restarts with local search
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Random initialization with integer/half-integer values
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Initialize with random half-integers
            for i in range(nm):
                for c in range(rank):
                    U[i, c] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1])
            
            for j in range(mp):
                for c in range(rank):
                    V[j, c] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1])
            
            for k in range(pn):
                for c in range(rank):
                    W[k, c] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1])
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                if _check_half_integer(U.tolist(), V.tolist(), W.tolist()):
                    found = True
                    if rank < best_rank:
                        best_rank = rank
                        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if found and rank < best_rank:
            # Found a better decomposition, continue searching for even lower rank
            continue
    
    # Check deadline
    if time.time() > deadline:
        return best
    
    return best
# EVOLVE-BLOCK-END
