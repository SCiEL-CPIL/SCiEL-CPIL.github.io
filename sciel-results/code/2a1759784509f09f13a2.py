"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Build the multiplication tensor T[i,j,k] = 1 if output (i,k) depends on input (i,j) and (j,k)
    # The tensor is defined by: T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    # We need to find rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with schoolbook solution
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions using structured search
    # The tensor has structure: for each (i,k), the "slice" T[i,*,k] is a 3x6 matrix with 1s in row i
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But we can do better by exploiting structure
    
    # Try a few candidate ranks
    for candidate_rank in [18, 19, 20, 21, 22, 24, 27, 30, 33, 36]:
        if time.time() > deadline - 10:
            break
        
        if candidate_rank >= best_rank:
            break
        
        # Try to find a valid decomposition at this rank
        success, U_cand, V_cand, W_cand = _try_find_decomposition(n, m, p, candidate_rank, deadline)
        
        if success:
            best_rank = candidate_rank
            best_U, best_V, best_W = U_cand, V_cand, W_cand
            break  # Found minimum, stop searching
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _try_find_decomposition(n, m, p, rank, deadline):
    """Try to find a valid rank-r decomposition using structured search."""
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is equivalent to: for each (i,j,k), the sum over c must equal 1
    
    # Use a simpler approach: try random initialization with refinement
    np.random.seed(42)
    
    # Initialize with random values in {-1, 0, 1}
    for attempt in range(50):
        if time.time() > deadline - 5:
            return False, None, None, None
        
        # Try different initialization strategies
        if attempt < 10:
            # Strategy 1: Random {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, rank))
            V = np.random.randint(-1, 2, size=(m*p, rank))
            W = np.random.randint(-1, 2, size=(p*n, rank))
        elif attempt < 20:
            # Strategy 2: Random {0, 1}
            U = np.random.randint(0, 2, size=(n*m, rank))
            V = np.random.randint(0, 2, size=(m*p, rank))
            W = np.random.randint(0, 2, size=(p*n, rank))
        else:
            # Strategy 3: Sparse random
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for c in range(rank):
                # Each column has a few non-zero entries
                for _ in range(2):
                    U[np.random.randint(n*m), c] = np.random.choice([-1, 1])
                for _ in range(2):
                    V[np.random.randint(m*p), c] = np.random.choice([-1, 1])
                for _ in range(2):
                    W[np.random.randint(p*n), c] = np.random.choice([-1, 1])
        
        # Check if this decomposition is valid
        if _check_decomposition(n, m, p, rank, U, V, W):
            # Convert to list of lists with integer values
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Verify all values are integers or half-integers
            valid = True
            for mat in [U_list, V_list, W_list]:
                for row in mat:
                    for val in row:
                        if not (val * 2 == int(val * 2)):  # Check if val is integer or half-integer
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return True, U_list, V_list, W_list
    
    return False, None, None, None

def _check_decomposition(n, m, p, rank, U, V, W):
    """Check if the decomposition is valid: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
