import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Verify decomposition using NumPy einsum for speed.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Checks if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    # Convert to numpy arrays for fast contraction
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_res = U_np.reshape(n, m, rank)
    V_res = V_np.reshape(m, p, rank)
    W_res = W_np.reshape(p, n, rank)
    
    # Contract: S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note: W is indexed by k*n+i, so W_res[k, i, c]
    S = np.einsum('ijc,jkc,kic->ijk', U_res, V_res, W_res)
    
    # Check if S is all ones
    return np.allclose(S, 1.0, atol=1e-9)

def _search_low_rank(n, m, p, max_rank, deadline):
    """
    Search for a low-rank decomposition using random initialization and fast verification.
    """
    start_time = time.time()
    best_result = None
    best_rank = max_rank
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from max_rank - 1 down to a reasonable limit
    # We know schoolbook is max_rank.
    # Let's try a few ranks below max_rank.
    for target_rank in range(max_rank - 1, max(20, max_rank - 10), -1):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random initializations
        # Use small integer/half-integer values
        # Values: -1, -0.5, 0, 0.5, 1
        possible_vals = [-1, -0.5, 0, 0.5, 1]
        
        num_trials = 200 if target_rank > max_rank - 5 else 500
        
        for trial in range(num_trials):
            if time.time() > deadline - 2.0:
                break
            
            # Random initialization
            U = np.array([[random.choice(possible_vals) for _ in range(target_rank)] for _ in range(n * m)])
            V = np.array([[random.choice(possible_vals) for _ in range(target_rank)] for _ in range(m * p)])
            W = np.array([[random.choice(possible_vals) for _ in range(target_rank)] for _ in range(p * n)])
            
            # Verify
            if _verify_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                return best_result
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # 1. Build Schoolbook (Rank 60)
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # 2. Try to find a better decomposition
    # We search for ranks slightly lower than schoolbook
    # Given the difficulty, we try a limited search.
    if time.time() < deadline - 10.0:
        result = _search_low_rank(n, m, p, best_rank, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    # Ensure output format is correct (lists of lists of numbers)
    # The entries are already integers or halves from the search, or integers from schoolbook.
    return best_result
# EVOLVE-BLOCK-END
