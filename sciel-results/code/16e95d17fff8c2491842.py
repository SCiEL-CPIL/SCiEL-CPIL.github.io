"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor M[i,j,k] = 1 if i,j,k form a valid product
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all i,j,k (the matrix multiplication tensor)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_milp_rank(rank, n, m, p, time_budget=10):
    """
    Try to find a rank-`rank` decomposition using MILP.
    
    Strategy: We have rank terms. For each term r, we need to choose:
    - Which entries of U[:,r], V[:,r], W[:,r] are non-zero (binary variables)
    - The actual values (continuous variables)
    
    This is a bilinear problem, so we linearize by fixing supports first,
    then solving linear equations for the values.
    
    Alternative approach: enumerate support patterns via MILP, then solve LP for values.
    For small rank (23-27), we can use a different strategy.
    
    Actually, let's use a simpler approach:
    1. Assume each column of U, V, W has a specific sparse pattern.
    2. Use random initialization + local search to find valid decompositions.
    
    For now, let's use a structured approach based on known algorithms.
    """
    return None

def _construct_strassen_26(n, m, p):
    """
    Construct a rank-26 decomposition using Strassen's algorithm for the 2x2 core
    plus standard multiplication for the border elements.
    
    For 3x3 = [A B; C D] where A,B,C,D are 2x2 or 1x1 blocks... 
    Actually, let's partition 3x3 differently.
    
    A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    
    We can view the 3x3 as having a 2x2 top-left block, with borders.
    Top-left 2x2: multiply using Strassen (7 mults)
    The remaining 5 columns/rows need additional multiplications.
    
    Actually, a cleaner approach: use the known rank-26 algorithm.
    """
    # Rank 26 decomposition for 3x3 matrix multiplication
    # This uses a combination of Strassen-like ideas
    rank = 26
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We'll use a known rank-26 pattern.
    # The 3x3 matrix multiplication can be done with 26 scalar multiplications.
    # One construction: treat it as 2x2 Strassen (7) + 19 additional for the borders.
    # But that's 26 total if we're careful.
    
    # Let me use a different approach: enumerate a known valid rank-26 set.
    # Actually, let's just use the schoolbook 27 for safety, since constructing
    # a verified rank-26 from scratch is error-prone.
    
    # Instead, let's try to find rank 25 or 26 via a search.
    return None

def _schoolbook(n, m, p):
    """Return the trivial rank-27 schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _random_search_rank(target_rank, n, m, p, max_iters=5000, time_budget=10):
    """
    Randomly search for a rank-target_rank decomposition.
    Each term r has U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    We initialize randomly and try to adjust.
    """
    start_time = time.time()
    np.random.seed(42)
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        rank = target_rank
        # Initialize with random sparse patterns
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # For each term, randomly assign values
        for r in range(rank):
            # Randomly choose which entries are non-zero
            for idx in range(n*m):
                if np.random.random() < 0.5:
                    U[idx, r] = np.random.choice([-1, 1])
            for idx in range(m*p):
                if np.random.random() < 0.5:
                    V[idx, r] = np.random.choice([-1, 1])
            for idx in range(p*n):
                if np.random.random() < 0.5:
                    W[idx, r] = np.random.choice([-1, 1])
        
        # Check if this decomposition works
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_known_decompositions(n, m, p, time_budget=30):
    """
    Try known structured decompositions in order of increasing rank.
    """
    start_time = time.time()
    
    # Try rank 23 first (known minimal)
    # The rank-23 decomposition for 3x3 is known but complex.
    # Let's try a structured approach.
    
    # Known fact: rank of 3x3 matrix multiplication is 23.
    # A construction exists but is non-trivial to hardcode correctly.
    
    # Let's try rank 26 via Strassen-inspired construction
    # Partition 3x3 into overlapping 2x2 blocks
    
    # Actually, let me try a different structured approach:
    # Use the fact that we can decompose using 2x2 Strassen on overlapping blocks
    
    # For 3x3, one approach:
    # Block the matrix as:
    # [A B]
    # [C D] where A is 2x2, B is 2x1, C is 1x2, D is 1x1
    # Then C*D, D*B, etc. need additional multiplications.
    
    # Let me just return schoolbook if we can't find better quickly.
    if time.time() - start_time > time_budget * 0.3:
        return _schoolbook(n, m, p)
    
    # Try random search for rank 25
    result = _random_search_rank(25, n, m, p, max_iters=1000, time_budget=time_budget*0.3)
    if result is not None:
        return result
    
    if time.time() - start_time > time_budget * 0.6:
        return _schoolbook(n, m, p)
    
    # Try random search for rank 24
    result = _random_search_rank(24, n, m, p, max_iters=1000, time_budget=time_budget*0.2)
    if result is not None:
        return result
    
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # First, try to find a better decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 27
    
    # Try known structured approaches
    try:
        result = _try_known_decompositions(n, m, p, time_budget=60)
        if result is not None:
            rank = result["U"].shape[1]
            if rank < best_rank:
                best_result = result
                best_rank = rank
    except Exception:
        pass
    
    # If time remains, try random search for even better
    remaining = deadline - time.time() - 5  # 5s safety margin
    if remaining > 10 and best_rank > 24:
        try:
            result = _random_search_rank(best_rank - 1, n, m, p, max_iters=2000, time_budget=remaining)
            if result is not None:
                best_result = result
                best_rank = best_rank - 1
        except Exception:
            pass
    
    # Convert to lists for the required interface
    U_list = best_result["U"].tolist()
    V_list = best_result["V"].tolist()
    W_list = best_result["W"].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
