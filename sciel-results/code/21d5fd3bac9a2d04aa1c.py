"""Program for matrix multiplication tensor decomposition task."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank)
    - V has shape (m*p, rank) 
    - W has shape (p*n, rank)
    and the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    start_time = time.time()
    DEADLINE = 150  # seconds
    SAFETY_MARGIN = 5  # seconds before deadline
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    def check_time():
        return time.time() - start_time > DEADLINE - SAFETY_MARGIN
    
    # Key insight: The matrix multiplication tensor for (n,m,p) can be decomposed
    # using the Kronecker product structure. If we can find decompositions for
    # smaller dimensions, we can combine them.
    
    # For matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy 1: Use the fact that 3 = 2+1, 5 = 2+2+1, 7 = 4+2+1
    # We can try to construct the decomposition using blocks.
    
    # Let's try a structured approach: assume U, V, W have a block structure
    # where each block corresponds to a sub-problem.
    
    # For now, let's try to find a lower rank by using linear algebra.
    # The equations are:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 105 equations in U, V, W.
    # We want to find a solution with minimal rank r.
    
    # Let's try r from 104 down to see if we can find a solution.
    # But this is a non-convex problem, so we'll use a heuristic.
    
    # Heuristic: Start with schoolbook and try to merge terms.
    # Two terms (r1, r2) can be merged if there exist scalars a, b such that:
    # a*U[:,r1] + b*U[:,r2], a*V[:,r1] + b*V[:,r2], a*W[:,r1] + b*W[:,r2]
    # still satisfy the equations.
    
    # This is complex, so let's try a different approach: use known results.
    # For 2x2x2, Strassen gives rank 7.
    # For 3x3x3, the best known is 23.
    # For 4x4x4, the best known is 49.
    
    # For 3x5x7, we can try to embed in a larger problem.
    # But this is complex, so let's try a random search with local optimization.
    
    def random_search(rank, seed=42, max_iter=1000):
        """Try random initialization and local search."""
        if check_time():
            return None
        
        rng = np.random.RandomState(seed)
        # Use small integer or half-integer values
        values = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], dtype=float)
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(50):
            if check_time():
                break
            
            U_rand = rng.choice(values, size=(nm, rank))
            V_rand = rng.choice(values, size=(mp, rank))
            W_rand = rng.choice(values, size=(pn, rank))
            
            # Evaluate
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_rand[i*m+j, :] * V_rand[j*p+k, :] * W_rand[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                if max_err < 1e-8:
                    return best_U, best_V, best_W, max_err
        
        if best_U is not None:
            return best_U, best_V, best_W, best_err
        return None
    
    # Try ranks from 104 down to 90
    for rank in range(104, 89, -1):
        if check_time():
            break
        
        result = random_search(rank, seed=rank)
        if result is not None:
            U_cand, V_cand, W_cand, err = result
            if err < 1e-8:
                best_U, best_V, best_W = U_cand, V_cand, W_cand
                best_rank = rank
                break
    
    # Convert to lists as required by the interface
    def mat_to_list(mat):
        return mat.tolist()
    
    return {
        "U": mat_to_list(best_U),
        "V": mat_to_list(best_V),
        "W": mat_to_list(best_W)
    }
# EVOLVE-BLOCK-END
