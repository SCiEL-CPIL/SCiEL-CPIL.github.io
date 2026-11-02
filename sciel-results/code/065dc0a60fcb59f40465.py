"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 145.0  # 5s safety margin
    
    # Build the target multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if (a,b,c) corresponds to a valid (i,j,k) triple, else 0
    # a = i*m + j, b = j*p + k, c = k*n + i
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                valid_indices.append((a, b, c))
    
    def verify(U, V, W):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] == T_target."""
        try:
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(T_comp, T_target, atol=1e-9)
        except Exception:
            return False
    
    # Trivial schoolbook decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial), dtype=np.float64)
    V_trivial = np.zeros((m * p, rank_trivial), dtype=np.float64)
    W_trivial = np.zeros((p * n, rank_trivial), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1.0
                V_trivial[b, col] = 1.0
                W_trivial[c, col] = 1.0
                col += 1
    
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = rank_trivial
    
    # Strategy: Start from a low-rank Tucker-like approximation and try to find
    # an exact integer/half-integer decomposition via structured search.
    # 
    # The multiplication tensor for (n,m,p) has rank at most n*m*p.
    # Known lower bounds: rank >= max(n*m, m*p, p*n) for some cases,
    # and rank >= n*m + m*p + p*n - 2 (for certain dimensions).
    # For 3x5x7: lower bound ~ 15+35+21-2 = 69.
    # 
    # We'll try a structured approach:
    # 1. Use the fact that the tensor is sparse (only 105 non-zero entries out of 15*35*21=11025)
    # 2. Try to find a decomposition where each rank-1 term covers multiple non-zero entries
    
    # Approach: Try to find a decomposition of rank R by solving a system of equations
    # For each valid (a,b,c), we need sum_r U[a,r] V[b,r] W[c,r] = 1
    # For each invalid (a,b,c), we need sum_r U[a,r] V[b,r] W[c,r] = 0
    
    # This is a nonlinear system. We'll use a greedy/structured approach.
    
    # Try a structured decomposition based on grouping
    # Group by i: for each i, we need to decompose the (m x p) sub-problem
    # C_i[k, j] = sum_r ... 
    
    # Alternative: Try known small-rank decompositions for small matrices
    # For 2x2x2, rank is 7 (Strassen)
    # For 3x3x3, rank is 23 (Landsberg)
    
    # For 3x5x7, let's try a block-based approach
    # Split the 5x7 matrix into blocks and use Strassen-like techniques
    
    # Let's try a direct search for low-rank solutions with integer entries
    # Start with a reasonable rank estimate and try to verify
    
    # First, let's try to use the fact that we can combine terms
    # If we have two rank-1 terms that can be merged, we reduce the rank
    
    # Try a structured decomposition:
    # For each i (3 values), decompose the m x p multiplication
    # The m x p = 5 x 7 multiplication can be done with rank <= 5*7 = 35
    # But we can do better with Strassen-like methods
    
    # Let's try: for each i, use a rank-35 decomposition of the 5x7 product
    # This gives total rank = 3 * 35 = 105 (same as schoolbook)
    
    # To do better, we need to share terms across different i values
    
    # Try a different approach: use the tensor's structure
    # The tensor T has the property that T[a,b,c] = 1 iff a mod m == b div p and b mod p == c div n and c mod n == a div m
    
    # Let's try a random search with integer entries for lower ranks
    np.random.seed(42)
    
    # Try ranks from a lower bound up to the trivial rank
    # Lower bound: max(n*m, m*p, p*n) = 35
    # But realistically, we expect rank > 35 for 3x5x7
    
    # Try a few candidate ranks
    candidate_ranks = []
    for r in range(50, min(105, 80)):  # Try ranks 50 to 79
        candidate_ranks.append(r)
    
    # For each candidate rank, try to find a solution
    # Use a simple iterative approach:
    # 1. Initialize U, V, W randomly with small integers
    # 2. Check if the decomposition is valid
    # 3. If not, adjust entries to reduce error
    
    # This is a hard combinatorial problem, so we'll use a heuristic approach
    
    # Let's try a more systematic approach:
    # The tensor can be written as a sum of rank-1 terms
    # Each rank-1 term is a tensor of the form u * v * w (outer product)
    
    # For the multiplication tensor, a natural rank-1 term corresponds to
    # a specific (i,j,k) triple, but we can have more general rank-1 terms
    
    # Let's try to find a decomposition where each rank-1 term is sparse
    # and covers multiple non-zero entries of T
    
    # Given the complexity, let's try a simpler approach:
    # Use the schoolbook decomposition but try to merge some terms
    
    # For now, return the schoolbook decomposition as the best known solution
    # The search for a lower-rank decomposition is complex and may not succeed
    # within the time limit
    
    # Try a few random initializations for lower ranks
    for target_rank in [80, 75, 70, 65, 60, 55]:
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-1, 2, (n * m, target_rank))
            V = np.random.randint(-1, 2, (m * p, target_rank))
            W = np.random.randint(-1, 2, (p * n, target_rank))
            
            if verify(U, V, W):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break
        
        if best_rank < rank_trivial:
            break
    
    # If we found a better solution, return it; otherwise return schoolbook
    if best_rank < rank_trivial:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_trivial.tolist(),
            "V": V_trivial.tolist(),
            "W": W_trivial.tolist()
        }

# EVOLVE-BLOCK-END
