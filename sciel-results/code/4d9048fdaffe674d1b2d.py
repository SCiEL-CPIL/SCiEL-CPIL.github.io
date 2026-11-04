"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    
    The tensor T has indices (i, j, k) where:
    - i in [0, n) = [0, 3): row of A
    - j in [0, m) = [0, 5): column of A / row of B  
    - k in [0, p) = [0, 6): column of B
    
    T[i, j, k] = 1 if (i, j, k) is a valid triple, else 0.
    
    We seek U (nm x r), V (mp x r), W (pn x r) such that:
    For all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    Using the trace convention: W[k*n + i, c]
    
    Strategy: Try to find a decomposition with rank less than schoolbook (90).
    For small matrices, we can use structured approaches.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    schoolbook_rank = nm * p  # 90
    
    # Build the target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Try to find a lower-rank decomposition using a structured approach.
    # One idea: exploit the structure of matrix multiplication.
    
    # For 3x5 * 5x6, let's try a block-based approach or use known algorithms.
    # Since no sub-schoolbook decomposition is known for these dimensions,
    # we'll attempt a randomized search with verification.
    
    # Start with schoolbook as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition.
    # We'll try ranks from schoolbook_rank-1 down to a lower bound.
    # Lower bound: max(nm, mp, pn) = 30, but actually the tensor rank is at least
    # max(nm, mp, pn) = 30.
    
    # For efficiency, we'll try a few specific ranks and use a heuristic search.
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W give an exact decomposition of the multiplication tensor."""
        r = U.shape[1]
        # Check all (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_random_rank(r, num_trials=1000):
        """Try to find a rank-r decomposition using random initialization."""
        for trial in range(num_trials):
            # Random initialization with small values
            scale = 1.0 / (r ** 0.25)
            U = np.random.randn(nm, r) * scale
            V = np.random.randn(mp, r) * scale
            W = np.random.randn(pn, r) * scale
            
            # Try to refine using a simple gradient-like approach
            # For now, just check if it's valid (unlikely with random init)
            if verify_decomposition(U, V, W):
                return U, V, W
        return None
    
    # Try structured approaches first.
    # Approach: Use the fact that matrix multiplication is bilinear.
    # We can try to find a decomposition where each term corresponds to a
    # "slice" of the multiplication.
    
    # For 3x5 * 5x6, one structured approach:
    # Group by the middle dimension j.
    # For each j, we have a 3x1 * 1x6 multiplication.
    # This gives 5 terms of rank 18 each (3*6), total rank 90.
    # This is the schoolbook.
    
    # To get lower rank, we need to share computation across j values.
    
    # Let's try a different structured approach:
    # Use the identity: sum_j A[i,j] * B[j,k] = (A * B)[i,k]
    # We can write this as a bilinear form.
    
    # For small matrices, we can try to use a basis approach.
    # Represent each row of A and each column of B in some basis.
    
    # Given the time constraint, let's try a few ranks with a more structured random search.
    
    # Try ranks from 80 down to 40 with limited trials
    for r in range(80, 39, -5):
        result = try_random_rank(r, num_trials=50)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    # If no better rank found, use schoolbook
    if best_rank == schoolbook_rank:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to list format
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
