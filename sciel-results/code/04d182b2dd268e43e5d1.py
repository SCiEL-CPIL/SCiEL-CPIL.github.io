"""
Initial program for the suite task.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _build_standard(n, m, p):
    """Build the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # The decomposition should satisfy:
    # C[i,k] = sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a]
    # for all i,j,k, where the sum is over a, and j is implied by U's row index.
    # More precisely: for each (i, j, k), C[i,k] gets contribution from terms where
    # U row = i*m+j, V row = j*p+k, W row = k*n+i.
    # Let's verify by constructing the full tensor and comparing.
    
    # Build the multiplication tensor T[i, j, k] = 1 if this is a valid (i,j,k) triple
    # The decomposition gives: sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a]
    # This should equal 1 for all valid (i,j,k) and 0 otherwise (but all triples are valid).
    
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = 1 for all i,j,k in range
    # The decomposition: for each a, U[:,a] * V[:,a] * W[:,a] (outer product)
    # Sum_a should give T[i, j, k] = 1 for all i,j,k
    
    # Check: for each (i, j, k), sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] == 1
    # And for invalid combinations (which don't exist here since all are valid), it should be 0.
    
    # Let's spot check all triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s safety margin
    
    # Start with standard decomposition
    U, V, W, rank = _build_standard(n, m, p)
    best_rank = rank
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Verify the standard decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should not happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: try to express some columns as linear combinations of others
    # and eliminate them, reducing rank.
    
    # For now, try a simple approach: check if any column is a linear combination
    # of others by solving linear systems.
    
    # Build the full tensor check matrix
    # Each column a defines a rank-1 tensor: U[:,a] outer V[:,a] outer W[:,a]
    # We want to find the minimum number of such tensors that sum to the multiplication tensor.
    
    # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k.
    # In our indexing: T[i, j, k] corresponds to U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Let's try random projections to find lower-rank approximations
    # and then round to integers/half-integers.
    
    np.random.seed(42)
    
    # Try various target ranks
    for target_rank in range(best_rank - 1, 80, -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of rank target_rank
        # Use a random initialization and iterative refinement
        
        max_iter = 50
        for trial in range(3):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_try = np.random.randn(n * m, target_rank)
            V_try = np.random.randn(m * p, target_rank)
            W_try = np.random.randn(p * n, target_rank)
            
            # Iteratively refine: for each column, project to best fit
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # For each column a, given U[:,a] and W[:,a], find best V[:,a]
                # and vice versa.
                
                # Update V: for each a, V[:,a] should minimize ||T - sum_b U[:,b] outer V[:,b] outer W[:,b]||
                # This is complex. Use a simpler approach:
                # Fix U and W, solve for V.
                
                # Actually, let's use a different approach: 
                # The constraint is that for all (i,j,k):
                # sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] = 1
                
                # This is a system of nonlinear equations. Let's use a greedy approach.
                
                # Alternative: try to find if the standard decomposition has redundant columns.
                # Check if any column in {U, V, W} is a linear combination of others.
                
                break  # Skip complex search for now
            
            # Check if the random initialization (after some refinement) works
            # For now, just check the standard one
            
        # Try structured reduction: remove columns that are linearly dependent
        # Build a matrix where each row is a (i,j,k) triple and each column is a term
        # M[i,j,k, a] = U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a]
        # We want sum_a M[i,j,k,a] = 1 for all (i,j,k)
        
        if time.time() > deadline:
            break
    
    # Return the best result found
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
