import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 3, 7
    
    # Try to find a lower-rank decomposition
    # Strategy: use known algebraic improvements
    
    # The schoolbook gives rank 63. 
    # Let's try to construct a valid decomposition with rank < 63.
    
    # One approach: if we can find any valid decomposition, we return it.
    # Let's try a randomized approach to find a lower rank decomposition.
    
    # Actually, for this specific problem, let me try to use the fact that
    # M_{3,3,7} can be decomposed using bilinear algorithms.
    
    # Let me try to find a 60-rank decomposition by merging some terms.
    
    # Initialize with schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to find linear dependencies to reduce rank
    # Check if columns are linearly dependent
    def try_reduce_rank(U, V, W, max_iterations=50):
        """Try to find linear dependencies to reduce rank."""
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            # Try to find a linear combination of columns that is zero
            # This means the columns are linearly dependent
            
            rank_current = best_U.shape[1]
            if rank_current <= 1:
                break
            
            # Pick a random subset of columns to check for dependencies
            # Use a simple approach: try to express one column as a combo of others
            
            # Check if any column is a linear combination of the first r-1 columns
            for target in range(1, rank_current):
                if time.time() > deadline - 5:
                    break
                    
                # Check if column 'target' is in span of columns [0, target-1]
                A = best_U[:, :target]
                b = best_U[:, target]
                
                try:
                    # Solve for coefficients
                    coeffs, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    if residuals and residuals[0] < 1e-10:
                        # Check if same coefficients work for V and W
                        V_pred = best_V[:, :target] @ coeffs
                        W_pred = best_W[:, :target] @ coeffs
                        if np.allclose(best_V[:, target], V_pred, atol=1e-10) and \
                           np.allclose(best_W[:, target], W_pred, atol=1e-10):
                            # We can eliminate column 'target'
                            # Remove column 'target' from all matrices
                            mask = [i for i in range(rank_current) if i != target]
                            best_U = best_U[:, mask]
                            best_V = best_V[:, mask]
                            best_W = best_W[:, mask]
                            best_rank = best_rank - 1
                            break
                except:
                    continue
        
        return best_U, best_V, best_W, best_rank
    
    # Try to reduce rank
    U, V, W, reduced_rank = try_reduce_rank(U, V, W)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
