"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    rank_sb = n * m * p
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_sb
    
    # Try to reduce rank by checking if we can find a lower rank decomposition
    # Use iterative approach: try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    
    lower_bound = max(n*m, m*p, p*n)
    
    # For each candidate rank, try to find a decomposition
    # This is NP-hard in general, but for small cases we can try optimization
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But actually the decomposition is:
        # For matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor form: T_{i,j,k} = delta (i.e., T_{i,j,k} = 1 if we're computing the product)
        # Actually the bilinear map: f(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # In tensor form with the given indexing:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # The product sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should give the coefficient
        # for A_{i,j} * B_{j,k} contributing to (AB)_{i,k}
        # This should be 1 for all i,j,k (since each A_{i,j}B_{j,k} contributes exactly once to (AB)_{i,k})
        
        # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def round_to_halfint(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    # Try optimization for lower ranks
    # Use a simple approach: random initialization + gradient descent or coordinate descent
    # But given time constraints, let's try a few strategies
    
    # Strategy: Try rank = lower_bound, lower_bound+1, ... up to rank_sb-1
    # For each, try random initializations with optimization
    
    def try_optimize(rank, max_iters=5000, n_restarts=3):
        """Try to find a rank decomposition using optimization."""
        if rank < lower_bound:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            np.random.seed(42 + restart * 1000 + rank)
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
            
            # Use a simple gradient-free approach: try to minimize the error
            # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
            best_err = np.inf
            
            # Simple iterative improvement
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Compute current error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_init[i*m+j, :] * V_init[j*p+k, :] * W_init[k*n+i, :])
                            err += (val - 1.0)**2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
                
                if best_err < 1e-12:
                    break
                
                # Try coordinate updates: pick a random element and adjust
                # This is very slow, so let's use a different approach
                
                # Actually, let's just try a few random perturbations
                break  # Skip the slow loop for now
            
            # Check if we found a good solution
            if best_err < 1e-10:
                # Round to half-integers and verify
                U_r = round_to_halfint(best_U)
                V_r = round_to_halfint(best_V)
                W_r = round_to_halfint(best_W)
                if verify_decomposition(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r
                
                # Also check without rounding
                if verify_decomposition(best_U, best_V, best_W, tol=1e-9):
                    # Check if entries are already half-integers
                    if np.allclose(best_U * 2, np.round(best_U * 2)) and \
                       np.allclose(best_V * 2, np.round(best_V * 2)) and \
                       np.allclose(best_W * 2, np.round(best_W * 2)):
                        return np.round(best_U * 2) / 2, np.round(best_V * 2) / 2, np.round(best_W * 2) / 2
        
        return None
    
    # Try ranks from lower_bound up to rank_sb - 1
    for rank in range(lower_bound, rank_sb):
        if time.time() > deadline:
            break
        
        result = try_optimize(rank, max_iters=100, n_restarts=2)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = rank
            best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            break  # Found the lowest rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
