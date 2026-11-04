import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    # Deadline guard: The task allows 120s. We use a generous margin.
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # First, build the trivial rank-24 decomposition as fallback
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((m * p, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((p * n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
    
    # Build the target tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We'll represent it as a 3D array
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Try to find a lower rank decomposition
    # Strategy: Randomized search with local refinement
    # Try ranks from 19 (known lower bound) up to 23
    
    best_result = {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
    best_rank = trivial_rank
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reconstructs T"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - T[i, j, k]) > tol:
                        return False
        return True
    
    def check_half_integer_constraint(U, V, W, tol=1e-10):
        """Check that all entries are integer or half-integer (multiples of 0.5)"""
        for matrix in [U, V, W]:
            for val in matrix.flatten():
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
        return True
    
    def random_decomposition_attempt(rank, num_trials=100):
        """Try to find a valid decomposition of given rank via random initialization + refinement"""
        np.random.seed(42 + rank)  # Reproducible but different per rank
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Simple gradient-free local search: try to reduce error
            # Error = sum over (i,j,k) of (computed - T[i,j,k])^2
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            err += (val - T[i, j, k]) ** 2
                return err
            
            # Try a few steps of coordinate descent
            best_err = compute_error(U, V, W)
            if best_err < 1e-10:
                return (U.copy(), V.copy(), W.copy())
            
            # Coordinate descent: try perturbing each element slightly
            for step in range(50):
                if time.time() > deadline:
                    return None
                
                improved = False
                for r in range(rank):
                    for i_row in range(n * m):
                        old_val = U[i_row, r]
                        for delta in [-0.5, 0.5]:
                            U[i_row, r] = old_val + delta
                            new_err = compute_error(U, V, W)
                            if new_err < best_err:
                                best_err = new_err
                                improved = True
                            else:
                                U[i_row, r] = old_val
                    
                    for j_row in range(m * p):
                        old_val = V[j_row, r]
                        for delta in [-0.5, 0.5]:
                            V[j_row, r] = old_val + delta
                            new_err = compute_error(U, V, W)
                            if new_err < best_err:
                                best_err = new_err
                                improved = True
                            else:
                                V[j_row, r] = old_val
                    
                    for k_row in range(p * n):
                        old_val = W[k_row, r]
                        for delta in [-0.5, 0.5]:
                            W[k_row, r] = old_val + delta
                            new_err = compute_error(U, V, W)
                            if new_err < best_err:
                                best_err = new_err
                                improved = True
                            else:
                                W[k_row, r] = old_val
                
                if not improved:
                    break
            
            # Check if we found an exact solution
            if best_err < 1e-10:
                # Verify it's valid
                if verify_decomposition(U, V, W):
                    return (U.copy(), V.copy(), W.copy())
        
        return None
    
    # Try ranks from 19 to 23
    for rank in range(19, trivial_rank):
        if time.time() > deadline - 5.0:
            break
        
        result = random_decomposition_attempt(rank, num_trials=20)
        
        if result is not None:
            U, V, W = result
            # Verify once more
            if verify_decomposition(U, V, W):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = rank
                break  # Found a valid lower-rank decomposition
    
    # If no lower rank found, return the trivial decomposition
    if best_rank == trivial_rank:
        return {
            "U": U_trivial.tolist(),
            "V": V_trivial.tolist(),
            "W": W_trivial.tolist()
        }
    
    return best_result
