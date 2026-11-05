import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x5 matrix. Starts with schoolbook (rank 45) and attempts to find lower rank.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = nm * p  # 45
    
    # Build schoolbook decomposition
    U = np.zeros((nm, schoolbook_rank))
    V = np.zeros((mp, schoolbook_rank))
    W = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 100  # Leave margin for 120s limit
    
    def verify(U, V, W, R):
        """Check if U, V, W of rank R satisfy the tensor equation with integer/half-integer entries."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check integer/half-integer constraint
        for matrix in [U, V, W]:
            for val in matrix.flat:
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
        return True
    
    # Try to find lower rank via optimization
    for target_rank in range(schoolbook_rank - 1, 39, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try a few random restarts
        for restart in range(2):
            if time.time() - start_time > time_limit:
                break
            
            rng = np.random.default_rng(restart * 1000 + target_rank)
            # Initialize with small random values
            U0 = rng.normal(0, 0.3, (nm, target_rank))
            V0 = rng.normal(0, 0.3, (mp, target_rank))
            W0 = rng.normal(0, 0.3, (pn, target_rank))
            
            # Use a simple gradient-free approach: coordinate descent with random perturbations
            # This is a heuristic, not guaranteed to find the optimum
            U_curr = U0.copy()
            V_curr = V0.copy()
            W_curr = W0.copy()
            
            # Evaluate initial error
            def compute_error(U, V, W, R):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(R):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += (val - 1.0) ** 2
                return err
            
            init_err = compute_error(U_curr, V_curr, W_curr, target_rank)
            
            # Simple local search: try perturbing each entry
            for iteration in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Randomly pick a matrix and entry to perturb
                mat_idx = rng.integers(0, 3)
                if mat_idx == 0:
                    mat = U_curr
                    shape = (nm, target_rank)
                elif mat_idx == 1:
                    mat = V_curr
                    shape = (mp, target_rank)
                else:
                    mat = W_curr
                    shape = (pn, target_rank)
                
                row = rng.integers(0, shape[0])
                col = rng.integers(0, shape[1])
                
                old_val = mat[row, col]
                # Try a few perturbation values
                best_local_val = old_val
                best_local_err = compute_error(U_curr, V_curr, W_curr, target_rank)
                
                for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                    mat[row, col] = old_val + delta
                    err = compute_error(U_curr, V_curr, W_curr, target_rank)
                    if err < best_local_err:
                        best_local_err = err
                        best_local_val = old_val + delta
                    mat[row, col] = old_val
                
                mat[row, col] = best_local_val
                
                if best_local_err < 1e-8:
                    break
            
            # Check if we found a valid solution
            # Round to half-integers
            U_rounded = np.round(U_curr * 2) / 2
            V_rounded = np.round(V_curr * 2) / 2
            W_rounded = np.round(W_curr * 2) / 2
            
            if verify(U_rounded, V_rounded, W_rounded, target_rank):
                best_U = U_rounded.copy()
                best_V = V_rounded.copy()
                best_W = W_rounded.copy()
                best_rank = target_rank
                break  # Found lower rank, try to go lower
        
        if best_rank < schoolbook_rank:
            continue  # Keep trying to find even lower rank
    
    # Convert to lists for the return interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
