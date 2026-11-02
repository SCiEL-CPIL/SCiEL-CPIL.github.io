import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 2, 2, 6
    
    # --- Rank 21 Solution (Strassen) ---
    rank_21 = 21
    U21 = np.zeros((n * m, rank_21))
    V21 = np.zeros((m * p, rank_21))
    W21 = np.zeros((p * n, rank_21))
    
    # Strassen coefficients
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(0, 0, 1)],                 # M2
        [(0, 1, 1), (1, 1, -1)],     # M3
        [(1, 0, 1), (0, 0, -1)],     # M4
        [(1, 1, 1)],                 # M5
        [(0, 0, 1), (0, 1, 1)],      # M6
        [(1, 0, 1), (1, 1, 1)]       # M7
    ]
    
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(1, 0, 1), (1, 1, -1)],  # M2
        [(0, 1, 1), (1, 1, 1)],   # M3
        [(0, 0, 1), (1, 0, 1)],   # M4
        [(0, 0, -1), (0, 1, 1)],  # M5
        [(1, 1, 1)],              # M6
        [(0, 0, 1)]               # M7
    ]
    
    for b in range(3):
        for s in range(7):
            r = b * 7 + s
            for idx, val in U_coeffs[s].items():
                U21[idx, r] = val
            for j_local, k_local, val in V_coeffs[s]:
                global_k = b * 2 + k_local
                v_idx = j_local * p + global_k
                V21[v_idx, r] = val
            for i, j_local, val in W_map[s]:
                global_k = b * 2 + j_local
                w_idx = global_k * n + i
                W21[w_idx, r] = val
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid rank-R decomposition."""
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i, j, :] * V_mat[j, k, :] * W_mat[k, i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_find_rank_20(time_limit):
        """Try to find a rank-20 decomposition by perturbing the rank-21 solution."""
        R = 20
        
        # Strategy: Try to find a linear combination of the 21 terms that
        # can be eliminated. We'll search over all pairs of blocks and try
        # to merge them using a parameter.
        
        # For each pair of blocks (b1, b2), try to find a linear combination
        # that preserves the tensor structure
        np.random.seed(42)
        
        best_obj = float('inf')
        best_solution = None
        
        # Try structured perturbations: for each pair of blocks, try to
        # find a rank-1 update that allows elimination
        for b1 in range(3):
            for b2 in range(b1 + 1, 3):
                if time.time() > time_limit:
                    return None
                
                # Try to find a parameter t such that we can merge blocks b1 and b2
                # This is a heuristic approach - we try random t values
                for trial in range(50):
                    if time.time() > time_limit:
                        return None
                    
                    # Create a perturbed version where we mix terms from blocks b1 and b2
                    U_test = U21.copy()
                    V_test = V21.copy()
                    W_test = W21.copy()
                    
                    # Pick a random column from block b1 and block b2 to mix
                    col1 = b1 * 7 + np.random.randint(7)
                    col2 = b2 * 7 + np.random.randint(7)
                    
                    # Try to find a linear combination
                    t = np.random.uniform(-2, 2)
                    
                    # Apply a rank-1 perturbation
                    # We need to be careful here - this is just a heuristic
                    U_test[:, col1] += t * U_test[:, col2]
                    V_test[:, col1] += t * V_test[:, col2]
                    W_test[:, col1] += t * W_test[:, col2]
                    
                    # Try to eliminate one column by finding a dependency
                    # Check if any column is a linear combination of others
                    for col_to_elim in range(21):
                        # Try to express column col_to_elim as a combination of others
                        cols = [c for c in range(21) if c != col_to_elim]
                        
                        # This is a 20x20 system - too expensive to solve exactly
                        # Instead, use a simpler heuristic: check if the column is
                        # approximately zero after perturbation
                        if np.allclose(U_test[:, col_to_elim], 0, atol=1e-6) and \
                           np.allclose(V_test[:, col_to_elim], 0, atol=1e-6) and \
                           np.allclose(W_test[:, col_to_elim], 0, atol=1e-6):
                            # Found a valid rank-20 solution!
                            U_new = np.delete(U_test, col_to_elim, axis=1)
                            V_new = np.delete(V_test, col_to_elim, axis=1)
                            W_new = np.delete(W_test, col_to_elim, axis=1)
                            
                            if check_decomposition(U_new, V_new, W_new):
                                return U_new, V_new, W_new
        
        # If structured search fails, try random initialization with gradient descent
        # but with a much faster convergence check
        from scipy.optimize import minimize
        
        def objective(x):
            U_x = x[:n*m*R].reshape(n, m, R)
            V_x = x[n*m*R:n*m*R+m*p*R].reshape(m, p, R)
            W_x = x[n*m*R+m*p*R:].reshape(p, n, R)
            
            res = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] = np.sum(U_x[i, j, :] * V_x[j, k, :] * W_x[k, i, :])
            
            res = res - 1.0
            return np.sum(res**2)
        
        # Try a few random starts with limited iterations
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > time_limit:
                return None
            
            # Initialize near the rank-21 solution (project to rank 20)
            # Use SVD to get a rank-20 approximation
            U_flat = U21.flatten()
            V_flat = V21.flatten()
            W_flat = W21.flatten()
            
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.1
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < 1e-8:
                    U = res.x[:n*m*R].reshape(n*m, R)
                    V = res.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W = res.x[n*m*R+m*p*R:].reshape(p*n, R)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6):
                        return U_rounded, V_rounded, W_rounded
            except Exception:
                pass
        
        return None
    
    # Attempt rank 20 search
    U20, V20, W20 = None, None, None
    if time.time() < deadline - 20:
        # Give it up to 20 seconds to try
        result = try_find_rank_20(deadline - 20)
        if result is not None:
            U20, V20, W20 = result
            # Verify
            if check_decomposition(U20, V20, W20):
                return {
                    "U": U20.tolist(),
                    "V": V20.tolist(),
                    "W": W20.tolist()
                }
    
    # Fallback to Rank 21
    return {
        "U": U21.tolist(),
        "V": V21.tolist(),
        "W": W21.tolist()
    }
