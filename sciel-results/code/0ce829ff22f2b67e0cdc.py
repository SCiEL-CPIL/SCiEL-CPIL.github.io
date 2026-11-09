"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute lower bounds from matrix flattening
    T0 = tensor.reshape(n*m, -1)
    rank0 = np.linalg.matrix_rank(T0)
    T1 = tensor.transpose(1,0,2).reshape(m*p, -1)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = tensor.transpose(2,0,1).reshape(p*n, -1)
    rank2 = np.linalg.matrix_rank(T2)
    lower_bound = max(rank0, rank1, rank2)
    
    # Trivial solution with rank = n*m*p
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    # If lower bound equals trivial rank, we're done
    if lower_bound >= rank_trivial:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    def check_and_round(U, V, W, rank, tol=1e-6):
        """Check if decomposition is valid and try to round to nice values."""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(recon - tensor)
        
        if error < tol:
            # Try to round to integers/half-integers with various factors
            for factor in [1, 2, 4, 8]:
                U_r = np.round(U * factor) / factor
                V_r = np.round(V * factor) / factor
                W_r = np.round(W * factor) / factor
                
                recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                error_r = np.linalg.norm(recon_r - tensor)
                
                if error_r < tol:
                    return U_r, V_r, W_r, True
            return U, V, W, True
        return None, None, None, False
    
    def run_als(U_init, V_init, W_init, trial_rank, max_iter=300, tol=1e-10):
        """Run ALS with better convergence handling."""
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            # Precompute Gram matrices
            VtV = V_curr.T @ V_curr
            WtW = W_curr.T @ W_curr
            UtU = U_curr.T @ U_curr
            
            G_U = VtV * WtW
            G_V = UtU * WtW
            G_W = UtU * VtV
            
            # Compute RHS vectors
            rhs_U = np.einsum('abc,br,cr->ar', tensor, V_curr, W_curr)
            rhs_V = np.einsum('abc,ar,cr->br', tensor, U_curr, W_curr)
            rhs_W = np.einsum('abc,ar,br->cr', tensor, U_curr, V_curr)
            
            # Solve with adaptive regularization
            reg_base = 1e-10
            for reg_factor in [1, 10, 100, 1000]:
                reg = reg_base * reg_factor * np.eye(trial_rank)
                
                try:
                    U_new = np.linalg.solve(G_U + reg, rhs_U.T).T
                    V_new = np.linalg.solve(G_V + reg, rhs_V.T).T
                    W_new = np.linalg.solve(G_W + reg, rhs_W.T).T
                    break
                except np.linalg.LinAlgError:
                    try:
                        U_new = np.linalg.lstsq(G_U + reg, rhs_U.T, rcond=None)[0].T
                        V_new = np.linalg.lstsq(G_V + reg, rhs_V.T, rcond=None)[0].T
                        W_new = np.linalg.lstsq(G_W + reg, rhs_W.T, rcond=None)[0].T
                        break
                    except:
                        continue
            else:
                break
            
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            
            # Check convergence
            recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
            error = np.linalg.norm(recon - tensor)
            if error < tol:
                break
        
        return U_curr, V_curr, W_curr
    
    # Try to find lower rank decompositions
    for trial_rank in range(lower_bound, rank_trivial):
        if time.time() > deadline - 30:
            break
        
        found = False
        
        # Multiple initialization strategies
        init_strategies = []
        
        # Strategy 1: Random Gaussian
        for seed in range(5):
            init_strategies.append(('rand', seed))
        
        # Strategy 2: Sparse random
        for seed in range(3):
            init_strategies.append(('sparse', seed))
        
        # Strategy 3: Structured based on tensor sparsity
        init_strategies.append(('structured', 0))
        
        for strategy, seed in init_strategies:
            if time.time() > deadline - 30:
                break
            
            np.random.seed(seed)
            
            if strategy == 'rand':
                U_curr = np.random.randn(n*m, trial_rank)
                V_curr = np.random.randn(m*p, trial_rank)
                W_curr = np.random.randn(p*n, trial_rank)
            elif strategy == 'sparse':
                U_curr = np.random.randn(n*m, trial_rank)
                V_curr = np.random.randn(m*p, trial_rank)
                W_curr = np.random.randn(p*n, trial_rank)
                # Make 50% of entries zero
                mask_U = np.random.rand(n*m, trial_rank) < 0.5
                mask_V = np.random.rand(m*p, trial_rank) < 0.5
                mask_W = np.random.rand(p*n, trial_rank) < 0.5
                U_curr[mask_U] = 0
                V_curr[mask_V] = 0
                W_curr[mask_W] = 0
            elif strategy == 'structured':
                # Initialize based on the structure of the tensor
                U_curr = np.zeros((n*m, trial_rank))
                V_curr = np.zeros((m*p, trial_rank))
                W_curr = np.zeros((p*n, trial_rank))
                # Use the first trial_rank non-zero tensor entries
                count = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if count < trial_rank:
                                U_curr[i*m+j, count] = 1.0
                                V_curr[j*p+k, count] = 1.0
                                W_curr[k*n+i, count] = 1.0
                                count += 1
                # Add small random perturbation
                U_curr += 0.01 * np.random.randn(n*m, trial_rank)
                V_curr += 0.01 * np.random.randn(m*p, trial_rank)
                W_curr += 0.01 * np.random.randn(p*n, trial_rank)
            
            # Run ALS
            U_curr, V_curr, W_curr = run_als(U_curr, V_curr, W_curr, trial_rank)
            
            # Check if decomposition is valid
            U_r, V_r, W_r, success = check_and_round(U_curr, V_curr, W_curr, trial_rank)
            
            if success:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = trial_rank
                found = True
                break
        
        if found:
            break
    
    # Final polish: if we found a solution, try to improve precision
    if best_rank < rank_trivial:
        if time.time() < deadline - 10:
            # Run more ALS iterations with tighter tolerance
            U_curr, V_curr, W_curr = run_als(best_U, best_V, best_W, best_rank, max_iter=500, tol=1e-12)
            U_r, V_r, W_r, success = check_and_round(U_curr, V_curr, W_curr, best_rank, tol=1e-8)
            if success:
                best_U, best_V, best_W = U_r, V_r, W_r
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
