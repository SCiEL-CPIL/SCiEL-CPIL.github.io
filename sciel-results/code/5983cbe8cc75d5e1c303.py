"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_optimize(U, V, W, n, m, p, max_iters=300, lr=0.005, seed=0, deadline=None, momentum_beta=0.9):
    """Optimize U, V, W to satisfy the tensor identity using gradient descent with momentum."""
    np.random.seed(seed)
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    momentum_U = np.zeros_like(U)
    momentum_V = np.zeros_like(V)
    momentum_W = np.zeros_like(W)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Compute error and gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        total_err_sq = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i*m + j, :]
                    vk = V[j*p + k, :]
                    wk = W[k*n + i, :]
                    s_val = np.dot(ui, vk * wk)
                    err = s_val - 1.0
                    total_err_sq += err * err
                    
                    factor = 2.0 * err
                    grad_U[i*m + j, :] += factor * (vk * wk)
                    grad_V[j*p + k, :] += factor * (ui * wk)
                    grad_W[k*n + i, :] += factor * (ui * vk)
        
        err_norm = np.sqrt(total_err_sq)
        
        if err_norm < 1e-8:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
        
        # Update momentum
        momentum_U = momentum_beta * momentum_U + (1 - momentum_beta) * grad_U
        momentum_V = momentum_beta * momentum_V + (1 - momentum_beta) * grad_V
        momentum_W = momentum_beta * momentum_W + (1 - momentum_beta) * grad_W
        
        # Update parameters
        U -= lr * momentum_U
        V -= lr * momentum_V
        W -= lr * momentum_W
        
        if err_norm < best_err:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-7):
        return best_U, best_V, best_W, best_err
    
    return None, None, None, best_err

def _try_structured_init(n, m, p, R, max_iters=300, lr=0.005, seed=0, deadline=None):
    """Try to find a valid decomposition of rank R using structured initialization."""
    np.random.seed(seed)
    
    # Start with schoolbook and truncate
    schoolbook_rank = n * m * p
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    U = U_school[:, :R].copy()
    V = V_school[:, :R].copy()
    W = W_school[:, :R].copy()
    
    # Add small noise
    U += np.random.randn(*U.shape) * 0.01
    V += np.random.randn(*V.shape) * 0.01
    W += np.random.randn(*W.shape) * 0.01
    
    return _try_optimize(U, V, W, n, m, p, max_iters=max_iters, lr=lr, seed=seed, deadline=deadline)

def _try_random_init(n, m, p, R, max_iters=300, lr=0.005, seed=0, deadline=None):
    """Try to find a valid decomposition of rank R using random initialization."""
    np.random.seed(seed)
    U = np.random.randn(n*m, R) * 0.1
    V = np.random.randn(m*p, R) * 0.1
    W = np.random.randn(p*n, R) * 0.1
    
    return _try_optimize(U, V, W, n, m, p, max_iters=max_iters, lr=lr, seed=seed, deadline=deadline)

def _try_rational_search(n, m, p, R, deadline=None):
    """
    Try to find a rational decomposition of rank R by solving a linear system.
    This is a heuristic: we fix all but a few entries and solve for the rest.
    """
    if deadline is not None and time.time() > deadline - 15:
        return None, None, None, float('inf')
    
    # For small R, we can try to solve the system of equations:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is nonlinear, but if we fix U and V, we can solve for W linearly.
    # Let's try fixing U and V to random small rationals and solving for W.
    
    np.random.seed(42)
    
    # Try a few random rational initializations for U and V
    for trial in range(5):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Random rational U and V with entries in {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n*m, R)).astype(float)
        V = np.random.choice([-1, 0, 1], size=(m*p, R)).astype(float)
        
        # Ensure no row is all zeros (to avoid trivial solutions)
        for r in range(n*m):
            if np.all(U[r] == 0):
                U[r, 0] = 1
        for r in range(m*p):
            if np.all(V[r] == 0):
                V[r, 0] = 1
        
        # Now solve for W: for each (i, k), we have:
        # sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is not linear in W because W appears multiplied by U and V.
        # Actually, for fixed U and V, the equation is:
        # sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
        # Let A[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        # Then sum_c A[i,k,c] * W[k*n+i, c] = 1
        # This is a linear system in W!
        
        # Build the linear system for W
        # Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,R)
        # Total variables: p * n * R
        # Equations: for each (i, k), sum_c A[i,k,c] * W[k*n+i, c] = 1
        # Total equations: n * p
        
        num_vars = p * n * R
        num_eqs = n * p
        
        # Build matrix M of shape (num_eqs, num_vars) and vector b of shape (num_eqs)
        M = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for k in range(p):
                eq_idx = i * p + k
                for c in range(R):
                    # A[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                    A_val = 0.0
                    for j in range(m):
                        A_val += U[i*m + j, c] * V[j*p + k, c]
                    var_idx = (k * n + i) * R + c
                    M[eq_idx, var_idx] = A_val
        
        # Solve M @ W_flat = b
        # Use least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
            W = W_flat.reshape(p * n, R)
        except:
            continue
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-7):
            return U, V, W, 0.0
    
    return None, None, None, float('inf')

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower ranks
    # Try a set of target ranks in descending order
    target_ranks = [120, 110, 100, 90, 80, 70, 60, 50, 48, 45, 40]
    
    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try rational search first (might find exact solutions)
        if time.time() < deadline - 30:
            U_opt, V_opt, W_opt, err = _try_rational_search(n, m, p, R, deadline=deadline)
            if U_opt is not None:
                best_rank = R
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                found = True
                print(f"Found valid decomposition with rank {R} using rational search")
        
        if not found:
            # Try both random and structured initializations
            for init_type in ['random', 'structured']:
                if time.time() > deadline - 10:
                    break
                    
                for seed in range(5):
                    if time.time() > deadline - 5:
                        break
                        
                    if init_type == 'random':
                        U_opt, V_opt, W_opt, err = _try_random_init(
                            n, m, p, R, max_iters=200, lr=0.005, seed=seed, deadline=deadline
                        )
                    else:
                        U_opt, V_opt, W_opt, err = _try_structured_init(
                            n, m, p, R, max_iters=200, lr=0.005, seed=seed, deadline=deadline
                        )
                        
                    if U_opt is not None:
                        best_rank = R
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        found = True
                        print(f"Found valid decomposition with rank {R} using {init_type} init")
                        break # Move to next lower rank
                
                if found:
                    break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
