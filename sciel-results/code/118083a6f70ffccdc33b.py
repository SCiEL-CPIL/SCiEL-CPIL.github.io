"""Product-formula design for Heisenberg-chain time evolution."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_hamiltonians():
    """
    Build the A and B Hamiltonians for a 6-site open XXZ Heisenberg chain.
    A: odd bonds + half field
    B: even bonds + half field
    Anisotropy lambda = 0.8, field h = 0.3
    """
    N = 6
    lambda_aniso = 0.8
    h_field = 0.3
    
    # Spin operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build operators for N sites
    def op_on_site(op, site, N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    # Build A (odd bonds: 0-1, 2-3, 4-5) and B (even bonds: 1-2, 3-4)
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term: each gets half the field
    field_term = 0.5 * h_field
    for i in range(N):
        A += field_term * op_on_site(sz, i, N)
        B += field_term * op_on_site(sz, i, N)
    
    # Bond terms
    def add_bond(H, i, j):
        sx_i = op_on_site(sx, i, N)
        sx_j = op_on_site(sx, j, N)
        sy_i = op_on_site(sy, i, N)
        sy_j = op_on_site(sy, j, N)
        sz_i = op_on_site(sz, i, N)
        sz_j = op_on_site(sz, j, N)
        
        H += sx_i @ sx_j
        H += sy_i @ sy_j
        H += lambda_aniso * (sz_i @ sz_j)
        
    # Odd bonds for A: (0,1), (2,3), (4,5)
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # Even bonds for B: (1,2), (3,4)
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)
    
    return A, B

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients.
    Returns (a, b) where each is a list of 12 finite numbers in [-50, 50]
    with sum(a) = sum(b) = 1.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Leave margin before 150s wall-clock kill
    
    if stages != 12:
        # Fallback for non-12 stages
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Pre-compute Hamiltonians
    A, B = _build_hamiltonians()
    H_total = A + B
    
    # Pre-compute exact propagators for the 3 time points using eigen-decomposition
    t_values = [0.25, 0.5, 1.0]
    eigvals_H, eigvecs_H = np.linalg.eigh(H_total)
    U_exacts = []
    for t in t_values:
        U_exact = (eigvecs_H * np.exp(-1j * t * eigvals_H)) @ eigvecs_H.conj().T
        U_exacts.append(U_exact)
    
    # Diagonalize A and B for faster exponential calculation
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    dim = 64
    
    # Helper to apply the product formula efficiently
    def apply_product_formula(a, b, t):
        # U = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
        # Order: k=1 first, so U = ... U_12 @ U_11 @ ... @ U_1
        # where U_k = exp(-i a_k t A) @ exp(-i b_k t B)
        # Actually: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
        # This means: U = (exp(-i b_12 t B) exp(-i a_12 t A)) @ ... @ (exp(-i b_1 t B) exp(-i a_1 t A))
        # Wait, "in order k = 1..12" typically means the first factor applied is k=1
        # U = U_12 * U_11 * ... * U_1 where U_k = exp(-i b_k t B) * exp(-i a_k t A)
        # So U @ |psi> = U_12 (U_11 (... (U_1 |psi>) ...))
        # Which means U = U_12 @ U_11 @ ... @ U_1
        
        U_curr = np.eye(dim, dtype=complex)
        
        for k in range(12):
            # exp(-i b_k t B)
            exp_B = np.exp(-1j * b[k] * t * eigvals_B)
            U_B = (eigvecs_B * exp_B) @ eigvecs_B.conj().T
            
            # exp(-i a_k t A)
            exp_A = np.exp(-1j * a[k] * t * eigvals_A)
            U_A = (eigvecs_A * exp_A) @ eigvecs_A.conj().T
            
            # U_k = exp(-i b_k t B) @ exp(-i a_k t A)
            U_k = U_B @ U_A
            U_curr = U_k @ U_curr
        
        return U_curr
    
    # Objective function
    def objective(x):
        if time.time() > deadline:
            return float('inf')
            
        # x is 24 parameters: first 12 are a, next 12 are b
        a = x[0:12]
        b = x[12:24]
        
        # Check constraints
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        penalty = (sum_a - 1.0)**2 * 1000 + (sum_b - 1.0)**2 * 1000
        
        # Check bounds
        for val in a:
            if val < -50 or val > 50:
                penalty += 10000
        for val in b:
            if val < -50 or val > 50:
                penalty += 10000
        
        error = 0.0
        for i, t in enumerate(t_values):
            U_approx = apply_product_formula(a, b, t)
            diff = U_approx - U_exacts[i]
            err = np.linalg.norm(diff, 2)
            error += err
        
        return error / len(t_values) + penalty
    
    # Initial guess: uniform
    x0 = np.array([1.0/12.0] * 24)
    
    # Try Nelder-Mead with all 24 parameters
    best_x = x0.copy()
    best_score = objective(x0)
    
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # If time remains, try a second optimization from a different starting point
    if time.time() < deadline - 25.0:
        # Try starting from a perturbed version
        np.random.seed(42)
        x0_2 = x0 + np.random.normal(0, 0.03, 24)
        try:
            from scipy.optimize import minimize
            res2 = minimize(objective, x0_2, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass
    
    # If more time remains, try L-BFGS-B with bounds
    if time.time() < deadline - 20.0:
        try:
            from scipy.optimize import minimize
            
            # Define bounds
            bounds = [(-50.0, 50.0)] * 24
            
            # Use a simpler objective without penalty for L-BFGS-B
            def objective_lbfgsb(x):
                if time.time() > deadline:
                    return float('inf')
                a = x[0:12]
                b = x[12:24]
                error = 0.0
                for i, t in enumerate(t_values):
                    U_approx = apply_product_formula(a, b, t)
                    diff = U_approx - U_exacts[i]
                    err = np.linalg.norm(diff, 2)
                    error += err
                return error / len(t_values)
            
            res3 = minimize(objective_lbfgsb, best_x, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 100, 'ftol': 1e-9})
            if res3.fun < best_score:
                best_score = res3.fun
                best_x = res3.x.copy()
        except Exception:
            pass
    
    # Extract a and b
    a = best_x[0:12].copy()
    b = best_x[12:24].copy()
    
    # Normalize to ensure sums are exactly 1
    if np.sum(a) != 0:
        a = a / np.sum(a)
    if np.sum(b) != 0:
        b = b / np.sum(b)
    
    # Final check for finite numbers in [-50, 50]
    a = [max(-50.0, min(50.0, float(ai))) for ai in a]
    b = [max(-50.0, min(50.0, float(bi))) for bi in b]
    
    # Re-normalize after clipping to maintain sum=1
    sum_a = sum(a)
    sum_b = sum(b)
    if sum_a != 0:
        a = [ai / sum_a for ai in a]
    if sum_b != 0:
        b = [bi / sum_b for bi in b]
    
    return a, b
# EVOLVE-BLOCK-END
