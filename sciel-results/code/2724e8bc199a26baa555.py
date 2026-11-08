import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

# Precompute matrices for the rational function representation
# c = [c0, c4, c8, c12]
# sigma_p^2 = c^T P c / (c^T c) where P = diag(1, 5, 9, 13)
# sigma_x^2 = c^T X c / (c^T c) where X has entries K_nm
# Objective = (c^T P c)(c^T X c) / (c^T c)^2

# Degrees
_degrees = np.array([0, 4, 8, 12])

# P matrix: diag(deg+1)
_P = np.diag(_degrees + 1.0)

# X matrix: K_nm
_X = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        n, m = _degrees[i], _degrees[j]
        if n == m:
            _X[i, j] = n + 0.5
        elif abs(n - m) == 2:
            if n > m:
                _X[i, j] = 0.5 * np.sqrt(n * (n - 1))
            else:
                _X[i, j] = 0.5 * np.sqrt((n + 1) * (n + 2))

# Constraint: c0 + 12*c4 + 1680*c8 + 3991680*c12 = 0
# c12 = -(c0 + 12*c4 + 1680*c8) / 3991680
# We parameterize by (c0, c4, c8) and compute c12.

def _c12_from(c0, c4, c8):
    return -(c0 + 12.0 * c4 + 1680.0 * c8) / 3991680.0

def _full_coeffs(params):
    """params = [c0, c4, c8], returns [c0, c4, c8, c12]"""
    c0, c4, c8 = params
    c12 = _c12_from(c0, c4, c8)
    return np.array([c0, c4, c8, c12])

def _objective_and_grad(params):
    """
    Returns (objective_value, gradient) for params = [c0, c4, c8].
    Objective = (c^T P c)(c^T X c) / (c^T c)^2 where c = [c0, c4, c8, c12]
    """
    c = _full_coeffs(params)
    
    # Compute quadratic forms
    N = c @ c  # c^T c
    if N < 1e-30:
        return np.inf, np.zeros(3)
    
    p_form = c @ _P @ c  # c^T P c
    x_form = c @ _X @ c  # c^T X c
    
    obj = (p_form * x_form) / (N * N)
    
    # Gradient via chain rule
    # Let f = (p_form * x_form) / N^2
    # df/dN = -2 * p_form * x_form / N^3
    # df/dp_form = x_form / N^2
    # df/dx_form = p_form / N^2
    # 
    # dN/dc = 2c
    # dp_form/dc = 2*P*c
    # dx_form/dc = 2*X*c
    #
    # df/dc = (df/dN)*(dN/dc) + (df/dp_form)*(dp_form/dc) + (df/dx_form)*(dx_form/dc)
    
    dN_dc = 2.0 * c
    dp_dc = 2.0 * (_P @ c)
    dx_dc = 2.0 * (_X @ c)
    
    df_dN = -2.0 * p_form * x_form / (N ** 3)
    df_dp = x_form / (N ** 2)
    df_dx = p_form / (N ** 2)
    
    grad_c = df_dN * dN_dc + df_dp * dp_dc + df_dx * dx_dc
    
    # Now we need gradient with respect to params = [c0, c4, c8]
    # c = [c0, c4, c8, c12] where c12 = -(c0 + 12*c4 + 1680*c8)/3991680
    # dc12/dc0 = -1/3991680
    # dc12/dc4 = -12/3991680
    # dc12/dc8 = -1680/3991680
    
    d_c12_dc0 = -1.0 / 3991680.0
    d_c12_dc4 = -12.0 / 3991680.0
    d_c12_dc8 = -1680.0 / 3991680.0
    
    # grad_params[i] = grad_c[i] + grad_c[3] * dc12/dparam[i]
    grad_params = np.array([
        grad_c[0] + grad_c[3] * d_c12_dc0,
        grad_c[1] + grad_c[3] * d_c12_dc4,
        grad_c[2] + grad_c[3] * d_c12_dc8
    ])
    
    return obj, grad_params

def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin before 210s kill
    
    # Structural Change: 
    # 1. Use a multi-stage optimization strategy combining global search (Differential Evolution) 
    #    with local refinement (BFGS and Nelder-Mead).
    # 2. Exploit scale invariance by fixing c0 = 1.0, reducing the problem to 2D (c4, c8).
    # 3. Use Sobol initialization for DE to ensure better coverage of the 2D parameter space.
    # 4. Apply high-precision local refinement to squeeze out the last bits of performance.
    
    def obj_2d(params):
        """
        Objective function for 2D optimization (c4, c8) with c0 fixed to 1.
        """
        c4, c8 = params
        full_params = np.array([1.0, c4, c8])
        val, _ = _objective_and_grad(full_params)
        return val

    def grad_2d(params):
        """
        Gradient for 2D optimization (c4, c8) with c0 fixed to 1.
        """
        c4, c8 = params
        full_params = np.array([1.0, c4, c8])
        _, grad_full = _objective_and_grad(full_params)
        return grad_full[1:]  # Only c4 and c8 gradients

    # Bounds for c4 and c8
    # Based on seed values: c4 ~ -0.035, c8 ~ -0.00027
    # Expand bounds significantly to allow global search
    bounds = [(-1.0, 1.0), (-0.1, 0.1)]
    
    best_params = np.array([-0.035, -0.00027]) # Fallback (c4, c8)
    best_score = np.inf
    
    # Phase 1: Global search using Differential Evolution with Sobol initialization
    if time.time() - start_time < deadline:
        try:
            res = differential_evolution(
                obj_2d, 
                bounds, 
                seed=42,
                maxiter=2000,
                tol=1e-14,
                polish=True,
                init='sobol',
                popsize=30,
                update_interval=1
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass

    # Phase 2: Local refinement with BFGS from the best global solution
    if time.time() - start_time < deadline:
        try:
            res = minimize(
                obj_2d,
                best_params,
                jac=grad_2d,
                method='BFGS',
                options={'maxiter': 10000, 'gtol': 1e-16}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass

    # Phase 3: Nelder-Mead polish for high precision
    if time.time() - start_time < deadline:
        try:
            res = minimize(
                obj_2d,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-18, 'adaptive': True}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass

    # Return [c0, c4, c8] as required by the interface
    return np.array([1.0, best_params[0], best_params[1]])
