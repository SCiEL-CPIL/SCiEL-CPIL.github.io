import numpy as np
from scipy.optimize import minimize
import time

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
    dN_dc = 2.0 * c
    dp_dc = 2.0 * (_P @ c)
    dx_dc = 2.0 * (_X @ c)
    
    df_dN = -2.0 * p_form * x_form / (N ** 3)
    df_dp = x_form / (N ** 2)
    df_dx = p_form / (N ** 2)
    
    grad_c = df_dN * dN_dc + df_dp * dp_dc + df_dx * dx_dc
    
    # Gradient with respect to params = [c0, c4, c8]
    d_c12_dc0 = -1.0 / 3991680.0
    d_c12_dc4 = -12.0 / 3991680.0
    d_c12_dc8 = -1680.0 / 3991680.0
    
    grad_params = np.array([
        grad_c[0] + grad_c[3] * d_c12_dc0,
        grad_c[1] + grad_c[3] * d_c12_dc4,
        grad_c[2] + grad_c[3] * d_c12_dc8
    ])
    
    return obj, grad_params

def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin before 210s kill
    
    # Structural Change: Multi-start L-BFGS-B in full 3D space (c0, c4, c8)
    # with analytic gradients. The objective is homogeneous of degree 0,
    # so the scale of c doesn't matter, but we optimize in 3D to avoid
    # potential issues with fixing c0=1 if the optimum has c0 near 0.
    
    best_val = np.inf
    best_params = np.array([1.0, -0.035, -0.00027])  # Fallback
    
    # Generate diverse starting points
    starting_points = [
        np.array([1.0, -0.035, -0.00027]),  # seed
        np.array([1.0, -0.05, -0.001]),
        np.array([1.0, -0.03, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.05, 0.001]),
        np.array([2.0, -0.07, -0.0005]),
        np.array([0.5, -0.0175, -0.000135]),
        np.array([-1.0, 0.035, 0.00027]),
    ]
    
    # Add a grid
    c0_vals = [0.5, 1.0, 2.0]
    c4_vals = [-0.05, -0.03, 0.0, 0.03, 0.05]
    c8_vals = [-0.001, 0.0, 0.001]
    for c0 in c0_vals:
        for c4 in c4_vals:
            for c8 in c8_vals:
                starting_points.append(np.array([c0, c4, c8]))
    
    for sp in starting_points:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(
                lambda p: _objective_and_grad(p)[0],
                sp,
                jac=lambda p: _objective_and_grad(p)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return best_params
