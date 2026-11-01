"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _precompute_constants():
    """Precompute all constants needed for the bound computation."""
    degrees = np.array([0, 4, 8, 12], dtype=np.float64)
    sqrt_pi = math.sqrt(math.pi)
    
    # w_n = 2^n * n! * sqrt(pi)  (norm squared of H_n * exp(-x^2/2))
    w = np.zeros(4)
    for i, n in enumerate([0, 4, 8, 12]):
        w[i] = (2.0 ** n) * math.factorial(n) * sqrt_pi
    
    # Diagonal M contribution: (n + 0.5) * w_n
    m_diag = (degrees + 0.5) * w
    
    # Hermite values at 0
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1680.0
    H12_0 = -103950.0
    
    # c12 = (c0 + c4*H4_0 + c8*H8_0) / (-H12_0)
    # c12 = (c0 - 12*c4 + 1680*c8) / 103950
    c12_coeffs = np.array([1.0, H4_0, H8_0]) / (-H12_0)
    
    return w, m_diag, c12_coeffs

def _bound_and_grad(x):
    """
    Compute the uncertainty bound M/N and its gradient with respect to (c0, c4, c8).
    
    x = [c0, c4, c8]
    c12 = c12_coeffs[0]*c0 + c12_coeffs[1]*c4 + c12_coeffs[2]*c8
    
    coeffs = [c0, c4, c8, c12]
    
    N = sum_i c_i^2 * w_i
    M = sum_i c_i^2 * m_diag_i   (no off-diagonal terms since no degrees differ by 2)
    
    Bound = M/N
    
    d(Bound)/dc_j = (dM/dc_j * N - M * dN/dc_j) / N^2
    
    dN/dc_j = 2 * c_j * w_j
    dM/dc_j = 2 * c_j * m_diag_j
    
    But c_3 = c12 depends on x, so we need chain rule:
    dc_j/dx_k for j=0,1,2 is delta_jk
    dc_3/dx_k = c12_coeffs[k]
    """
    w, m_diag, c12_coeffs = _precompute_constants()
    
    c0, c4, c8 = x
    c12 = c12_coeffs[0] * c0 + c12_coeffs[1] * c4 + c12_coeffs[2] * c8
    coeffs = np.array([c0, c4, c8, c12])
    
    # Compute N and M
    c_sq = coeffs * coeffs
    N = np.dot(c_sq, w)
    M = np.dot(c_sq, m_diag)
    
    if N <= 0:
        return np.inf, np.array([0.0, 0.0, 0.0])
    
    bound = M / N
    
    # Gradient computation
    # dN/dcoeffs = 2 * coeffs * w
    # dM/dcoeffs = 2 * coeffs * m_diag
    dN_dc = 2.0 * coeffs * w
    dM_dc = 2.0 * coeffs * m_diag
    
    # d(bound)/dcoeffs_j = (dM_dc[j] * N - M * dN_dc[j]) / N^2
    dbound_dc = (dM_dc * N - M * dN_dc) / (N * N)
    
    # Chain rule: dbound/dx_k = sum_j dbound_dc[j] * dc_j/dx_k
    # dc_0/dx = [1, 0, 0]
    # dc_1/dx = [0, 1, 0]
    # dc_2/dx = [0, 0, 1]
    # dc_3/dx = c12_coeffs
    grad = np.zeros(3)
    grad[0] = dbound_dc[0] + dbound_dc[3] * c12_coeffs[0]
    grad[1] = dbound_dc[1] + dbound_dc[3] * c12_coeffs[1]
    grad[2] = dbound_dc[2] + dbound_dc[3] * c12_coeffs[2]
    
    return bound, grad

def _objective(x):
    """Objective function for optimization."""
    val, _ = _bound_and_grad(x)
    return val

def _objective_grad(x):
    """Objective function and gradient for optimization."""
    return _bound_and_grad(x)

def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f)).
    
    Returns three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 coefficient is fixed so that g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 170  # Safety margin: finish well before 210s
    
    # Precompute constants
    w, m_diag, c12_coeffs = _precompute_constants()
    
    # Initial guess from prior work
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _objective(x0)
    
    # Define bounds for the differential evolution
    # The coefficients are expected to be small. 
    # c0 ~ -1, c4 ~ 0.04, c8 ~ 0.0003
    # Let's allow a wide range to be safe, but not infinite.
    bounds = [(-100, 100), (-10, 10), (-10, 10)]
    
    # 1. Global search using Differential Evolution
    # DE is robust to local minima and scaling issues.
    try:
        res_de = minimize(
            _objective, 
            x0, 
            bounds=bounds, 
            method='differential_evolution',
            options={
                'maxiter': 100, 
                'polish': True, 
                'init': 'sobol',
                'popsize': 20,
                'seed': 42
            }
        )
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_x = res_de.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_x

    # 2. Local polish with L-BFGS-B from the best DE result
    try:
        res_local = minimize(_objective, best_x, jac=_objective_grad, method='L-BFGS-B',
                          options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
        if res_local.fun < best_score:
            best_score = res_local.fun
            best_x = res_local.x.copy()
    except Exception:
        pass
        
    # 3. Additional local polish from a few random points to ensure robustness
    np.random.seed(123)
    for i in range(10):
        if time.time() > deadline - 10:
            break
        x_start = np.random.uniform(-5, 5, 3)
        try:
            res_rand = minimize(_objective, x_start, jac=_objective_grad, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res_rand.fun < best_score:
                best_score = res_rand.fun
                best_x = res_rand.x.copy()
        except Exception:
            pass

    return best_x

# EVOLVE-BLOCK-END
