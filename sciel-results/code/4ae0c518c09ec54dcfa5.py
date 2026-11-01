"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

def _precompute_constants():
    """Precompute all constants needed for the objective and gradient."""
    sqrt_pi = math.sqrt(math.pi)
    
    # Degrees of the Hermite polynomials in the combination
    degrees = [0, 4, 8, 12]
    n_terms = len(degrees)
    
    # Norms: norm_n = 2^n * n! * sqrt(pi)
    norms = np.zeros(n_terms)
    for i, n in enumerate(degrees):
        norms[i] = (2.0 ** n) * float(math.factorial(n)) * sqrt_pi
    
    # N matrix: N[i,j] = int H_i H_j exp(-x^2) dx = norms[i] * delta_{ij}
    # So N = sum_i c_i^2 * norms[i]
    
    # M matrix: M[i,j] = int x^2 H_i(x) H_j(x) exp(-x^2) dx
    # Using x^2 H_n = (1/4) H_{n+2} + (n+1/2) H_n + (n(n-1)/4) H_{n-2}
    # M[i,j] is non-zero only if degrees[i] == degrees[j] (diag) or |degrees[i]-degrees[j]| == 2
    M_mat = np.zeros((n_terms, n_terms))
    for i, ni in enumerate(degrees):
        for j, nj in enumerate(degrees):
            if ni == nj:
                # Diagonal: (n + 1/2) * 2^n * n! * sqrt(pi)
                M_mat[i, j] = (ni + 0.5) * norms[i]
            elif abs(ni - nj) == 2:
                # Off-diagonal
                n_low = min(ni, nj)
                n_high = max(ni, nj)
                # int x^2 H_{n_low} H_{n_high} exp(-x^2) dx
                # = (1/4) * 2^{n_high} * n_high! * sqrt(pi)  [from the H_{n_low+2} term when n_low+2=n_high]
                # = (1/4) * 2^{n_high} * float(factorial(n_high)) * sqrt_pi
                # Also equals (n_high*(n_high-1)/4) * 2^{n_low} * float(factorial(n_low)) * sqrt_pi
                # Both should be equal: let's verify
                val1 = 0.25 * (2.0 ** n_high) * float(math.factorial(n_high)) * sqrt_pi
                val2 = (n_high * (n_high - 1) / 4.0) * (2.0 ** n_low) * float(math.factorial(n_low)) * sqrt_pi
                # They should be equal; use val1
                M_mat[i, j] = val1
                M_mat[j, i] = val1
    
    # Hermite polynomial values at 0
    H_vals_0 = {}
    for n in degrees:
        # Compute H_n(0) using the explicit formula
        # H_n(x) = n! * sum_{k=0}^{floor(n/2)} (-1)^k * (2x)^(n-2k) / (k! * (n-2k)!)
        # H_n(0) = 0 if n is odd
        # H_n(0) = n! * sum_{k=0}^{n/2} (-1)^k * 0^(n-2k) / (k! * (n-2k)!)
        # Only the term where n-2k=0 survives, i.e., k=n/2
        # H_n(0) = n! * (-1)^(n/2) / ((n/2)! * 0!) = n! * (-1)^(n/2) / (n/2)!
        if n % 2 == 1:
            H_vals_0[n] = 0.0
        else:
            half = n // 2
            H_vals_0[n] = float(math.factorial(n)) * ((-1.0) ** half) / float(math.factorial(half))
    
    # c12 = (c0*H0(0) + c4*H4(0) + c8*H8(0)) / (-H12(0))
    H0_0 = H_vals_0[0]
    H4_0 = H_vals_0[4]
    H8_0 = H_vals_0[8]
    H12_0 = H_vals_0[12]
    
    return {
        'sqrt_pi': sqrt_pi,
        'degrees': degrees,
        'norms': norms,
        'M_mat': M_mat,
        'H0_0': H0_0,
        'H4_0': H4_0,
        'H8_0': H8_0,
        'H12_0': H12_0,
    }

_CONSTANTS = _precompute_constants()

def _bound_and_grad(x):
    """
    Compute the uncertainty bound M/N and its gradient with respect to (c0, c4, c8).
    
    x = [c0, c4, c8]
    c12 = (c0*H0_0 + c4*H4_0 + c8*H8_0) / (-H12_0)
    
    coeffs = [c0, c4, c8, c12]
    N = sum_i coeffs[i]^2 * norms[i]
    M = coeffs^T @ M_mat @ coeffs
    
    bound = M / N
    """
    c0, c4, c8 = x
    H0_0 = _CONSTANTS['H0_0']
    H4_0 = _CONSTANTS['H4_0']
    H8_0 = _CONSTANTS['H8_0']
    H12_0 = _CONSTANTS['H12_0']
    norms = _CONSTANTS['norms']
    M_mat = _CONSTANTS['M_mat']
    
    # c12 as function of (c0, c4, c8)
    c12 = (c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / (-H12_0)
    
    coeffs = np.array([c0, c4, c8, c12])
    
    # N = coeffs^T @ diag(norms) @ coeffs
    N = float(np.dot(coeffs * coeffs, norms))
    
    if N <= 0:
        return np.inf, np.zeros(3)
    
    # M = coeffs^T @ M_mat @ coeffs
    M = float(coeffs @ M_mat @ coeffs)
    
    bound = M / N
    
    # Gradient: d(bound)/d(x_k) for k in {0,1,2}
    # coeffs = [c0, c4, c8, c12(c0,c4,c8)]
    # d(coeffs)/d(c0) = [1, 0, 0, H0_0/(-H12_0)]
    # d(coeffs)/d(c4) = [0, 1, 0, H4_0/(-H12_0)]
    # d(coeffs)/d(c8) = [0, 0, 1, H8_0/(-H12_0)]
    
    dcoeffs_dc0 = np.array([1.0, 0.0, 0.0, H0_0 / (-H12_0)])
    dcoeffs_dc4 = np.array([0.0, 1.0, 0.0, H4_0 / (-H12_0)])
    dcoeffs_dc8 = np.array([0.0, 0.0, 1.0, H8_0 / (-H12_0)])
    
    # dN/d(coeffs) = 2 * coeffs * norms
    dN_dc = 2.0 * coeffs * norms
    
    # dM/d(coeffs) = 2 * M_mat @ coeffs
    dM_dc = 2.0 * (M_mat @ coeffs)
    
    # d(bound)/d(coeffs) = (dM_dc * N - M * dN_dc) / N^2
    dbound_dc = (dM_dc * N - M * dN_dc) / (N * N)
    
    # Chain rule: d(bound)/d(c0) = dbound_dc @ dcoeffs_dc0, etc.
    grad = np.array([
        float(dbound_dc @ dcoeffs_dc0),
        float(dbound_dc @ dcoeffs_dc4),
        float(dbound_dc @ dcoeffs_dc8),
    ])
    
    return bound, grad

def _objective(x):
    """Objective function for scipy optimizers."""
    val, _ = _bound_and_grad(x)
    return val

def _objective_grad(x):
    """Objective and gradient for scipy optimizers."""
    return _bound_and_grad(x)

def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f)).
    
    Returns three coefficients (c0, c4, c8) of the Hermite combination.
    """
    start_time = time.time()
    deadline = start_time + 175  # Safety margin: finish well before 210s
    
    # Initial guess from prior work
    c0_init, c4_init, c8_init = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    x0 = np.array([c0_init, c4_init, c8_init])
    
    best_result = x0.copy()
    best_score = _objective(x0)
    
    # Step 1: L-BFGS-B from initial point with analytic gradient
    try:
        res = minimize(_objective, x0, jac=_objective_grad, method='L-BFGS-B',
                      options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 80:
        return best_result
    
    # Step 2: Differential evolution for global search
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    try:
        de_result = differential_evolution(_objective, bounds, seed=42, 
                                            maxiter=200, tol=1e-10, 
                                            popsize=15, polish=True,
                                            init='sobol')
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 40:
        return best_result
    
    # Step 3: Multiple random restarts with L-BFGS-B
    np.random.seed(42)
    n_restarts = 0
    while time.time() < deadline - 20 and n_restarts < 100:
        x_rand = np.random.uniform(-5, 5, 3)
        try:
            res = minimize(_objective, x_rand, jac=_objective_grad, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
        n_restarts += 1
    
    # Step 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_result, jac=_objective_grad, method='L-BFGS-B',
                          options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
