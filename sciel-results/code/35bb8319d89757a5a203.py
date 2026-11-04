"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _compute_legendre_basis(N, max_degree):
    """
    Compute the Legendre polynomial basis vectors truncated to N points.
    Returns a matrix of shape (N, max_degree + 1) where each column is a basis vector.
    The basis functions are normalized to have unit L2 norm over the discrete grid.
    """
    # Use Gauss-Legendre nodes and weights for a better approximation of orthogonality
    # Actually, for discrete orthogonality on a uniform grid, we can just use the standard
    # Legendre polynomials evaluated at uniform points and then orthonormalize.
    # But simpler: just use the standard Legendre polynomials P_k(t) evaluated at t in [-1, 1].
    # We'll use scipy.special.eval_legendre
    
    from scipy.special import eval_legendre
    
    # Uniform grid in [-1, 1]
    t = np.linspace(-1, 1, N)
    
    basis = np.zeros((N, max_degree + 1))
    for k in range(max_degree + 1):
        basis[:, k] = eval_legendre(k, t)
        
    # Orthonormalize using Gram-Schmidt to ensure orthogonality on the discrete grid
    # This is important because Legendre polynomials are orthogonal w.r.t. continuous measure,
    # but we need discrete orthogonality for our FFT-based computations to be clean.
    # Actually, let's just use the raw Legendre polynomials and let the optimizer handle it.
    # The key insight is that low-degree polynomials have smooth spectra, which helps with autoconvolution.
    
    return basis

def _coeff_to_sequence(coeffs, basis):
    """Convert Legendre coefficients to the time-domain sequence."""
    return basis @ coeffs

def _sequence_to_coeffs(x, basis):
    """Convert time-domain sequence to Legendre coefficients (projection)."""
    # If basis is orthonormal, this is just basis.T @ x
    # For non-orthonormal, we'd need (B^T B)^-1 B^T x
    # Let's assume basis is orthonormal for simplicity, or compute the projection properly
    B = basis
    # Projection: c = (B^T B)^-1 B^T x
    # For large N, B^T B should be close to identity if basis is orthonormal
    BTx = B.T @ x
    BTB = B.T @ B
    try:
        coeffs = np.linalg.solve(BTB, BTx)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(BTB, BTx, rcond=None)[0]
    return coeffs

def _smooth_metric_and_grad_coeffs(coeffs, basis, beta=50.0):
    """
    Compute the smooth metric and its gradient with respect to the Legendre coefficients.
    """
    x = _coeff_to_sequence(coeffs, basis)
    
    # Compute autoconvolution
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    
    s = _integral(x)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(coeffs)
    
    # Soft max absolute value
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    if max_val == 0:
        val = 0.0
        grad_conv = np.zeros_like(conv)
    else:
        shifted = abs_conv - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        val = max_val + np.log(sum_exp) / beta
        grad_conv = (exp_shifted * np.sign(conv)) / sum_exp
    
    # Gradient with respect to x
    # dval/dx via FFT adjoint
    grad_conv_full = np.zeros(_FFT_SIZE)
    grad_conv_full[:2 * N - 1] = grad_conv
    dval_dC = np.fft.rfft(grad_conv_full, _FFT_SIZE)
    dval_dX = 2 * fx * dval_dC
    dval_dx = np.fft.irfft(dval_dX, _FFT_SIZE)[:N]
    
    # Objective J = val / s^2
    # dJ/dx = (dval_dx * s - 2 * val) / s^3
    grad_J_dx = (dval_dx * s - 2 * val) / (s * s * s)
    
    # Gradient with respect to coeffs: grad_J_dc = basis.T @ grad_J_dx
    grad_J_dc = basis.T @ grad_J_dx
    
    return val / (s * s), grad_J_dc

def _optimize_legendre(max_degree, deadline, rng, beta=50.0):
    """
    Optimize the Legendre coefficients to minimize the smooth metric.
    """
    # Compute the basis
    from scipy.special import eval_legendre
    t = np.linspace(-1, 1, N)
    basis = np.zeros((N, max_degree + 1))
    for k in range(max_degree + 1):
        basis[:, k] = eval_legendre(k, t)
    
    # Orthonormalize the basis
    # Use QR decomposition for numerical stability
    Q, R = np.linalg.qr(basis)
    # Q has shape (N, max_degree+1) with orthonormal columns
    # But we only need the first max_degree+1 columns
    basis_orth = Q[:, :max_degree + 1]
    
    # Initialize coefficients randomly
    coeffs = rng.normal(0, 1, max_degree + 1)
    
    best_coeffs = coeffs.copy()
    best_val = _metric(_coeff_to_sequence(coeffs, basis_orth))
    
    def objective_and_grad(coeffs_flat):
        val, grad = _smooth_metric_and_grad_coeffs(coeffs_flat, basis_orth, beta=beta)
        return val, grad
    
    start_time = time.time()
    try:
        stop_flag = [False]
        
        def callback(xk):
            if time.time() > deadline - 5:
                stop_flag[0] = True
                return True
            return False
        
        res = minimize(
            objective_and_grad,
            coeffs,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 30},
            callback=callback
        )
        
        if res.x is not None:
            coeffs_new = res.x
            x_new = _coeff_to_sequence(coeffs_new, basis_orth)
            val_new = _metric(x_new)
            if val_new < best_val:
                best_val = val_new
                best_coeffs = coeffs_new.copy()
                
    except Exception as e:
        pass
    
    x_best = _coeff_to_sequence(best_coeffs, basis_orth)
    return x_best, best_val

def _try_local_search_fast(x, deadline, rng, max_rounds=20000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(x)
    no_improve = 0
    step_scale = 0.05
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 30:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        # Modify a random subset
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        
        # Try a few perturbations
        for trial in range(3):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    # Try different Legendre degrees
    degrees_to_try = [5, 10, 15, 20, 25, 30]
    
    time_per_attempt = 120.0 / len(degrees_to_try)
    
    for degree in degrees_to_try:
        if time.time() > deadline - 10:
            break
        
        attempt_deadline = min(deadline - 5, time.time() + time_per_attempt)
        
        x_leg, val_leg = _optimize_legendre(degree, attempt_deadline, rng)
        
        if val_leg < best_val:
            best_val = val_leg
            best_x = x_leg.copy()
    
    # Final polish with local search
    if best_x is not None and time.time() < deadline - 5:
        best_x, best_val = _try_local_search_fast(best_x, deadline, rng, max_rounds=5000)
    
    if best_x is None:
        return np.ones(N)
        
    return best_x
# EVOLVE-BLOCK-END
