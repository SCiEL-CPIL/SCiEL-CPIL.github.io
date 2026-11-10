# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import differential_evolution

K = 10
N_THETA = 70
N_PHI = 130

# Precompute quadrature weights and nodes
_x, _w = np.polynomial.legendre.leggauss(N_THETA)
_cos_theta = _x
_theta_weights = _w * 0.5

# phi grid
_phi = np.arange(N_PHI) * (2 * np.pi / N_PHI)

# Precompute associated Legendre functions P_l^m(cos theta) with Condon-Shortley phase
_P = {}
for m in range(K + 1):
    _P[m] = lpmv(m, K, _cos_theta)  # shape (N_THETA,)

# Normalization constants N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
def _norm_const(m):
    m = abs(m)
    val = 21.0 / (4.0 * np.pi) * float(factorial(10 - m) / factorial(10 + m))
    return np.sqrt(val)

# Build the full coefficient matrix: shape (2K+1, N_THETA, N_PHI)
_basis = np.zeros((2 * K + 1, N_THETA, N_PHI), dtype=np.float64)
for idx, m in enumerate(range(-K, K + 1)):
    am = abs(m)
    norm = _norm_const(am)
    Pm = _P[am]  # shape (N_THETA,)
    if m == 0:
        _basis[idx, :, :] = norm * Pm[:, None]
    elif m > 0:
        cos_term = np.cos(m * _phi)  # shape (N_PHI,)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * cos_term[None, :]
    else:
        sin_term = np.sin(am * _phi)  # shape (N_PHI,)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * sin_term[None, :]

# Quadrature weights on the sphere
_w_phi = 2.0 * np.pi / N_PHI
_sphere_weights = np.outer(_theta_weights, np.full(N_PHI, _w_phi))  # shape (N_THETA, N_PHI)
_sphere_weights_flat = _sphere_weights.ravel()  # shape (N_THETA * N_PHI,)

# Precompute basis as flat matrix: shape (N_POINTS, 2K+1)
# f(point) = B(point, :) @ c
_B_flat = _basis.transpose(1, 2, 0).reshape(-1, 2 * K + 1)  # shape (N_POINTS, 21)
N_POINTS = N_THETA * N_PHI

def _compute_R(coeffs):
    """Compute ||f||_4 / ||f||_2 for given coefficients."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.shape != (2 * K + 1,):
        return np.inf
    if not np.all(np.isfinite(coeffs)):
        return np.inf
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return np.inf
    
    f = _B_flat @ coeffs  # shape (N_POINTS,)
    
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq <= 0:
        return np.inf
    l2_norm = np.sqrt(l2_sq)
    
    l4_pow4 = np.sum(np.abs(f) ** 4 * _sphere_weights_flat)
    if l4_pow4 <= 0:
        return np.inf
    l4_norm = l4_pow4 ** 0.25
    
    return l4_norm / l2_norm

def _objective_de(coeffs):
    """
    Objective for Differential Evolution.
    We minimize R^4 = (sum w f^4) / (sum w f^2)^2.
    Since the problem is scale-invariant, we can fix the scale during evaluation 
    or just minimize the ratio directly. 
    However, DE works better with bounded variables. 
    We map the 21 coefficients to a bounded space [-1, 1] and scale them internally 
    to satisfy the validity constraints if needed, but primarily we rely on the 
    scale invariance of the ratio.
    
    Actually, it's safer to minimize R directly to avoid numerical issues with 
    large/small scales, but R is non-convex.
    Let's stick to minimizing R.
    """
    if not np.all(np.isfinite(coeffs)):
        return 1e10
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # If out of bounds, return a penalty
        return 1e10
        
    f = _B_flat @ coeffs
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq <= 0:
        return 1e10
    l4_pow4 = np.sum(np.abs(f) ** 4 * _sphere_weights_flat)
    if l4_pow4 <= 0:
        return 1e10
        
    R = (l4_pow4 ** 0.25) / (l2_sq ** 0.5)
    return R

def rudin_problem10() -> np.ndarray:
    deadline = time.time() + 590  # Safety margin
    start_time = time.time()
    
    # Define bounds for DE. 
    # The coefficients can be any real number. 
    # Since the ratio is scale invariant, we can restrict the search to a sphere 
    # or a box. A box of [-1, 1] is good for DE, but we might miss solutions 
    # with larger norms? No, because we can scale. 
    # However, if we restrict to [-1, 1], the optimizer might find a solution 
    # with max|c| < 1e-3. We must ensure the final output is valid.
    # Strategy: Search in [-1, 1]^21. If the best solution has max|c| < 1e-3, 
    # scale it up. If max|c| > 1e5 (unlikely with bounds), scale down.
    
    bounds = [(-1.0, 1.0)] * (2 * K + 1)
    
    # Initial population strategy:
    # DE handles initialization. We can provide a seed or let it generate.
    # Let's use a moderate population size.
    popsize = 20
    
    # Time management:
    # We will run DE for a fixed number of iterations or until time runs out.
    # DE in scipy does not have a direct time callback in older versions, 
    # but we can check time in the objective function? No, that's bad practice.
    # Instead, we estimate the time per iteration and set maxiter accordingly.
    
    # Estimate time per function evaluation
    # f = B @ c is 9100 x 21 matvec. Very fast.
    # Let's assume 1ms per evaluation.
    # DE with popsize=20, 21 dims. 
    # Let's run for up to 400 seconds.
    
    best_coeffs = None
    best_R = np.inf
    
    try:
        # Run Differential Evolution
        # tol=1e-8, maxiter=10000 (will be cut off by time if we implement a wrapper)
        # But scipy.optimize.differential_evolution doesn't support early stopping by time directly in the main loop easily without a custom callback.
        # We can use a callback to check time.
        
        def _callback(xk, convergence):
            if time.time() > deadline - 10:
                return True
            return False

        result = differential_evolution(
            _objective_de,
            bounds,
            maxiter=5000,  # High number, will be cut by callback
            popsize=popsize,
            tol=1e-8,
            seed=42,
            callback=_callback,
            polish=True  # Use L-BFGS-B to polish the result, which helps with local minima
        )
        
        if result.success or result.fun < best_R:
            best_coeffs = result.x
            best_R = result.fun
            
    except Exception as e:
        pass
        
    # If DE didn't run or failed, fallback to a simple grid/random search if time permits
    if best_coeffs is None:
        # Fallback: Random search
        rng = np.random.default_rng(42)
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            c = rng.uniform(-1, 1, size=2 * K + 1)
            R = _compute_R(c)
            if R < best_R:
                best_R = R
                best_coeffs = c.copy()
    else:
        # Refine with local optimization from the DE result
        # Use L-BFGS-B from scipy.optimize.minimize
        from scipy.optimize import minimize
        
        # We need to ensure the starting point is valid for minimize
        # Normalize to avoid extreme values if needed, though DE bounds keep it in [-1,1]
        x0 = best_coeffs.copy()
        
        # Check validity of x0
        max_abs = np.max(np.abs(x0))
        if max_abs < 1e-3:
            # Scale up to be safe for local optimizer
            x0 = x0 / max_abs * 0.1 if max_abs > 0 else np.ones(2*K+1)
            
        try:
            res = minimize(
                _objective_de, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.fun < best_R:
                best_coeffs = res.x
                best_R = res.fun
        except Exception:
            pass

    # Final Validation and Scaling
    if best_coeffs is None:
        # Absolute fallback
        best_coeffs = np.ones(2 * K + 1)
        best_R = _compute_R(best_coeffs)
        
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    
    # Ensure finite
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(2 * K + 1)
        
    # Ensure max|c| in [1e-3, 1e5]
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs * 1e-2
        else:
            best_coeffs = np.ones(2 * K + 1)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5
        
    return best_coeffs
# EVOLVE-BLOCK-END
