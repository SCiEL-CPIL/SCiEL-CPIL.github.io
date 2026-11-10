# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
from scipy.integrate import fixed_quad
from scipy.optimize import differential_evolution, minimize
import time


def _compute_ratio(coeffs: np.ndarray) -> float:
    """
    Compute R = ||f||_4 / ||f||_2 for a degree-10 spherical harmonic
    with real coefficients coeffs ordered m = -10..10.
    
    Uses 70-point Gauss-Legendre in cos(theta) and 130 equally spaced phi.
    """
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    mu, w = np.polynomial.legendre.leggauss(n_theta)
    
    # Equally spaced phi
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    
    # Build the spherical harmonic values on the grid
    # f(theta, phi) = sum_m c_m Y_m(theta, phi)
    
    # Precompute cos(theta) and sin/cos of m*phi
    cos_theta = mu
    sin_theta = np.sqrt(np.maximum(0, 1 - mu**2))
    
    # For each m, compute P_10^m(cos_theta)
    # Y_0 = N_0 P_10^0
    # Y_m = sqrt(2) N_m P_10^m cos(m*phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m| sin(|m|*phi) for m < 0
    
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    
    # We'll compute f on the grid as a 2D array
    f_grid = np.zeros((n_theta, n_phi))
    
    for idx, m in enumerate(range(-k, k + 1)):
        c = coeffs[idx]
        if c == 0.0:
            continue
        
        abs_m = abs(m)
        
        # Compute P_10^abs_m(cos_theta) using Condon-Shortley phase
        # scipy.special.lpmv(m, n, x) includes the Condon-Shortley phase (-1)^m
        P = lpmv(abs_m, k, cos_theta)
        
        # Normalization constant
        # N_m = sqrt(21/(4*pi) * (10-abs_m)!/(10+abs_m)!)
        # Note: for real harmonics, the normalization uses (2l+1)/(4*pi) * (l-m)!/(l+m)!
        # Wait, let me re-read the task: N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
        # 21 = 2*10 + 1 = 2l+1, so this is correct.
        
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + 
                       np.log(factorial(k - abs_m, exact=False)) - 
                       np.log(factorial(k + abs_m, exact=False)))
        N = np.exp(log_N)
        
        if m == 0:
            Y = N * P
        elif m > 0:
            Y = np.sqrt(2.0) * N * P * np.cos(m * phi)
        else:
            Y = np.sqrt(2.0) * N * P * np.sin(abs_m * phi)
        
        f_grid += c * Y[np.newaxis, :]  # Y is shape (n_theta,) for m=0, or (n_theta, n_phi) for m!=0
        # Wait, for m != 0, Y has shape (n_theta, n_phi) already since phi is broadcast
        # Actually, let me be more careful. P has shape (n_theta,). 
        # For m > 0: cos(m*phi) has shape (n_phi,). So Y = P[:, None] * cos(m*phi)[None, :]
        # Let me redo this.
    
    # Let me redo the computation more carefully
    f_grid = np.zeros((n_theta, n_phi))
    
    for idx, m in enumerate(range(-k, k + 1)):
        c = coeffs[idx]
        if c == 0.0:
            continue
        
        abs_m = abs(m)
        
        # Compute P_10^abs_m(cos_theta)
        P = lpmv(abs_m, k, cos_theta)  # shape (n_theta,)
        
        # Normalization
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + 
                       np.log(factorial(k - abs_m, exact=False)) - 
                       np.log(factorial(k + abs_m, exact=False)))
        N = np.exp(log_N)
        
        if m == 0:
            Y = (N * P)[:, np.newaxis]  # shape (n_theta, 1), broadcast over phi
        elif m > 0:
            Y = (np.sqrt(2.0) * N * P)[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            Y = (np.sqrt(2.0) * N * P)[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
        
        f_grid += c * Y
    
    # Surface measure: dOmega = sin(theta) dtheta dphi = -d(mu) dphi
    # Since mu = cos(theta), dmu = -sin(theta) dtheta, so sin(theta) dtheta = -dmu
    # The integral over the sphere is:
    # ∫∫ f dOmega = ∫_0^{2π} ∫_{-1}^{1} f(mu, phi) dmu dphi
    # With Gauss-Legendre: ∑_i w_i f(mu_i, phi) for the mu integral
    # With equal spacing in phi: (2π/n_phi) ∑_j f(mu, phi_j)
    
    # So the surface integral of g(mu, phi) is:
    # (2π/n_phi) * ∑_j ∑_i w_i g(mu_i, phi_j)
    
    # ||f||_2^2 = (2π/n_phi) * ∑_j ∑_i w_i f(mu_i, phi_j)^2
    # ||f||_4^4 = (2π/n_phi) * ∑_j ∑_i w_i f(mu_i, phi_j)^4
    
    # Let's compute the weighted sums
    f2 = f_grid ** 2
    f4 = f_grid ** 4
    
    # Weight by Gauss-Legendre weights in mu
    # The quadrature for ∫_{-1}^{1} g(mu) dmu ≈ ∑_i w_i g(mu_i)
    # And ∫_0^{2π} h(phi) dphi ≈ (2π/n_phi) ∑_j h(phi_j)
    
    # So ||f||_2^2 ≈ (2π/n_phi) * ∑_j ∑_i w_i f(mu_i, phi_j)^2
    # = (2π/n_phi) * ∑_i w_i ∑_j f(mu_i, phi_j)^2
    
    sum_f2 = np.sum(w[:, np.newaxis] * f2, axis=0)  # shape (n_phi,)
    norm2_sq = (2 * np.pi / n_phi) * np.sum(sum_f2)
    
    sum_f4 = np.sum(w[:, np.newaxis] * f4, axis=0)  # shape (n_phi,)
    norm4_4 = (2 * np.pi / n_phi) * np.sum(sum_f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _objective(coeffs: np.ndarray) -> float:
    """Objective function to minimize: R = ||f||_4 / ||f||_2"""
    try:
        r = _compute_ratio(coeffs)
        if not np.isfinite(r):
            return 1e10
        return r
    except Exception:
        return 1e10


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses differential evolution for global search, then L-BFGS-B for local polish.
    """
    k = 10
    n = 2 * k + 1  # 21 coefficients
    
    # Deadline guard: 630s total, leave 30s margin
    t0 = time.time()
    deadline = t0 + 600.0
    
    # Initial best: all ones (baseline)
    best_coeffs = np.ones(n, dtype=np.float64)
    best_score = _compute_ratio(best_coeffs)
    
    # Define bounds for differential evolution
    # Coefficients should be in a reasonable range
    bounds = [(-10.0, 10.0)] * n
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-8,
            mutation=(0.3, 1.2),
            recombination=0.7,
            polish=False,
            callback=lambda x, convergence: None,
            init='latinhypercube',
            seed=42
        )
        
        if de_result.success or np.isfinite(de_result.fun):
            candidate = de_result.x
            score = _compute_ratio(candidate)
            if score < best_score:
                best_score = score
                best_coeffs = candidate.copy()
        
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 60:
        return best_coeffs
    
    # Local polish with L-BFGS-B
    try:
        result = minimize(
            _objective,
            best_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if np.isfinite(result.fun):
            score = _compute_ratio(result.x)
            if score < best_score:
                best_score = score
                best_coeffs = result.x.copy()
                
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 30:
        return best_coeffs
    
    # Try a few more random restarts if time permits
    rng = np.random.default_rng(123)
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random initialization in a reasonable range
        x0 = rng.uniform(-5, 5, size=n)
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if np.isfinite(result.fun):
                score = _compute_ratio(result.x)
                if score < best_score:
                    best_score = score
                    best_coeffs = result.x.copy()
                    
        except Exception:
            continue
    
    # Final validation
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(n, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = np.ones(n, dtype=np.float64)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5
    
    return best_coeffs


# EVOLVE-BLOCK-END
