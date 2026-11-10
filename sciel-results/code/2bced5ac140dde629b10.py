import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import minimize
import time

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi

def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
    """
    Build the matrix Y[i,j] where i indexes quadrature points (n_gl * n_phi)
    and j indexes m = -k..k.
    """
    n_gl = len(x_gl)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_gl * n_phi, n_m), dtype=np.float64)
    
    for m_idx in range(n_m):
        m = m_idx - k
        abs_m = abs(m)
        # Normalization
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        # Associated Legendre with Condon-Shortley phase
        P = lpmv(abs_m, k, x_gl)  # shape (n_gl,)
        
        if m == 0:
            vals = N_m * P  # shape (n_gl,)
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
    
    return Y

def _compute_ratio(coeffs, Y, weights):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    f = Y @ coeffs
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    if norm2_sq <= 1e-15:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    # ||f||_4 = (sum w |f|^4)^(1/4)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    # Strategy: Exploit rotational symmetry.
    # The "flattest" function is expected to be axisymmetric or have high symmetry.
    # Specifically, we restrict the search to coefficients where:
    # 1. All sine terms (m < 0) are zero.
    # 2. Cosine terms (m > 0) are free, but we can further exploit that 
    #    for a fixed axis, the optimal solution might have specific symmetries.
    # However, a stronger constraint often used in such "flattest" problems is that
    # the function is invariant under rotation around the z-axis, meaning only m=0 survives?
    # No, that would be a single mode, which has a fixed ratio.
    # The ratio ||f||_4 / ||f||_2 is invariant under global rotation.
    # If we fix the coordinate system, we can rotate the function f to align its principal axis with z.
    # For a generic f, this breaks the m=0 only constraint.
    # BUT, there is a known result that for "flattest" spherical harmonics, the optimal
    # configuration often involves only even m modes (if we consider symmetry under theta -> pi - theta)
    # and potentially specific combinations.
    
    # Let's try a different reduction: 
    # The problem is scale invariant. We can fix the norm ||f||_2 = 1.
    # We want to minimize ||f||_4.
    # This is equivalent to minimizing the 4th moment.
    # 
    # A key insight from similar problems (e.g., flattest harmonic) is that the optimal
    # function often has a high degree of symmetry. 
    # Let's assume the optimal function is axisymmetric (m=0 only)? 
    # If m=0 only, f = c_0 Y_0. The ratio is fixed for that mode.
    # Let's calculate the ratio for pure m=0.
    # Y_0 = N_0 P_10(cos theta).
    # P_10(x) has 10 roots. It oscillates.
    # The ratio for a single mode is constant. 
    # By adding other modes, we can flatten the peaks.
    
    # Let's restrict to a subspace that is likely to contain the global minimum.
    # Common symmetry: invariance under phi -> -phi (so no sine terms, m<0 = 0).
    # This leaves 11 dimensions (m = -10..10 with m<0=0, so m=0..10? No, m=-10..10, 
    # if we set sine=0, we keep m=0, 1..10 cosines. That's 11 dims).
    
    # Even stronger: invariance under theta -> pi - theta (cos theta -> -cos theta).
    # P_l^m(-x) = (-1)^(l-m) P_l^m(x).
    # For l=10, if m is even, l-m is even, so P is even.
    # If m is odd, l-m is odd, so P is odd.
    # If we restrict to even m (0, 2, 4, 6, 8, 10), the function is even in z.
    # This reduces the dimension to 6 (if we also drop sines).
    # Let's try optimizing over these 6 coefficients: m = 0, 2, 4, 6, 8, 10.
    # The indices in the 21-vector are:
    # m=0 -> idx 10
    # m=2 -> idx 12
    # m=4 -> idx 14
    # m=6 -> idx 16
    # m=8 -> idx 18
    # m=10 -> idx 20
    
    active_m = [0, 2, 4, 6, 8, 10]
    active_indices = [m + 10 for m in active_m] # m=0->10, m=2->12, etc.
    
    # Build a reduced matrix Y_reduced containing only these columns
    Y_red = Y[:, active_indices]
    
    def objective_reduced(coeffs_red):
        # Map reduced coeffs to full coeffs
        coeffs_full = np.zeros(21)
        for i, idx in enumerate(active_indices):
            coeffs_full[idx] = coeffs_red[i]
        return _compute_ratio(coeffs_full, Y, weights)
    
    best_coeffs_red = None
    best_val = np.inf
    
    # Grid search over the 6-dimensional space? 
    # 6 dimensions is still high for a dense grid.
    # Let's use Differential Evolution on the 6-dimensional space.
    # It should be much faster and more reliable than 21D.
    
    bounds = [(-10.0, 10.0) for _ in range(6)]
    
    try:
        from scipy.optimize import differential_evolution
        res = differential_evolution(
            objective_reduced, 
            bounds, 
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            polish=True
        )
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs_red = res.x.copy()
            
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random restarts
        rng = np.random.default_rng(42)
        for i in range(50):
            x0 = rng.normal(size=6)
            x0 /= np.linalg.norm(x0)
            res = minimize(objective_reduced, x0, method='L-BFGS-B', options={'maxiter': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs_red = res.x.copy()
            if time.time() > deadline - 30.0:
                break

    # If DE didn't run or failed, try a few random restarts with L-BFGS-B
    if best_coeffs_red is None or not np.isfinite(best_val):
        rng = np.random.default_rng(42)
        for i in range(50):
            x0 = rng.normal(size=6)
            x0 /= np.linalg.norm(x0)
            res = minimize(objective_reduced, x0, method='L-BFGS-B', options={'maxiter': 200})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs_red = res.x.copy()
            if time.time() > deadline - 30.0:
                break

    # Final polish
    if best_coeffs_red is not None and time.time() < deadline - 10.0:
        res = minimize(objective_reduced, best_coeffs_red, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        if res.fun < best_val:
            best_coeffs_red = res.x
            best_val = res.fun

    # Map back to full 21 coefficients
    best_coeffs_full = np.zeros(21)
    if best_coeffs_red is not None:
        for i, idx in enumerate(active_indices):
            best_coeffs_full[idx] = best_coeffs_red[i]
    else:
        # Fallback to m=0 only if something went wrong
        best_coeffs_full[10] = 1.0

    # Validate
    best_coeffs_full = np.asarray(best_coeffs_full, dtype=np.float64)
    if len(best_coeffs_full) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs_full)) or np.any(np.isinf(best_coeffs_full)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs_full))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Normalize to valid range if possible
        if max_abs > 0:
            best_coeffs_full = best_coeffs_full / max_abs * 1.0
        else:
            return np.ones(21)
            
    return best_coeffs_full
