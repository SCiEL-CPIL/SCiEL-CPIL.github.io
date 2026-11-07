"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Hermite polynomial constants (physicists' convention)
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0

def compute_q_y_coeffs(c0, c1, c2):
    """Compute coefficients of q(y) where y=x^2, degree 5.
    Returns [a0, a1, a2, a3, a4, a5] for q(y) = a0 + a1*y + ... + a5*y^5."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # q(y) coefficients (in y = x^2):
    a0 = c1 * (-48) + c2 * (-1344) + c3 * (-921600)
    a1 = c1 * 16 + c2 * 2688 + c3 * 1424640
    a2 = c2 * (-1344) + c3 * (-1003520)
    a3 = c2 * 256 + c3 * 332160
    a4 = c3 * (-46080)
    a5 = c3 * 4096
    
    return np.array([a0, a1, a2, a3, a4, a5]), c3

def evaluate_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) where r is the largest sign-changing root of q(x)."""
    coeffs, c3 = compute_q_y_coeffs(c0, c1, c2)
    
    # Check leading coefficient sign: if c3 < 0, g -> -inf, replace g by -g
    if c3 < 0:
        coeffs = -coeffs
        c3 = -c3
    
    if c3 <= 0:
        return 1e10
    
    # Find roots of the degree-5 polynomial in y
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    if len(real_roots) == 0:
        return 1e10
    
    # Find largest positive sign-changing root
    best_root = -np.inf
    for r in real_roots:
        if r <= 1e-12:
            continue
        eps = 1e-10 * max(1.0, abs(r))
        q_left = np.polyval(coeffs, r - eps)
        q_right = np.polyval(coeffs, r + eps)
        if q_left * q_right < 0:
            if r > best_root:
                best_root = r
    
    if best_root <= 0:
        return 1e10
    
    return best_root / (2.0 * np.pi)

def objective(params):
    """Objective function for optimization."""
    try:
        val = evaluate_value(params[0], params[1], params[2])
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 175  # Safety margin before 180s limit
    
    # Seed solution - this is the known best
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_params = seed.copy()
    best_value = evaluate_value(*seed)
    
    # Strategy 1: Very tight local refinement from seed
    try:
        result = minimize(
            objective, best_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
        )
        if time.time() < deadline - 20:
            if result.fun < best_value:
                best_params = result.x.copy()
                best_value = result.fun
    except Exception:
        pass
    
    # Strategy 2: Perturb around seed with multiple scales
    rng = np.random.RandomState(42)
    scales = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    for scale in scales:
        if time.time() > deadline - 10:
            break
        for _ in range(15):
            if time.time() > deadline - 5:
                break
            perturbation = rng.normal(0, scale, 3)
            x0 = best_params + perturbation
            try:
                result = minimize(
                    objective, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if result.fun < best_value:
                    best_params = result.x.copy()
                    best_value = result.fun
            except Exception:
                pass
    
    # Strategy 3: Final ultra-fine refinement
    if time.time() < deadline - 5:
        try:
            result = minimize(
                objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
            )
            if result.fun < best_value:
                best_params = result.x.copy()
                best_value = result.fun
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
