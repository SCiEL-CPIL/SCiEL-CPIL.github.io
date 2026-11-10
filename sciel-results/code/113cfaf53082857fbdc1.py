# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
warnings.filterwarnings('ignore')

def _build_Y_table(x_gl):
    """
    Build the full spherical harmonic table for degree 10.
    Returns an array Y_table of shape (21, 70, 130) where
    Y_table[m+10, i, j] = Y_m(theta_i, phi_j)
    """
    k = 10
    n_phi = 130
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    Y_table = np.zeros((21, len(x_gl), n_phi))
    
    for m in range(-k, k + 1):
        idx = m + k
        abs_m = abs(m)
        P = lpmv(abs_m, k, x_gl)
        n_fact = factorial(k - abs_m, exact=True)
        d_fact = factorial(k + abs_m, exact=True)
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * (n_fact / d_fact))
        
        if m == 0:
            Y_table[idx] = N_m * P[:, None]
        elif m > 0:
            Y_table[idx] = np.sqrt(2) * N_m * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y_table[idx] = np.sqrt(2) * N_m * P[:, None] * np.sin(abs_m * phi)[None, :]
    
    return Y_table

def _compute_ratio(coeffs, Y_table, w_theta, dphi):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed Y_table.
    coeffs: length 21 vector
    Returns: ratio R (lower is better)
    """
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        return np.inf
    
    # f_val = sum_m c_m * Y_table[m+10]
    # This is a matrix-vector product: (21, 70*130) @ (21,) -> (70*130,)
    # Reshape to (70, 130) for computation
    f_flat = coeffs @ Y_table.reshape(21, -1)
    f_val = f_flat.reshape(Y_table.shape[1], Y_table.shape[2])
    
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.sum(f2 * w_theta[:, None]) * dphi
    norm4_4 = np.sum(f4 * w_theta[:, None]) * dphi
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2

def _compute_ratio_log(coeffs, Y_table, w_theta, dphi):
    """
    Log of the ratio for optimization stability.
    """
    r = _compute_ratio(coeffs, Y_table, w_theta, dphi)
    if r <= 0 or np.isinf(r) or np.isnan(r):
        return 1e10
    return np.log(r)

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients for degree-10 spherical harmonic minimizing ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = 600  # seconds, with safety margin
    
    k = 10
    n_coeffs = 2 * k + 1
    
    # Precompute quadrature and Y_table
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    dphi = 2 * np.pi / 130
    Y_table = _build_Y_table(x_gl)
    
    # Precompute for fast evaluation
    w_theta = w_gl
    
    def objective(c):
        return _compute_ratio(c, Y_table, w_theta, dphi)
    
    # Initial baseline
    c0 = np.ones(n_coeffs)
    best_coeffs = c0.copy()
    best_ratio = objective(c0)
    
    # Strategy:
    # 1. Differential Evolution (global) to explore landscape
    # 2. L-BFGS-B (local) from best points
    # 3. More L-BFGS-B restarts if time permits
    
    # Since the objective is scale-invariant, we should normalize.
    # But L-BFGS-B and DE handle this reasonably well if we don't fix scale.
    # However, DE explores all scales. Let's see.
    
    # 1. Differential Evolution
    # Bounds: Since scale is invariant, we can restrict to ||c||_2 = 1 sphere.
    # DE works on boxes. We can map to box and normalize?
    # Or just use a wide box like [-10, 10] and normalize inside objective?
    # No, DE will try to move in radial direction.
    # Better: Use a constraint ||c||_2 = 1?
    # scipy.optimize.differential_evolution doesn't support constraints directly.
    
    # Alternative: Parameterize on the sphere?
    # Or just use L-BFGS-B with normalization trick?
    # Let's try DE with bounds [-1, 1] and normalize inside objective?
    # If we normalize, DE gradient info is lost for the radial component.
    # But DE is gradient-free. It might work.
    
    # Let's try a different approach:
    # Use L-BFGS-B with a soft normalization penalty?
    # Or just trust L-BFGS-B to handle the flat direction?
    # The current code uses L-BFGS-B without normalization and it works (score 1.0).
    
    # Let's try DE on the normalized sphere implicitly by normalizing in objective.
    def objective_norm(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-15:
            return 1e10
        c_norm = c / norm_c
        return objective(c_norm)
    
    # DE bounds
    bounds = [(-1.0, 1.0)] * n_coeffs
    
    # Run DE
    try:
        if time.time() < deadline - 120:
            de_res = differential_evolution(
                objective_norm,
                bounds,
                maxiter=20,  # Keep low to save time, DE is slow
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False  # We'll polish manually
            )
            c_de = de_res.x
            # Normalize and evaluate
            norm_de = np.linalg.norm(c_de)
            if norm_de > 1e-15:
                c_de = c_de / norm_de
                r_de = objective(c_de)
                if r_de < best_ratio:
                    best_ratio = r_de
                    best_coeffs = c_de.copy()
    except Exception:
        pass
    
    # 2. L-BFGS-B from multiple starts
    # Starts: best_coeffs, c0, random points
    rng = np.random.default_rng(42)
    
    starts = [best_coeffs.copy(), c0.copy()]
    # Add some random starts
    for _ in range(10):
        c_rand = rng.normal(size=n_coeffs)
        norm_rand = np.linalg.norm(c_rand)
        if norm_rand > 1e-15:
            starts.append(c_rand / norm_rand)
    
    # Also add starts with some structure?
    # e.g., only m=0, only m=10, etc.
    for m in [-10, -5, 0, 5, 10]:
        c_struct = np.zeros(n_coeffs)
        c_struct[m + 10] = 1.0
        starts.append(c_struct)
    
    # Also try pairs
    for m1 in [-10, -5, 0, 5, 10]:
        for m2 in [-10, -5, 0, 5, 10]:
            if m1 != m2:
                c_pair = np.zeros(n_coeffs)
                c_pair[m1 + 10] = 1.0
                c_pair[m2 + 10] = 1.0
                starts.append(c_pair)
    
    # Deduplicate starts (approx)
    unique_starts = []
    for s in starts:
        if np.linalg.norm(s) < 1e-10:
            continue
        s = s / np.linalg.norm(s)
        # Check if similar to existing
        is_dup = False
        for u in unique_starts:
            if np.linalg.norm(s - u) < 1e-3 or np.linalg.norm(s + u) < 1e-3:
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)
    
    # Run L-BFGS-B from each start
    for i, c_start in enumerate(unique_starts):
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(
                objective,
                c_start,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_opt = res.x
            # Normalize? The objective is scale-invariant, but L-BFGS-B might drift.
            # Let's evaluate as is.
            r_opt = objective(c_opt)
            
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass
    
    # 3. If time permits, do more random restarts
    n_extra = 20
    for i in range(n_extra):
        if time.time() > deadline - 30:
            break
        
        c_rand = rng.normal(size=n_coeffs)
        norm_rand = np.linalg.norm(c_rand)
        if norm_rand < 1e-15:
            continue
        c_start = c_rand / norm_rand
        
        try:
            res = minimize(
                objective,
                c_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_opt = res.x
            r_opt = objective(c_opt)
            
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass
    
    # Final validation
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = c0
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            best_coeffs = c0
        else:
            best_coeffs = best_coeffs / max_abs * 1.0
            if np.max(np.abs(best_coeffs)) < 1e-3:
                best_coeffs = c0
    
    return best_coeffs.astype(np.float64)

# EVOLVE-BLOCK-END
