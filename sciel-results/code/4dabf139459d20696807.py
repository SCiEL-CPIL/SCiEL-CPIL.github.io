"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, sqrt, pi
from scipy.optimize import differential_evolution, minimize

def _hermite_val(n, x):
    """Compute physicists' Hermite polynomial H_n(x) at scalar x."""
    if n == 0:
        return 1.0
    elif n == 1:
        return 2.0 * x
    H_prev = 1.0
    H_curr = 2.0 * x
    for k in range(2, n + 1):
        H_next = 2.0 * x * H_curr - 2.0 * (k - 1) * H_prev
        H_prev = H_curr
        H_curr = H_next
    return H_curr

# Basis indices: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Precompute H_n(0) for n in indices
H_at_0 = np.array([_hermite_val(n, 0.0) for n in indices])

# c12 is determined by g(0) = 0: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
H12_0 = H_at_0[3]

# Norm factors: ||H_n * exp(-x^2/2)||^2 = 2^n * n! * sqrt(pi)
norms_sq = np.array([(2.0**n) * factorial(n) * sqrt(pi) for n in indices])

def _compute_uncertainty_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for given coefficients.
    Returns <x^2> which equals the uncertainty product for this even function.
    """
    c12 = -(c0 * H_at_0[0] + c4 * H_at_0[1] + c8 * H_at_0[2]) / H12_0
    c = np.array([c0, c4, c8, c12], dtype=float)
    
    norm_sq = np.sum(c**2 * norms_sq)
    if norm_sq < 1e-15:
        return 1e10
    
    # <x^2> = sum c_n^2 * norms_sq[n] * (n + 0.5) / norm_sq
    x2_vals = np.array([n + 0.5 for n in indices])
    x2 = np.sum(c**2 * norms_sq * x2_vals) / norm_sq
    
    return x2

def _objective_spherical(theta, phi):
    """
    Objective in spherical coordinates.
    theta in [0, pi], phi in [0, 2*pi]
    c0 = cos(theta)
    c4 = sin(theta) * cos(phi)
    c8 = sin(theta) * sin(phi)
    """
    c0 = np.cos(theta)
    c4 = np.sin(theta) * np.cos(phi)
    c8 = np.sin(theta) * np.sin(phi)
    return _compute_uncertainty_bound(c0, c4, c8)

def construct():
    """Find optimal coefficients for H0, H4, H8 using spherical coordinates and DE."""
    start_time = time.time()
    deadline = start_time + 165.0  # Safety margin
    
    best_c = np.array([1.0, 0.0, 0.0])
    best_score = _compute_uncertainty_bound(1.0, 0.0, 0.0)
    
    # Phase 1: Global search with Differential Evolution in spherical coordinates
    try:
        bounds_spherical = [(0.0, np.pi), (0.0, 2.0 * np.pi)]
        
        result = differential_evolution(
            _objective_spherical,
            bounds=bounds_spherical,
            maxiter=300,
            popsize=20,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        if result.fun < best_score and time.time() < deadline:
            theta, phi = result.x
            c0 = np.cos(theta)
            c4 = np.sin(theta) * np.cos(phi)
            c8 = np.sin(theta) * np.sin(phi)
            best_c = np.array([c0, c4, c8])
            best_score = result.fun
    except Exception:
        pass
    
    # Phase 2: Local polish with L-BFGS-B in Cartesian coordinates
    if time.time() < deadline - 10.0:
        try:
            result_local = minimize(
                _compute_uncertainty_bound,
                x0=best_c,
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result_local.fun < best_score:
                best_c = np.array(result_local.x)
                best_score = result_local.fun
        except Exception:
            pass
    
    # Phase 3: Try multiple random restarts in Cartesian if time allows
    if time.time() < deadline - 5.0:
        rng = np.random.RandomState(42)
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            try:
                c_rand = rng.uniform(-2.0, 2.0, size=3)
                score_rand = _compute_uncertainty_bound(*c_rand)
                if score_rand < best_score:
                    best_score = score_rand
                    best_c = c_rand.copy()
                    # Quick local polish from this point
                    if time.time() < deadline - 3.0:
                        try:
                            res = minimize(
                                _compute_uncertainty_bound,
                                x0=best_c,
                                method='L-BFGS-B',
                                bounds=[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
                                options={'maxiter': 100, 'ftol': 1e-14}
                            )
                            if res.fun < best_score:
                                best_score = res.fun
                                best_c = np.array(res.x)
                        except Exception:
                            pass
            except Exception:
                pass
    
    return best_c

# EVOLVE-BLOCK-END
