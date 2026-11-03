"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

# Precompute Hermite polynomials at quadrature nodes for speed
_N_QUAD = 100
_X_NODES, _X_WEIGHTS = np.polynomial.hermite.hermgauss(_N_QUAD)

# Precompute Hermite polynomial values and derivatives at quadrature nodes
def _precompute_hermite():
    """Precompute H_n and H_n' at quadrature nodes for n=0,3,4,7,8,11,12."""
    x = _X_NODES
    H = {}
    H[0] = np.ones_like(x)
    H[1] = 2 * x
    for k in range(1, 12):
        H[k+1] = 2 * x * H[k] - 2 * k * H[k-1]
    # Derivatives: H_n'(x) = 2n * H_{n-1}(x)
    Hp = {}
    Hp[0] = np.zeros_like(x)
    for n in [4, 8, 12]:
        Hp[n] = 2 * n * H[n-1]
    return H, Hp

_H_VALS, _HP_VALS = _precompute_hermite()

# H_n(0) values for physicists' Hermite polynomials
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1344.0
_H12_0 = 199584.0

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product A(f)*A(F(f)) for the Hermite-Gaussian test function.
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2), c12 chosen so g(0)=0.
    Returns the product (Δx)²·(Δξ)² using the convention that makes seed score 1.0.
    """
    c12 = -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0
    
    P_vals = (c0 * _H_VALS[0] + c4 * _H_VALS[4] + c8 * _H_VALS[8] + c12 * _H_VALS[12])
    P_prime = (c0 * _HP_VALS[0] + c4 * _HP_VALS[4] + c8 * _HP_VALS[8] + c12 * _HP_VALS[12])
    
    norm_sq = np.dot(_X_WEIGHTS, P_vals * P_vals)
    if norm_sq < 1e-15:
        return 1e10
    
    x2_norm_sq = np.dot(_X_WEIGHTS, _X_NODES**2 * P_vals * P_vals)
    g_prime = P_prime - _X_NODES * P_vals
    deriv_norm_sq = np.dot(_X_WEIGHTS, g_prime * g_prime)
    
    # Uncertainty product: (Δx)² · (Δξ)²
    # Using convention: Δξ² = ||g'||² / (4π² ||g||²)
    delta_x_sq = x2_norm_sq / norm_sq
    delta_xi_sq = deriv_norm_sq / (4.0 * np.pi**2 * norm_sq)
    
    return delta_x_sq * delta_xi_sq

def _objective_2d(theta, phi):
    """
    Objective in spherical coordinates. The uncertainty product is scale-invariant,
    so we fix ||c|| = 1 and parametrize the direction by two angles.
    """
    c0 = math.sin(theta) * math.cos(phi)
    c4 = math.sin(theta) * math.sin(phi)
    c8 = math.cos(theta)
    return _compute_bound(c0, c4, c8)

def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the uncertainty product bound.
    Uses Differential Evolution on spherical parameters (removing scale ambiguity),
    followed by local Cartesian polish.
    """
    start_time = time.time()
    deadline = start_time + 195  # Safety margin before 210s limit
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_val = _compute_bound(seed[0], seed[1], seed[2])
    
    # Convert seed direction to spherical coordinates
    seed_norm = np.linalg.norm(seed)
    if seed_norm > 1e-15:
        s = seed / seed_norm
        theta_seed = math.acos(max(-1.0, min(1.0, s[2])))
        phi_seed = math.atan2(s[1], s[0])
    else:
        theta_seed = math.pi / 2
        phi_seed = 0.0
    
    best_val = seed_val
    best_coeffs = seed.copy()
    
    # Phase 1: Differential Evolution on (theta, phi)
    bounds = [(0.0, math.pi), (0.0, 2.0 * math.pi)]
    try:
        de_result = differential_evolution(
            _objective_2d,
            bounds,
            seed=42,
            maxiter=300,
            tol=1e-14,
            popsize=15,
            polish=False,
            init='sobol'
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            theta_best, phi_best = de_result.x
            c0 = math.sin(theta_best) * math.cos(phi_best)
            c4 = math.sin(theta_best) * math.sin(phi_best)
            c8 = math.cos(theta_best)
            best_coeffs = np.array([c0, c4, c8]) * seed_norm
    except Exception:
        pass
    
    if time.time() - start_time > deadline - 20:
        return best_coeffs
    
    # Phase 2: Local polish in Cartesian space
    try:
        res = minimize(
            lambda c: _compute_bound(c[0], c[1], c[2]),
            best_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-16}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    # Phase 3: Additional random restarts in Cartesian
    if time.time() - start_time < deadline - 10:
        rng = np.random.RandomState(7)
        for _ in range(8):
            if time.time() - start_time > deadline - 5:
                break
            # Random direction, scaled to seed magnitude
            v = rng.normal(size=3)
            nrm = np.linalg.norm(v)
            if nrm < 1e-15:
                continue
            v = v / nrm * seed_norm
            try:
                res = minimize(
                    lambda c: _compute_bound(c[0], c[1], c[2]),
                    v,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                lambda c: _compute_bound(c[0], c[1], c[2]),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    if time.time() - start_time > deadline:
        return seed
    
    return best_coeffs
# EVOLVE-BLOCK-END
