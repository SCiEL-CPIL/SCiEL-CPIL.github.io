"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from math import factorial, sqrt, pi

# Pre-compute constants for analytic evaluation
# Indices in basis: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Norms of Hermite polynomials: <H_n, H_n> = sqrt(pi) * 2^n * n!
norms = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])

# Diagonal elements of <x^2 H_i, H_i> = (i + 0.5) * <H_i, H_i>
x2_diag = np.array([(n + 0.5) * norm for n, norm in zip(indices, norms)])

# Values of H_n(0) for n in indices to determine c12
# H_0(0) = 1
# H_4(0) = 12
# H_8(0) = 1680
# H_12(0) = 665280
H_vals_at_0 = np.array([1.0, 12.0, 1680.0, 665280.0])

# We need the norms for the odd Hermite polynomials that appear in Q(x) = P' - xP
# Q(x) coefficients d_k for H_k where k is odd: 1, 3, 5, 7, 9, 11, 13
odd_indices = [1, 3, 5, 7, 9, 11, 13]
odd_norms = np.array([sqrt(pi) * (2**k) * factorial(k) for k in odd_indices])

def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the uncertainty product bound A(f)*A(F) analytically.
    """
    # c12 is determined by g(0) = 0 => c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    c12 = -(c0 * H_vals_at_0[0] + c4 * H_vals_at_0[1] + c8 * H_vals_at_0[2]) / H_vals_at_0[3]
    
    coeffs = np.array([c0, c4, c8, c12])
    
    # 1. Calculate Norm N = sum c_m^2 * <H_m, H_m>
    norm_sq = np.sum(coeffs**2 * norms)
    if norm_sq < 1e-15:
        return 1e15
        
    # 2. Calculate <x^2> numerator = sum c_m^2 * (m+0.5) * <H_m, H_m>
    x2_num = np.sum(coeffs**2 * x2_diag)
    
    # 3. Calculate <p^2> numerator = sum d_k^2 * <H_k, H_k> for odd k
    # d_k are coefficients of H_k in Q(x) = P'(x) - x P(x)
    # P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # Q(x) = c0(-0.5 H1) + c4(4 H3 - 0.5 H5) + c8(8 H7 - 0.5 H9) + c12(12 H11 - 0.5 H13)
    
    d1 = -0.5 * c0
    d3 = 4.0 * c4
    d5 = -0.5 * c4
    d7 = 8.0 * c8
    d9 = -0.5 * c8
    d11 = 12.0 * c12
    d13 = -0.5 * c12
    
    d_coeffs = np.array([d1, d3, d5, d7, d9, d11, d13])
    p2_num = np.sum(d_coeffs**2 * odd_norms)
    
    # Variance Delta_x^2 = <x^2> - <x>^2. <x>=0 by symmetry.
    delta_x_sq = x2_num / norm_sq
    delta_p_sq = p2_num / norm_sq
    
    # Uncertainty product
    return np.sqrt(delta_x_sq * delta_p_sq)

def _compute_bound_2d(c4, c8):
    """
    Compute the uncertainty product with c0 fixed to 1 (scale invariance).
    """
    return _compute_bound_analytic(1.0, c4, c8)

def construct():
    start_time = time.time()
    deadline = start_time + 150.0  # Safety margin for 180s limit
    
    # Initial guess from seed (normalized to c0=1)
    seed_x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    # Normalize: divide by c0 to get ratios
    c0_seed = seed_x0[0]
    c4_seed = seed_x0[1] / c0_seed
    c8_seed = seed_x0[2] / c0_seed
    
    # We'll optimize in 2D: (c4, c8) with c0=1
    # The optimal solution has a specific scale, but since the objective is scale-invariant,
    # we can find the optimal ratios and then reconstruct.
    
    best_2d = np.array([c4_seed, c8_seed])
    best_val_2d = _compute_bound_2d(*best_2d)
    
    # Strategy: Use differential evolution in 2D (much faster), then BFGS polish
    bounds_2d = [(-10.0, 10.0), (-10.0, 10.0)]
    
    # Phase 1: Global search with DE in 2D
    try:
        result = differential_evolution(
            _compute_bound_2d,
            bounds_2d,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-10,
            atol=1e-12,
            init='sobol'
        )
        if result.fun < best_val_2d:
            best_val_2d = result.fun
            best_2d = result.x
    except Exception:
        pass
    
    # Phase 2: Local refinement with BFGS (uses gradient via finite differences)
    if time.time() < deadline:
        try:
            result2 = minimize(
                _compute_bound_2d,
                best_2d,
                method='BFGS',
                options={'maxiter': 10000, 'gtol': 1e-12, 'disp': False}
            )
            if result2.fun < best_val_2d:
                best_val_2d = result2.fun
                best_2d = result2.x
        except Exception:
            pass
        
        # Phase 3: Try multiple starting points with BFGS
        if time.time() < deadline:
            rng = np.random.default_rng(123)
            for i in range(20):
                if time.time() > deadline:
                    break
                # Random starting point in 2D
                x_start = rng.uniform(-5, 5, size=2)
                try:
                    res = minimize(
                        _compute_bound_2d,
                        x_start,
                        method='BFGS',
                        options={'maxiter': 5000, 'gtol': 1e-12, 'disp': False}
                    )
                    if res.fun < best_val_2d:
                        best_val_2d = res.fun
                        best_2d = res.x
                except Exception:
                    pass
        
        # Phase 4: Final high-precision polish
        if time.time() < deadline:
            try:
                result4 = minimize(
                    _compute_bound_2d,
                    best_2d,
                    method='BFGS',
                    options={'maxiter': 50000, 'gtol': 1e-15, 'disp': False}
                )
                if result4.fun < best_val_2d:
                    best_val_2d = result4.fun
                    best_2d = result4.x
            except Exception:
                pass
    
    # Reconstruct the 3-coefficient solution
    # The optimal ratios are best_2d = [c4/c0, c8/c0]
    # We need to return [c0, c4, c8] such that the bound is minimized.
    # Since the objective is scale-invariant, any scale works, but we should
    # return a reasonable scale. Let's use c0 = 1.
    best_x = np.array([1.0, best_2d[0], best_2d[1]])
    
    # Verify the value
    final_val = _compute_bound_analytic(*best_x)
    
    # If the seed was better (shouldn't happen, but safety check)
    seed_val = _compute_bound_analytic(*seed_x0)
    if seed_val < final_val:
        best_x = seed_x0
    
    return best_x
# EVOLVE-BLOCK-END
