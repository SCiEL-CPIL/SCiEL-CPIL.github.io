"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize_scalar, minimize, brentq


def _compute_objective(a0, a4, a8):
    """
    Compute the variance product Var(x)*Var(xi) for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)*exp(-x^2/2)
    where a12 is derived from g(0)=0.
    
    Uses exact Hermite polynomial orthogonality relations.
    """
    # a12 from g(0) = 0: a0*1 + a4*(-12) + a8*1680 + a12*(-103950) = 0
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 103950.0
    a = np.array([a0, a4, a8, a12])
    degrees = [0, 4, 8, 12]
    
    # Precompute norms: c_k = sqrt(2^k * k! / sqrt(pi)) so that ||H_k * exp(-x^2/2)||_2^2 = sqrt(pi)*2^k*k!
    # norm_sq = sum_k a_k^2 * sqrt(pi) * 2^k * k!
    sqrt_pi = math.sqrt(math.pi)
    norm_sq = 0.0
    for k, ak in zip(degrees, a):
        norm_sq += ak**2 * sqrt_pi * (2.0**k) * math.factorial(k)
    
    if norm_sq <= 0:
        return float('inf')
    
    # Second moment: sum_{k,l} a_k * a_l * M[k,l]
    # M[k,l] = integral x^2 H_k(x) H_l(x) exp(-x^2) dx
    # Using x^2 H_k = (1/4)H_{k+2} + (2k+1)/2 * H_k + k(k-1)/2 * H_{k-2}
    # M[k,l] = sqrt(pi) * 2^l * l! * [ (1/4)*delta(l,k+2) + (2k+1)/2*delta(k,l) + k(k-1)/2*delta(l,k-2) ]
    
    second_moment = 0.0
    for i, ki in enumerate(degrees):
        for j, lj in enumerate(degrees):
            term = 0.0
            if lj == ki + 2:
                term += 0.25
            if ki == lj:
                term += (2.0*ki + 1.0) / 2.0
            if lj == ki - 2:
                term += ki*(ki-1) / 2.0
            M_kl = sqrt_pi * (2.0**lj) * math.factorial(lj) * term
            second_moment += a[i] * a[j] * M_kl
    
    variance_x = second_moment / norm_sq
    
    # For Fourier side with standard convention, variance is the same
    # (Hermite functions are eigenfunctions of Fourier transform up to phase)
    variance_xi = variance_x
    
    return variance_x * variance_xi


def _objective_ratio(r):
    """
    1D objective: fix a0=1, a4=r, a8=r^2 (or some parameterization).
    Actually, let's use a4=r, a8=r^2 to cover a 2D subspace, but that's still 1D.
    Better: use a4=r, a8=s*r where we fix one ratio.
    
    Actually, the scale invariance means we can fix a0=1 (if a0 != 0) and optimize over (a4, a8).
    But to make it truly 1D, we can parameterize a4 = r, a8 = r^2 or a4=r, a8=c*r.
    
    Let me use a different approach: fix a0=1, then optimize over a4 and a8 using the fact
    that the problem is 2D but we can do a smart 1D scan.
    
    Actually, let's just do a proper 2D optimization since it's fast with exact formulas.
    """
    pass


def construct():
    """
    Find the coefficients (a0, a4, a8) that minimize the variance product.
    Uses scale invariance: fix a0=1, optimize over (a4, a8).
    """
    t_start = time.time()
    deadline = t_start + 180.0
    
    def _check_time():
        return time.time() > deadline - 5.0
    
    # The objective is scale-invariant: (ca0, ca4, ca8) gives same result as (a0, a4, a8)
    # So fix a0 = 1 and optimize over (a4, a8)
    
    def _obj_2d(params):
        a4, a8 = params
        return _compute_objective(1.0, a4, a8)
    
    # Multi-start optimization
    best_val = float('inf')
    best_params = None
    
    # Initial guesses
    starts = [
        (0.1, 0.01),
        (-0.1, 0.01),
        (0.01, 0.001),
        (-0.01, 0.001),
        (0.5, 0.1),
        (-0.5, 0.1),
        (1.0, 0.5),
        (-1.0, 0.5),
        (0.0, 0.0),
        (0.2, 0.05),
        (-0.2, 0.05),
        (0.3, 0.03),
        (-0.3, 0.03),
        (0.05, 0.005),
        (-0.05, 0.005),
        (1.0, 0.1),
        (-1.0, 0.1),
        (2.0, 0.5),
        (-2.0, 0.5),
        (0.5, 0.05),
    ]
    
    # Add the original seed values (scaled so a0=1)
    a0_seed, a4_seed, a8_seed = -113.0/100, 1.0/25, 1.0/3240
    if abs(a0_seed) > 1e-12:
        starts.append((a4_seed/a0_seed, a8_seed/a0_seed))
    
    for a4_s, a8_s in starts:
        if _check_time():
            break
        try:
            result = minimize(
                _obj_2d,
                x0=np.array([a4_s, a8_s]),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Also try L-BFGS-B from best point
    if best_params is not None and not _check_time():
        try:
            result = minimize(
                _obj_2d,
                x0=best_params,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final polish with tight tolerances
    if best_params is not None and not _check_time():
        try:
            result = minimize(
                _obj_2d,
                x0=best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is not None:
        a4_opt = best_params[0]
        a8_opt = best_params[1]
        return np.array([1.0, a4_opt, a8_opt])
    else:
        # Fallback to seed
        return np.array([-113.0/100, 1.0/25, 1.0/3240])
# EVOLVE-BLOCK-END
