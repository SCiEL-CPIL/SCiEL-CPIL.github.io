"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_start_time = time.time()
_DEADLINE = 210.0
_SAFETY = 10.0  # seconds margin

def _hermite_physicists(n, x):
    """Compute physicist's Hermite polynomial H_n(x) for n in {0,4,8,12}."""
    if n == 0:
        return np.ones_like(x)
    elif n == 4:
        return 16*x**4 - 48*x**2 + 12
    elif n == 8:
        return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    elif n == 12:
        return (4096*x**12 - 98304*x**10 + 1036800*x**8 
                - 5529600*x**6 + 12902400*x**4 - 12902400*x**2 + 10321920)
    else:
        raise ValueError(f"Unsupported n={n}")

def _objective(coeffs):
    """Compute upper bound on C4 = sup|g| * sup|g_hat|."""
    a0, a4, a8 = coeffs
    # Fix a12 so that g(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=10321920
    a12 = -(a0 + 12*a4 + 1680*a8) / 10321920.0
    
    # Evaluate on a grid
    x = np.linspace(-8, 8, 4000)
    gauss = np.exp(-x**2 / 2.0)
    
    H0 = _hermite_physicists(0, x)
    H4 = _hermite_physicists(4, x)
    H8 = _hermite_physicists(8, x)
    H12 = _hermite_physicists(12, x)
    
    g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss
    
    # Fourier transform: H_n(x) e^{-x^2/2} <-> (-i)^n H_n(xi) e^{-xi^2/2}
    # So g_hat(xi) = (a0*H0(xi) + a4*(-i)^4*H4(xi) + a8*(-i)^8*H8(xi) + a12*(-i)^12*H12(xi)) * gauss
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = (-i)^4 * (-i)^8 = 1*1 = 1... wait
    # (-i)^0 = 1, (-i)^4 = ((-i)^2)^2 = (-1)^2 = 1, (-i)^8 = 1, (-i)^12 = 1
    # Actually: (-i)^2 = -1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    # So for even n that are multiples of 4, (-i)^n = 1
    # 0, 4, 8, 12 are all multiples of 4, so all coefficients are +1
    g_hat = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss  # same polynomial, just evaluated at xi
    
    # Actually the Fourier transform of H_n(x)e^{-x^2/2} is (-i)^n H_n(xi) e^{-xi^2/2}
    # For n=0: (-i)^0 = 1
    # For n=4: (-i)^4 = 1
    # For n=8: (-i)^8 = 1
    # For n=12: (-i)^12 = 1
    # So g_hat(xi) has the same form as g(x)
    
    sup_g = np.max(np.abs(g))
    sup_g_hat = np.max(np.abs(g_hat))
    
    return sup_g * sup_g_hat

def construct():
    global _start_time, _DEADLINE, _SAFETY
    
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_score = _objective(best_coeffs)
    
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - _SAFETY - elapsed
    
    if remaining <= 5:
        return best_coeffs
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    
    # Starting points
    starts = [best_coeffs.copy()]
    # Add some random perturbations
    for _ in range(15):
        pert = rng.normal(0, 0.5, 3)
        starts.append(best_coeffs + pert)
    # Some structured starts
    starts.append(np.array([-1.0, 0.5, 0.5]))
    starts.append(np.array([-1.5, 0.3, 0.3]))
    starts.append(np.array([-0.8, 0.6, 0.4]))
    starts.append(np.array([-1.2, 0.4, 0.6]))
    
    for start in starts:
        elapsed = time.time() - _start_time
        if _DEADLINE - _SAFETY - elapsed < 10:
            break
        
        try:
            result = opt.minimize(
                _objective, start, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    elapsed = time.time() - _start_time
    if _DEADLINE - _SAFETY - elapsed > 5:
        try:
            result = opt.minimize(
                _objective, best_coeffs, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
