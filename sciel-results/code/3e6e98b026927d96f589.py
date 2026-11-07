"""Bootstrap search program for the Uncertainty Inequality task."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, differential_evolution

# Deadline: evaluator kills at 210s. Use 195s budget to be safe.
_BUDGET_SECONDS = 195.0

def _evaluate(c0, c1, c2):
    """
    Compute the value = r^2 / (2*pi) for the given coefficients.
    Returns a positive float (lower is better), or inf if invalid.
    """
    try:
        # Hermite polynomials: H_n(x)
        # H0 = 1
        # H4 = 16x^4 - 48x^2 + 12
        # H8 = 256x^8 - 4480x^6 + 24192x^4 - 40320x^2 + 1680
        # H12 = 4096x^12 - 147456x^10 + 1684608x^8 - 9067200x^6 + 20901120x^4 - 20321280x^2 + 665280
        #
        # Values at x=0:
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 1680.0
        H12_0 = 665280.0
        
        if abs(H12_0) < 1e-12:
            return float('inf')
        
        # c3 is fixed so that g(0) = 0
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Build g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
        # Coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
        
        a0 = c0 + c1*12 + c2*1680 + c3*665280
        
        a2 = c1*(-48) + c2*(-40320) + c3*(-20321280)
        a4 = c1*16 + c2*24192 + c3*20901120
        a6 = c2*(-4480) + c3*(-9067200)
        a8 = c2*256 + c3*1684608
        a10 = c3*(-147456)
        a12 = c3*4096
        
        # Let y = x^2, then q = P(y) where P(y) = a2 + a4*y + a6*y^2 + a8*y^3 + a10*y^4 + a12*y^5
        
        # Check if g tends to a negative limit at infinity.
        # g(x) ~ c3*4096*x^12 as x -> infinity.
        # If c3 < 0, then g -> -infinity, so we replace g by -g, which flips all coefficients.
        if c3 < 0:
            a2 = -a2
            a4 = -a4
            a6 = -a6
            a8 = -a8
            a10 = -a10
            a12 = -a12
        
        # Now find roots of P(y) = a2 + a4*y + a6*y^2 + a8*y^3 + a10*y^4 + a12*y^5
        # Coefficients in descending order for np.roots: [a12, a10, a8, a6, a4, a2]
        coeffs_P_desc = np.array([a12, a10, a8, a6, a4, a2])
        
        # Find real positive roots
        roots = np.roots(coeffs_P_desc)
        real_roots = []
        for root in roots:
            if abs(root.imag) < 1e-10:
                real_root = root.real
                if real_root > 1e-12:
                    real_roots.append(real_root)
        
        if not real_roots:
            return float('inf')
        
        # Sort real positive roots
        real_roots.sort()
        
        # For each root, check if P changes sign at that root.
        # P changes sign at y if P(y - eps) and P(y + eps) have opposite signs.
        # Use Horner's method for evaluation
        def _eval_P(y_val):
            result = a12
            result = result * y_val + a10
            result = result * y_val + a8
            result = result * y_val + a6
            result = result * y_val + a4
            result = result * y_val + a2
            return result
        
        best_y = None
        for y in real_roots:
            eps = 1e-7 * max(1.0, abs(y))
            P_minus = _eval_P(y - eps)
            P_plus = _eval_P(y + eps)
            if P_minus * P_plus < 0:
                if best_y is None or y > best_y:
                    best_y = y
        
        if best_y is None:
            return float('inf')
        
        value = best_y / (2 * np.pi)
        return value
    
    except Exception:
        return float('inf')


def _seed_value():
    """Return the value for the seed coefficients."""
    return _evaluate(-113.0/100.0, 1.0/25.0, 1.0/3240.0)


def construct():
    start_time = time.time()
    
    # Start with the seed
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_value = _evaluate(best_c[0], best_c[1], best_c[2])
    
    def _time_left():
        return _BUDGET_SECONDS - (time.time() - start_time)
    
    # Strategy 1: Local optimization from seed using Nelder-Mead
    if _time_left() > 30:
        try:
            res = minimize(
                lambda x: _evaluate(x[0], x[1], x[2]),
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Strategy 2: Differential evolution over a wider range
    if _time_left() > 60:
        try:
            bounds = [
                (-20.0, 20.0),
                (-20.0, 20.0),
                (-1.0, 1.0)
            ]
            res = differential_evolution(
                lambda x: _evaluate(x[0], x[1], x[2]),
                bounds,
                maxiter=150,
                popsize=40,
                tol=1e-10,
                seed=42,
                init='sobol'
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Strategy 3: Multi-start local optimization from random points
    if _time_left() > 60:
        rng = np.random.default_rng(12345)
        n_starts = 0
        while _time_left() > 20:
            # Use a mix of scales
            scale = rng.choice([0.1, 1.0, 5.0, 10.0])
            x0 = rng.uniform(-scale, scale, size=3)
            try:
                res = minimize(
                    lambda x: _evaluate(x[0], x[1], x[2]),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-15}
                )
                if res.fun < best_value:
                    best_value = res.fun
                    best_c = res.x
            except Exception:
                pass
            n_starts += 1
            if n_starts > 300:
                break
    
    # Strategy 4: Coordinate descent polish
    if _time_left() > 15:
        try:
            current = best_c.copy()
            improved = True
            while improved and _time_left() > 10:
                improved = False
                for i in range(3):
                    best_local = current.copy()
                    best_local_val = best_value
                    # Try small perturbations
                    for delta in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                        for sign in [-1, 1]:
                            trial = current.copy()
                            trial[i] += sign * delta
                            val = _evaluate(trial[0], trial[1], trial[2])
                            if val < best_local_val:
                                best_local_val = val
                                best_local = trial.copy()
                    if best_local_val < best_value:
                        current = best_local
                        best_value = best_local_val
                        improved = True
            best_c = current
        except Exception:
            pass
    
    # Final polish: very tight local optimization around the best found
    if _time_left() > 10:
        try:
            res = minimize(
                lambda x: _evaluate(x[0], x[1], x[2]),
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-18}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass
    
    return np.array(best_c, dtype=np.float64)

# EVOLVE-BLOCK-END
