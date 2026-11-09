"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _compute_laguerre_zeros(n):
    """Compute zeros of the Laguerre polynomial L_n(x) using numpy."""
    import numpy as np
    # L_n(x) coefficients: c_k = (-1)^k * binom(n+k, k) / k!
    c = []
    for k in range(n + 1):
        val = ((-1) ** k) * math.comb(n + k, k) / math.factorial(k)
        c.append(val)
    # numpy.roots expects highest power first
    c_rev = list(reversed(c))
    roots = np.roots(c_rev)
    # Take real positive roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    real_roots.sort()
    return real_roots

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for the Cohn-Goncalves construction.
    
    For the Fourier sign uncertainty with m double roots in 1D,
    the bound involves the L4 norm ratio. A reasonable objective
    based on the theory is related to the integral of the squared
    modulus of the Fourier transform of the indicator function
    defined by the roots.
    
    We use a surrogate: minimize the maximum of the Laguerre-weighted
    variance of the roots, which correlates with the C4 bound.
    """
    import numpy as np
    roots = np.array(roots, dtype=float)
    
    # The C4 bound for the Cohn-Goncalves construction is related to
    # the ratio of norms. A key quantity is the second moment under
    # the Laguerre weight.
    
    # For Laguerre weight w(x) = x^0 * exp(-x) on [0, infinity):
    # The moments are E[x^k] = k! for the weight exp(-x)
    # But our roots are at specific locations, not continuous.
    
    # A better surrogate: the C4 bound involves the integral
    # int_0^inf |sum_i c_i exp(-i*a_i*t)|^4 dt / int_0^inf |sum_i c_i exp(-i*a_i*t)|^2 dt
    # for some coefficients c_i.
    
    # For the simplified case with equal weights, this becomes:
    # Compute the L4/L2 norm ratio of the sum of exponentials.
    
    n = len(roots)
    # Sample over a range of t values
    t_max = 50.0
    n_samples = 10000
    t = np.linspace(0, t_max, n_samples)
    
    # Sum of exponentials: S(t) = sum_i exp(-i * a_i * t)
    # |S(t)|^2 and |S(t)|^4
    S = np.zeros_like(t, dtype=complex)
    for a in roots:
        S += np.exp(-1j * a * t)
    
    S2 = np.abs(S) ** 2
    S4 = np.abs(S) ** 4
    
    # Integrate using trapezoidal rule
    dt = t[1] - t[0]
    int_S2 = np.trapz(S2, t)
    int_S4 = np.trapz(S4, t)
    
    # The bound is related to (int_S4 / int_S2^2)^(1/2) or similar
    # For n=1, this should give a known value.
    # Let's use: C4_bound = sqrt(int_S4 / int_S2)
    if int_S2 > 1e-30:
        bound = math.sqrt(int_S4 / int_S2)
    else:
        bound = float('inf')
    
    return bound

def construct():
    # Deadline guard
    t_start = time.time()
    DEADLINE = 620.0
    
    # Compute Laguerre zeros of L_12
    try:
        lag_roots = _compute_laguerre_zeros(12)
        if len(lag_roots) != 12:
            raise ValueError("Expected 12 Laguerre zeros")
    except Exception:
        return [4.0 * (i + 1) for i in range(12)]
    
    # Try different scalings of the Laguerre zeros
    # The seed uses [4, 8, 12, ..., 48]
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    seed_bound = _compute_c4_bound(seed_roots)
    
    best_roots = seed_roots
    best_bound = seed_bound
    
    # Try various scalings
    lag_min = min(lag_roots)
    lag_max = max(lag_roots)
    lag_range = lag_max - lag_min
    
    # Try scaling to match different target ranges
    for target_min in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for target_max in [30.0, 40.0, 48.0, 60.0, 80.0, 100.0]:
            if target_max <= target_min:
                continue
            scale = (target_max - target_min) / lag_range
            scaled = [(r - lag_min) * scale + target_min for r in lag_roots]
            if min(scaled) <= 0:
                continue
            bound = _compute_c4_bound(scaled)
            if bound < best_bound:
                best_bound = bound
                best_roots = scaled
    
    # Also try the raw Laguerre zeros (unscaled)
    bound_raw = _compute_c4_bound(lag_roots)
    if bound_raw < best_bound:
        best_bound = bound_raw
        best_roots = lag_roots
    
    # Try a few more targeted scalings around the best found
    if best_roots != seed_roots:
        best_min = min(best_roots)
        best_max = max(best_roots)
        for frac in [0.8, 0.9, 1.1, 1.2]:
            new_min = best_min * frac
            new_max = best_max * frac
            if new_min <= 0:
                continue
            scale = (new_max - new_min) / lag_range
            scaled = [(r - lag_min) * scale + new_min for r in lag_roots]
            if min(scaled) <= 0:
                continue
            bound = _compute_c4_bound(scaled)
            if bound < best_bound:
                best_bound = bound
                best_roots = scaled
    
    if time.time() - t_start > DEADLINE:
        return seed_roots
    
    return best_roots
# EVOLVE-BLOCK-END
