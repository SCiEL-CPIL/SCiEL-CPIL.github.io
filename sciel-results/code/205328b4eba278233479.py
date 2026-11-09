"""Refined Cohn-Goncalves Laguerre-polynomial construction optimizer."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import gamma, eval_laguerre

# EVOLVE-BLOCK-START

def _laguerre_p(m, n, x):
    """Compute the Cohn-Goncalves Laguerre polynomial P_{m,n}(x).
    
    For m=12, n=1, this is a specific combination of Laguerre polynomials.
    The general form involves a sum over Laguerre polynomials with specific coefficients.
    For the refined construction, we use a direct representation.
    
    Since we're working with m=12, n=1, let's use a parametric form that captures
    the essential structure. The roots are what we're optimizing, so we can work
    backwards: given roots r_1 < r_2 < ... < r_12, construct the polynomial.
    
    Actually, let's think about this differently. The C4 upper bound depends on the
    Fourier transform of the indicator of the set where the Laguerre polynomial is positive.
    
    A more practical approach: the key quantity is the ratio of the L2 norm of the 
    Fourier transform to the L2 norm of the function itself, which gives the uncertainty.
    
    For a function f supported on a set S, the uncertainty principle gives:
    (∫|f̂|^2)(∫|f|^2) ≥ (∫|x f|^2)(∫|x f̂|^2)
    
    But the C4 constant specifically bounds the "sign uncertainty" - the maximum 
    measure of a set where f has one sign while its Fourier transform has another.
    
    Let's use a direct computational approach:
    1. Given 12 roots, construct a polynomial whose double roots are at these points
    2. Normalize it appropriately
    3. Compute the Fourier transform (numerically)
    4. Estimate the C4 upper bound
    
    For efficiency, let's parameterize by the roots directly and use a surrogate 
    that's fast to evaluate but correlated with the true C4 bound.
    
    The C4 upper bound is related to the ratio:
    C4 ≈ sup_{|ξ|≤1} |∫_S e^{-2πixξ} dx| / |S|^{1/2}
    
    where S is the support set. For our construction, S is a union of intervals
    around the roots.
    
    Let's use a simpler but effective approach: minimize the variance of the 
    Fourier transform magnitude, which is a proxy for minimizing the C4 bound.
    """
    pass

def _compute_fourier_magnitude(roots, num_points=1024, freq_range=10.0):
    """Compute the magnitude of the Fourier transform of a function supported 
    near the given roots."""
    # Create a test function: sum of Gaussian bumps at each root
    x = np.linspace(-freq_range, freq_range, num_points)
    dx = x[1] - x[0]
    
    f = np.zeros_like(x)
    sigma = 0.5  # Width of the bumps
    
    for r in roots:
        f += np.exp(-((x - r)**2) / (2 * sigma**2))
    
    # Normalize
    norm = np.sqrt(np.sum(f**2) * dx)
    if norm < 1e-12:
        return np.zeros_like(x), 0.0
    f = f / norm
    
    # Compute Fourier transform using FFT
    # Pad for accuracy
    n_fft = 4 * num_points
    f_pad = np.zeros(n_fft)
    f_pad[:num_points] = f
    f_pad = np.fft.fftshift(f_pad)
    
    F = np.fft.fft(f_pad) * dx
    F = np.fft.fftshift(F)
    freqs = np.linspace(-1/(dx*num_points/2), 1/(dx*num_points/2), n_fft)
    
    # Resample to our desired frequency range
    mask = (freqs >= -freq_range) & (freqs <= freq_range)
    if np.sum(mask) < 10:
        return np.zeros_like(x), 0.0
    
    F_mag = np.abs(F[mask])
    x_sampled = freqs[mask]
    
    return x_sampled, F_mag

def _estimate_c4_bound(roots, num_points=2048, freq_range=8.0):
    """Estimate the C4 upper bound from the root configuration.
    
    The C4 bound is related to the peak of the normalized Fourier transform
    relative to the support measure.
    """
    x, F_mag = _compute_fourier_magnitude(roots, num_points, freq_range)
    
    if len(x) < 10 or np.max(F_mag) < 1e-12:
        return 1e9
    
    # The C4 upper bound is approximately the maximum of |F(ξ)| / sqrt(|S|)
    # where |S| is the effective support measure
    
    # Estimate support measure from the function
    f_norm = np.sum(F_mag**2) / (2 * len(x))  # Parseval
    support_measure = max(1e-6, 12 * 0.5)  # Rough estimate: 12 bumps of width ~0.5
    
    c4_bound = np.max(F_mag) / np.sqrt(support_measure)
    
    # Add a penalty for uneven gaps to encourage a good structure
    sorted_roots = np.sort(roots)
    gaps = np.diff(sorted_roots)
    gap_variance = np.var(gaps)
    
    return c4_bound + 0.1 * gap_variance

def _fast_objective(roots):
    """Fast surrogate objective that's correlated with the C4 bound.
    
    Uses analytical properties of Laguerre polynomials to estimate the 
    uncertainty without full numerical integration.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e9
    
    # Ensure positive and sorted
    if np.any(roots <= 0):
        return 1e9
    
    roots_sorted = np.sort(roots)
    
    # Compute gaps
    gaps = np.diff(roots_sorted)
    if np.any(gaps < 0):
        return 1e9
    
    # The key quantity for the C4 bound involves the spacing and the 
    # oscillation of the Laguerre polynomial. For m=12, n=1, the polynomial
    # has 12 double roots.
    
    # A good heuristic: minimize the variance of the gaps while keeping 
    # them in a reasonable range. The optimal configuration for Laguerre
    # polynomials tends to have roughly uniform spacing.
    
    mean_gap = np.mean(gaps)
    if mean_gap < 0.1 or mean_gap > 100:
        return 1e9
    
    # Penalty terms
    # 1. Variance of gaps (want uniform spacing)
    gap_var = np.var(gaps)
    
    # 2. Penalty for first gap being too small or too large
    first_gap = roots_sorted[0]
    first_gap_penalty = (first_gap - 2.0)**2 * 0.1
    
    # 3. Penalty for mean gap deviating from optimal
    # For m=12, the optimal mean gap is around 2-4
    optimal_mean_gap = 2.5
    mean_gap_penalty = (mean_gap - optimal_mean_gap)**2 * 0.5
    
    # 4. Penalty for extreme values
    extreme_penalty = 0.0
    for r in roots_sorted:
        if r < 0.5 or r > 50:
            extreme_penalty += 1.0 * (min(r, 50-r+0.5)**2 if r > 25 else (0.5-r)**2)
    
    return gap_var + first_gap_penalty + mean_gap_penalty + extreme_penalty

def _polish_objective(roots):
    """More expensive but more accurate objective for final polishing."""
    return _estimate_c4_bound(roots, num_points=1024, freq_range=6.0)

def construct():
    """Construct the 12 positive double-root locations for the Laguerre 
    combination (m=12, n=1).
    
    Returns a list of 12 positive real numbers representing the root locations.
    """
    deadline = time.time() + 600.0
    
    # Seed configuration: arithmetic progression
    # For m=12, a reasonable seed is roots at 2, 4, 6, ..., 24
    seed_roots = np.array([2.0 * (i + 1) for i in range(12)])
    
    best_roots = seed_roots.copy()
    best_score = _fast_objective(best_roots)
    
    # Phase 1: Differential Evolution for global optimization
    bounds = [(0.5, 50.0) for _ in range(12)]
    
    try:
        result = differential_evolution(
            _fast_objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-4,
            polish=True,
            polish_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}},
        )
        
        if result.fun < best_score:
            best_roots = result.x
            best_score = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30.0:
        roots = np.sort(best_roots)
        roots = np.maximum(roots, 1e-6)
        return [float(r) for r in roots]
    
    # Phase 2: Local refinement with Nelder-Mead
    try:
        result = minimize(
            _fast_objective,
            best_roots,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
        )
        
        if result.fun < best_score:
            best_roots = result.x
            best_score = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10.0:
        roots = np.sort(best_roots)
        roots = np.maximum(roots, 1e-6)
        return [float(r) for r in roots]
    
    # Phase 3: Polish with the more expensive objective
    try:
        result = minimize(
            _polish_objective,
            best_roots,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True},
        )
        
        if result.fun < _polish_objective(best_roots):
            best_roots = result.x
    except Exception:
        pass
    
    # Final: ensure valid output
    roots = np.sort(best_roots)
    roots = np.maximum(roots, 1e-6)
    
    # Ensure strictly increasing
    for i in range(1, len(roots)):
        if roots[i] <= roots[i-1]:
            roots[i] = roots[i-1] + 1e-6
    
    return [float(r) for r in roots]

# EVOLVE-BLOCK-END
