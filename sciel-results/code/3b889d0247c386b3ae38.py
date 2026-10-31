"""
Optimized program: minimize normalized maximum of autoconvolution of a nonnegative
step function with 600 equal intervals on [-1/4, 1/4].

The score is the upper bound on C1 (raw value, lower is better).
We use a structural change: optimize a Gaussian mixture model in continuous
parameter space (means, sigmas, amplitudes) to produce smooth step functions,
then refine with direct coordinate descent. The Gaussian mixture family is
known to escape discrete local minima and produce smoother profiles.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit

N = 600
# Interval width
dx = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # = 1/2 / 600 = 1/1200

# Midpoints of intervals
_x = -1.0/4 + (np.arange(N) + 0.5) * dx


def _autoconv_peak(h):
    """Compute max of discrete autoconvolution using FFT."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(H * np.conj(H), n=size)
    # The autoconvolution at lag k (k=0..n-1) is ac[k]
    # For a function on [0, n) with unit spacing, (f*f)(k) = sum_i f(i)*f(i-k)
    # We want max over all lags
    return np.max(ac[:n])


def _score(h):
    """
    Normalized maximum of autoconvolution.
    For a step function with heights h_i on intervals of width dx:
    ∫f dx ≈ dx * sum(h_i)
    (f*f)(t) at t=0 ≈ dx * sum(h_i^2)  [each pair contributes h_i^2 * dx]
    
    Actually, more carefully:
    f(x) = h_i for x in [(i-0.5)*dx - 1/4, (i+0.5)*dx - 1/4]
    (f*f)(t) = ∫ f(s) f(t-s) ds
    
    At t=0: (f*f)(0) = ∫ f(s)^2 ds ≈ dx * sum(h_i^2)
    ∫f dx ≈ dx * sum(h_i)
    
    Normalized max = max_t (f*f)(t) / (∫f)^2
    = [dx * max_k sum_i h_i h_{i+k}] / (dx * sum(h_i))^2
    = max_k sum_i h_i h_{i+k} / (dx * sum(h_i)^2)
    
    Hmm, the dx factor matters. Let me think again.
    
    Actually for the discrete approximation:
    (f*f)(k*dx) ≈ dx * sum_i h_i * h_{i+k}  (for k >= 0, with h_j = 0 outside)
    
    ∫f dx = dx * sum(h_i)
    
    So normalized max = [dx * max_k sum_i h_i h_{i+k}] / (dx * sum(h_i))^2
                      = max_k sum_i h_i h_{i+k} / (dx * sum(h_i)^2)
    
    For uniform h_i = 1: sum(h_i) = N, max_k sum_i h_i h_{i+k} = N (at k=0)
    Score = N / (dx * N^2) = 1/(dx*N) = 1/(N * 1/(2N)) = 2
    
    Hmm, that gives 2 for uniform. Let me reconsider.
    
    Maybe the normalization is different. Perhaps:
    normalized max = max_k (f*f)(k) / max_k (f*f)(k) at uniform
    Or maybe it's just max_k sum_i h_i h_{i+k} / (sum(h_i)/N)^2 * (1/N)
    
    Let me try: score = max_k sum_i h_i h_{i+k} / (sum(h_i)^2 / N)
    For uniform: N / (N^2/N) = N/N = 1
    
    Or: score = max_k sum_i h_i h_{i+k} * N / sum(h_i)^2
    For uniform: N * N / N^2 = 1
    
    I'll use: score = N * max_k sum_i h_i h_{i+k} / sum(h_i)^2
    This equals 1 for uniform and is scale-invariant.
    """
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s == 0:
        return float('inf')
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(H * np.conj(H), n=size)
    # ac[k] for k=0..n-1 gives sum_i h_i * h_{i+k} (circular, but for k < n and
    # if we zero-pad properly it's linear)
    # Actually with zero-padding to 2n, ac[k] for k=0..n-1 is the linear autocorrelation
    peak = np.max(ac[:n])
    return N * peak / (s * s)


def _gaussian_mixture_to_heights(params, n_comp=5):
    """
    Convert Gaussian mixture parameters to step heights.
    params: [amp_1, mean_1, sigma_1, amp_2, mean_2, sigma_2, ...]
    Returns array of N nonnegative heights.
    """
    h = np.zeros(N)
    for j in range(n_comp):
        amp = params[3*j]
        mean = params[3*j + 1]
        sigma = max(1e-6, params[3*j + 2])
        h += amp * np.exp(-0.5 * ((_x - mean) / sigma) ** 2)
    return np.maximum(h, 0.0)


def _objective_mixture(params, n_comp=5):
    """Objective for Gaussian mixture optimization."""
    h = _gaussian_mixture_to_heights(params, n_comp)
    return _score(h)


def _objective_direct(h):
    """Objective for direct height optimization."""
    return _score(h)


def _refine_direct(h0, max_iter=2000, deadline=None):
    """
    Coordinate descent refinement with random restarts within the function.
    """
    h = h0.copy().astype(np.float64)
    best_h = h.copy()
    best_score = _score(h)
    current_score = best_score
    
    np.random.seed(123)
    no_improve = 0
    patience = 300
    step_size = 0.05
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if no_improve > patience:
            break
        
        # Pick a random coordinate
        idx = np.random.randint(N)
        old_val = h[idx]
        # Try a random perturbation
        new_val = max(0.0, old_val + np.random.normal(0, step_size))
        h[idx] = new_val
        new_score = _score(h)
        
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            h[idx] = old_val
            no_improve += 1
    
    return best_h, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # --- Phase 1: Gaussian mixture optimization via differential evolution ---
    # Use 4 Gaussian components: 12 parameters
    n_comp = 4
    n_params = 3 * n_comp
    
    # Bounds: amplitude in [0, 5], mean in [-0.25, 0.25], sigma in [0.005, 0.3]
    bounds = []
    for j in range(n_comp):
        bounds.append((0.0, 5.0))    # amplitude
        bounds.append((-0.25, 0.25)) # mean
        bounds.append((0.005, 0.3))  # sigma
    
    # Try a few different n_comp values
    for nc in [3, 4, 5]:
        if time.time() > deadline - 60:
            break
        
        n_p = 3 * nc
        bnds = []
        for j in range(nc):
            bnds.append((0.0, 5.0))
            bnds.append((-0.25, 0.25))
            bnds.append((0.005, 0.3))
        
        def obj_mixture(params, nc=nc):
            h = _gaussian_mixture_to_heights(params, nc)
            return _score(h)
        
        try:
            result = differential_evolution(
                obj_mixture, bnds,
                seed=42, maxiter=200, tol=1e-6,
                polish=False, popsize=15,
                init='latinhypercube'
            )
        except Exception:
            continue
        
        h_candidate = _gaussian_mixture_to_heights(result.x, nc)
        s = _score(h_candidate)
        if s < best_score:
            best_score = s
            best_h = h_candidate.copy()
    
    # --- Phase 2: Local refinement with scipy minimize (L-BFGS-B) ---
    if time.time() < deadline - 30:
        h0 = best_h.copy()
        try:
            result = minimize(
                _objective_direct, h0,
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    # --- Phase 3: Direct coordinate descent refinement ---
    if time.time() < deadline - 10:
        h_refined, s_refined = _refine_direct(
            best_h, max_iter=3000, deadline=deadline - 5
        )
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined
    
    # --- Phase 4: Try some structured candidates ---
    if time.time() < deadline - 5:
        i = np.arange(N)
        
        # Raised cosine
        rc = 0.5 * (1.0 + np.cos(np.pi * (i - N/2.0) / (N/2.0)))
        s = _score(rc)
        if s < best_score:
            best_score = s
            best_h = rc.copy()
        
        # Pure cosine bump
        cb = np.maximum(np.cos(np.pi * (i - N/2.0) / N), 0.0)
        s = _score(cb)
        if s < best_score:
            best_score = s
            best_h = cb.copy()
        
        # Two-lobed
        tl = 0.5 + 0.5 * np.cos(4 * np.pi * i / N)
        s = _score(tl)
        if s < best_score:
            best_score = s
            best_h = tl.copy()
        
        # Edge-concentrated
        ec = np.abs(np.cos(np.pi * i / N))
        s = _score(ec)
        if s < best_score:
            best_score = s
            best_h = ec.copy()
        
        # Smooth bump (Gaussian centered)
        gb = np.exp(-0.5 * ((i - N/2.0) / (N/6.0)) ** 2)
        s = _score(gb)
        if s < best_score:
            best_score = s
            best_h = gb.copy()
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
