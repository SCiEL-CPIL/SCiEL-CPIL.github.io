# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 31

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    This is differentiable and allows gradient-based optimization.
    """
    abs_ac = np.abs(ac)
    # Shift for numerical stability
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    # log-sum-exp trick: log(sum(exp(x_i))) = max(x_i) + log(sum(exp(x_i - max(x_i))))
    # We want (1/beta) * log(sum(exp(beta * |ac_i|)))
    # = (1/beta) * (beta * m + log(sum(exp(beta * (|ac_i| - m)))))
    # = m + (1/beta) * log(sum(exp(beta * (|ac_i| - m))))
    scaled = beta * (abs_ac - m)
    # Clip to avoid overflow/underflow issues in exp
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    We optimize this to find good candidates, then evaluate the true objective.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_chirp(n):
    """Generate a chirp sequence: h[i] = exp(i * pi * i^2 / n)"""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    # Real part for signed step function
    return np.cos(phase)

def _gen_barker_like(n):
    """
    Generate a sequence with good autocorrelation properties.
    Uses a simple construction: alternating signs with some structure.
    """
    h = np.zeros(n)
    for i in range(n):
        # Use a simple pattern that tends to have lower sidelobes
        # This is a heuristic; not guaranteed to be optimal
        if i % 4 == 0:
            h[i] = 1.0
        elif i % 4 == 1:
            h[i] = -1.0
        elif i % 4 == 2:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        indices = rng.permutation(N)
        for i in indices:
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _local_search_perturb(h, max_passes=3, deadline=None, step_size=0.1):
    """
    Local optimization: try small perturbations to each element, keep if improves score.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(123)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            # Try several perturbation sizes
            for delta in [step_size, -step_size, 2*step_size, -2*step_size]:
                h[i] = old_val + delta
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                    break
            if not improved:
                h[i] = old_val  # revert if no improvement
        if not improved:
            break
    return h, best_val

def _gen_mersenne_twister(n, seed=42):
    """Generate a pseudo-random sequence with good autocorrelation properties."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)

def _gen_golden_ratio(n):
    """Generate a sequence based on golden ratio properties."""
    phi = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    # Use fractional parts of i * phi to create a low-discrepancy sequence
    frac = (i * phi) % 1.0
    # Map to [-1, 1]
    return 2 * frac - 1

def _subspace_optimize(h_init, K=20, beta=50.0, deadline=None):
    """
    Optimize h by projecting onto a K-dimensional cosine subspace.
    h = Phi @ a
    """
    x = np.arange(N)
    # Cosine basis: cos(pi * k * x / N)
    # Use k=0..K-1
    k = np.arange(K)
    # Phi has shape (N, K)
    Phi = np.cos(np.pi * np.outer(x, k) / N)
    
    # Initialize a from h_init
    # Least squares projection
    try:
        a_init, _, _, _ = np.linalg.lstsq(Phi, h_init, rcond=None)
    except Exception:
        return h_init
    
    def obj(a):
        if deadline is not None and time.time() > deadline:
            return 1e10
        h = Phi @ a
        s = np.sum(h)
        if abs(s) < 1e-9:
            return 1e10
        ac = _autoconv_fft(h)
        sm = _soft_max_abs(ac, beta=beta)
        return sm / (s * s)
    
    # Optimize a
    try:
        res = minimize(obj, a_init, method='L-BFGS-B', options={'maxiter': 100, 'disp': False, 'gtol': 1e-5})
        h_opt = Phi @ res.x
        # Ensure valid
        if np.all(np.isfinite(h_opt)):
            return h_opt
        else:
            return h_init
    except Exception:
        return h_init

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian window
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle window
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 7. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
    
    # 8. Golden ratio sequence
    inits.append(_gen_golden_ratio(N))
    
    # 9. Mersenne Twister random sequences
    for seed in [1, 2, 3]:
        inits.append(_gen_mersenne_twister(N, seed))
        
    # Optimization loop using subspace projection
    # We try different subspace dimensions K
    K_values = [10, 20, 50]
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        for K in K_values:
            if time.time() > deadline - 10:
                break
                
            h_opt = _subspace_optimize(h_norm, K=K, beta=50.0, deadline=deadline-5)
            val = _score(h_opt)
            if val < best_score:
                best_score = val
                best_h = h_opt.copy()

    # Final local search refinement
    if time.time() < deadline - 5:
        # First try sign flips
        h_refined, val_refined = _local_search_flip(best_h, max_passes=5, deadline=deadline-2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
        
        # Then try perturbations if time remains
        if time.time() < deadline - 3:
            h_perturbed, val_perturbed = _local_search_perturb(best_h, max_passes=2, deadline=deadline-1, step_size=0.05)
            if val_perturbed < best_score:
                best_score = val_perturbed
                best_h = h_perturbed.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
