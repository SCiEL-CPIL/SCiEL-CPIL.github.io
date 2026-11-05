"""Signed step function search for autoconvolution upper bound C3."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.fft import fft, ifft

_DEADLINE = time.time() + 115  # safety margin before 120s kill
N = 400


def _autoconv_peak(x):
    """Compute the peak of the autoconvolution of x (signed step heights)."""
    # Full linear autoconvolution via FFT
    n = len(x)
    size = 2 * n
    X = fft(x, size=size)
    conv = ifft(X * X).real
    # The autoconvolution of a step function on equal intervals;
    # peak magnitude is the upper bound contribution
    return np.max(np.abs(conv))


def _fbm_signal(n, H, seed):
    """
    Generate a deterministic fractional Brownian motion signal of length n
    using the Cholesky / spectral method (fractional Gaussian noise integrated).
    
    We use the spectral representation: fGn has PSD ~ |k|^{-(2H+1)} for k != 0.
    Generate complex spectrum with that decay, take real part, integrate to get fBm.
    """
    rng = np.random.RandomState(seed)
    k = np.arange(n)
    k[0] = 1  # avoid zero frequency issues for PSD
    # fGn PSD: |k|^{-(2H+1)}
    psd = np.power(k, -(2 * H + 1))
    # Random phase
    phase = 2 * np.pi * rng.rand(n)
    # Real signal via cosine/sine combination
    # Use: x[k] = sqrt(2*psd[k]) * cos(phase[k]) for k>0, and handle k=0
    amp = np.sqrt(2 * psd)
    x = amp * np.cos(phase)
    # Zero mean
    x -= x.mean()
    # Integrate to get fBm (cumulative sum approximates integral)
    fbm = np.cumsum(x)
    return fbm


def _thue_morse(n):
    """Generate Thue-Morse sequence mapped to +/-1."""
    # Thue-Morse: t(n) = popcount(n) mod 2
    tm = np.array([bin(i).count('1') % 2 for i in range(n)])
    return 1 - 2 * tm  # 0 -> 1, 1 -> -1


def _sign(x):
    return np.where(x >= 0, 1.0, -1.0)


def _local_search_flip(x, max_iters, time_limit):
    """
    Local search: try flipping individual signs to reduce autoconv peak.
    Uses FFT-based incremental updates for speed.
    """
    n = len(x)
    size = 2 * n
    X = fft(x, size=size)
    conv = ifft(X * X).real
    best_peak = np.max(np.abs(conv))
    x = x.copy()
    
    # Precompute basis vectors for flip updates
    # Flipping x[i] -> -x[i] changes x by -2*x[i]*e_i
    # In frequency: delta_X = -2*x[i] * fft(e_i)
    # We'll compute fft of basis vectors on the fly or use structure
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > time_limit:
            break
        
        # Try each position; for speed, try a random subset or all if fast
        # For N=400, full scan each pass is 400 FFT updates ~ okay for few passes
        # But we can be smarter: only try positions where local structure suggests gain.
        # For simplicity and robustness, do full scan but limit passes.
        
        # To keep it fast, do a single pass per iteration, trying all positions once
        for i in range(n):
            if time.time() > time_limit:
                break
            # Compute new signal
            x_new = x.copy()
            x_new[i] = -x_new[i]
            # Fast recompute: delta = -2*x[i]*e_i
            delta_x = np.zeros(n)
            delta_x[i] = -2 * x[i]
            delta_X = fft(delta_x, size=size)
            new_X = X + delta_X
            new_conv = ifft(new_X * new_X).real
            new_peak = np.max(np.abs(new_conv))
            if new_peak < best_peak - 1e-12:
                x = x_new
                X = new_X
                conv = new_conv
                best_peak = new_peak
                improved = True
                # After a flip, restart the scan (greedy)
                break
    
    return x, best_peak


def _evaluate(x):
    """Return the C3 upper bound metric for a signed step function."""
    return _autoconv_peak(x)


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the autoconvolution upper bound C3.
    
    Strategy:
    1. Generate candidate sign patterns from fBm with varying Hurst exponent H.
    2. Also include Thue-Morse as a structured baseline.
    3. Evaluate each candidate.
    4. Take the best and refine with local search (sign flips).
    5. Return the best sign pattern found within time budget.
    """
    start_time = time.time()
    time_limit = _DEADLINE
    
    best_x = None
    best_score = np.inf
    
    # Phase 1: Generate candidates from fBm with various H and seeds
    # Coarse grid over H in [0.1, 0.9]
    H_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    seeds = [42, 123, 777, 2024, 555]
    
    candidates = []
    
    # Thue-Morse baseline
    candidates.append(_thue_morse(N))
    
    # fBm-based candidates
    for H in H_values:
        for seed in seeds:
            if time.time() > time_limit - 30:  # leave time for local search
                break
            sig = _fbm_signal(N, H, seed)
            s = _sign(sig)
            candidates.append(s)
    
    # Also try negations and shifts of some candidates for diversity
    for c in list(candidates[:5]):
        candidates.append(-c)
        candidates.append(np.roll(c, N // 4))
        candidates.append(np.roll(c, -N // 4))
    
    # Evaluate all candidates
    for c in candidates:
        if time.time() > time_limit - 25:
            break
        score = _evaluate(c)
        if score < best_score:
            best_score = score
            best_x = c.copy()
    
    # If no candidates evaluated (shouldn't happen), use Thue-Morse
    if best_x is None:
        best_x = _thue_morse(N)
        best_score = _evaluate(best_x)
    
    # Phase 2: Local search refinement
    # Allow multiple restarts from different starting points if time permits
    local_time_limit = time_limit - 5  # 5s safety margin
    
    # First, refine the best candidate
    best_x, best_score = _local_search_flip(best_x, max_iters=50, time_limit=local_time_limit)
    
    # If time remains, try refining a few other good candidates
    if time.time() < local_time_limit - 10:
        # Sort candidates by score (re-evaluate top ones)
        # For simplicity, try refining the next best few
        tried = set()
        tried.add(tuple(best_x))
        
        for c in candidates:
            if time.time() > local_time_limit - 5:
                break
            key = tuple(c)
            if key in tried:
                continue
            tried.add(key)
            # Quick evaluate
            s = _evaluate(c)
            if s < best_score * 1.1:  # within 10% of best, worth refining
                x_ref, s_ref = _local_search_flip(c, max_iters=20, time_limit=local_time_limit)
                if s_ref < best_score:
                    best_score = s_ref
                    best_x = x_ref
    
    # Final check: ensure we return a valid signed step function
    best_x = _sign(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
