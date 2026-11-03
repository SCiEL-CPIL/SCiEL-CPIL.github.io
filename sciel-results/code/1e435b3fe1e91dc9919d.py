"""
Optimize nonnegative step function on 600 equal intervals of [-1/4, 1/4]
to minimize the normalized maximum of its autoconvolution.

Strategy:
- Start with a parametric sharp-peak family (concentrating mass at center).
- Combine with diverse random candidates and structured asymmetric shapes.
- Use iterative local search (coordinate descent + random perturbations)
  to refine candidates within a time budget.
- Return the best candidate found before the deadline.
"""
import time
import numpy as np

N = 600
_DOMAIN_START = -0.25
_DOMAIN_END = 0.25
_DOMAIN_LEN = _DOMAIN_END - _DOMAIN_START  # 0.5
dx = _DOMAIN_LEN / N  # 1/1200

_DEADLINE_S = 140.0
_T0 = time.monotonic()


def _remaining() -> float:
    return _DEADLINE_S - (time.monotonic() - _T0)


def _score_autoconv(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of length N, nonnegative.
    The autoconvolution (discrete) is computed via FFT for efficiency.
    We normalize by the L2 norm squared of h (which equals the autoconv at lag 0).
    The score is max(autoconv) / (sum(h^2) * dx) where dx accounts for the step width.
    But since the problem says "normalized maximum of its autoconvolution",
    and the seed uniform box scores 1.0, let's compute:
    
    For a step function f(x) = h[i] on interval i, the autoconvolution is:
    (f * f)(t) = integral f(x) f(t-x) dx
    
    In discrete approximation with step width dx:
    (f*f)[k] ≈ dx^2 * sum_i h[i] h[i-k]  (for appropriate k)
    
    The L2 norm squared is: ∫ f^2 dx ≈ dx * sum h[i]^2
    
    The normalized maximum would be: max_k (f*f)[k] / (L2^2) 
    = dx^2 * max_k sum_i h[i]h[i-k] / (dx * sum h[i]^2)
    = dx * max_k sum_i h[i]h[i-k] / sum h[i]^2
    
    But this depends on the exact normalization. Let's look at the seed:
    Uniform box: h[i] = 1 for all i.
    Autoconv at lag 0: sum_i h[i]^2 = N = 600
    Autoconv at other lags: for lag k, sum_i h[i]h[i-k] = N - |k| (for |k| < N)
    Max is at lag 0: 600
    L2^2 (continuous) = dx * sum h[i]^2 = dx * 600
    Normalized max = (dx^2 * 600) / (dx * 600) = dx = 1/1200? That doesn't match score 1.0.
    
    Let me reconsider. Perhaps the normalization is different.
    Maybe it's max(autoconv) / autoconv[0]? For uniform box, that's 600/600 = 1.0. Yes!
    
    So the score is: max_k (sum_i h[i] * h[i+k]) / (sum_i h[i]^2)
    where we take the maximum over all lags k.
    
    This is always >= 1.0 because at k=0, the numerator equals the denominator.
    """
    h = np.asarray(h, dtype=np.float64)
    # Use FFT to compute autoconvolution efficiently
    # Pad to avoid circular convolution artifacts
    n_fft = 2 * N
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    
    autoconv = np.fft.irfft(np.fft.rfft(h_pad) ** 2)
    
    # The autoconv at lag 0 is sum(h^2), at lag k is sum(h[i]*h[i+k])
    # We want max over all lags, normalized by autoconv[0]
    if autoconv[0] <= 0:
        return 1.0
    return float(np.max(autoconv) / autoconv[0])


def _random_candidate(rng):
    """Generate a random nonnegative candidate."""
    # Mix of different types:
    # 1. Sparse peak
    # 2. Smooth bump
    # 3. Random walk
    # 4. Symmetric/asymmetric shapes
    
    r = rng.random()
    
    if r < 0.3:
        # Sharp peak at random location
        h = np.zeros(N)
        center = rng.randint(0, N)
        width = rng.randint(1, max(2, N // 10))
        for i in range(N):
            dist = abs(i - center)
            if dist < width:
                h[i] = 1.0 - (dist / width) ** 2
        # Add small noise
        h += rng.random(N) * 0.1
        return np.clip(h, 0, None)
    
    elif r < 0.5:
        # Gaussian-like bump
        center = rng.random() * N
        sigma = rng.uniform(5, N / 4)
        i = np.arange(N)
        h = np.exp(-0.5 * ((i - center) / sigma) ** 2)
        h += rng.random(N) * 0.05
        return np.clip(h, 0, None)
    
    elif r < 0.7:
        # Symmetric tent
        center = rng.randint(0, N)
        h = np.zeros(N)
        for i in range(N):
            dist = min(abs(i - center), N - abs(i - center))
            max_dist = max(center, N - 1 - center)
            if max_dist > 0:
                h[i] = 1.0 - (dist / max_dist)
        h += rng.random(N) * 0.1
        return np.clip(h, 0, None)
    
    elif r < 0.85:
        # Random smooth function via low-frequency Fourier
        freqs = np.fft.rfftfreq(N)
        # Keep only low frequencies
        cutoff = rng.randint(1, 20)
        coeffs = np.zeros(len(freqs), dtype=complex)
        low_mask = np.abs(freqs) <= (cutoff / N)
        coeffs[low_mask] = rng.randn(len(freqs))[low_mask] + 1j * rng.randn(len(freqs))[low_mask]
        h = np.fft.irfft(coeffs, n=N)
        h = np.clip(h, 0, None)
        if np.max(h) > 0:
            h /= np.max(h)
        return h
    
    else:
        # Completely random
        h = np.abs(rng.randn(N))
        if np.max(h) > 0:
            h /= np.max(h)
        return h


def _local_search(h, rng, max_iter=500):
    """
    Coordinate descent + random perturbations to improve a candidate.
    Returns improved candidate and its score.
    """
    best_h = h.copy()
    best_score = _score_autoconv(best_h)
    
    # Scale to avoid numerical issues
    if np.max(best_h) > 0:
        best_h = best_h / np.max(best_h)
        best_score = _score_autoconv(best_h)
    
    for _ in range(max_iter):
        if _remaining() <= 5.0:
            break
        
        # Try random coordinate updates
        n_trials = rng.randint(10, 50)
        for _t in range(n_trials):
            i = rng.randint(0, N)
            old_val = best_h[i]
            
            # Try different multipliers
            mult = rng.choice([0.5, 0.8, 1.2, 1.5, 2.0, 0.0])
            new_val = old_val * mult
            
            best_h[i] = new_val
            new_score = _score_autoconv(best_h)
            
            if new_score < best_score - 1e-12:
                best_score = new_score
            else:
                best_h[i] = old_val
        
        # Try small random perturbations
        if rng.random() < 0.3:
            perturb_idx = rng.choice(N, size=rng.randint(1, 10), replace=False)
            perturb_amount = rng.uniform(-0.1, 0.1, size=len(perturb_idx))
            old_vals = best_h[perturb_idx].copy()
            best_h[perturb_idx] = np.clip(best_h[perturb_idx] + perturb_amount, 0, None)
            new_score = _score_autoconv(best_h)
            if new_score < best_score - 1e-12:
                best_score = new_score
            else:
                best_h[perturb_idx] = old_vals
        
        # Occasionally try a larger structured update
        if rng.random() < 0.1:
            # Try shifting mass from high to low values
            sort_idx = np.argsort(best_h)
            n_move = rng.randint(1, 10)
            low_idx = sort_idx[:n_move]
            high_idx = sort_idx[-n_move:]
            amount = rng.uniform(0.01, 0.1) * np.max(best_h)
            old_lows = best_h[low_idx].copy()
            old_highs = best_h[high_idx].copy()
            best_h[low_idx] += amount
            best_h[high_idx] -= amount
            best_h = np.clip(best_h, 0, None)
            new_score = _score_autoconv(best_h)
            if new_score < best_score - 1e-12:
                best_score = new_score
            else:
                best_h[low_idx] = old_lows
                best_h[high_idx] = old_highs
    
    return best_h, best_score


def construct():
    """Return 600 nonnegative step heights on equal intervals of [-1/4, 1/4]."""
    if _remaining() <= 0.0:
        # Fallback: uniform box (score 1.0)
        return [1.0] * N
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = float('inf')
    
    # Generate initial diverse candidates
    n_initial = 20
    for i in range(n_initial):
        if _remaining() <= 10.0:
            break
        h = _random_candidate(rng)
        score = _score_autoconv(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Also try some parametric sharp peaks
    for peak_width in [5, 10, 20, 50, 100]:
        if _remaining() <= 10.0:
            break
        h = np.zeros(N)
        center = N // 2
        for i in range(N):
            dist = abs(i - center)
            if dist < peak_width:
                h[i] = 1.0 - (dist / peak_width)
        h += rng.random(N) * 0.05
        score = _score_autoconv(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # If we have a good candidate, refine it
    if best_h is not None and _remaining() > 20.0:
        refined_h, refined_score = _local_search(best_h, rng, max_iter=1000)
        if refined_score < best_score:
            best_h = refined_h
            best_score = refined_score
    
    # Try a few more random restarts with local search if time permits
    n_restarts = 5
    for i in range(n_restarts):
        if _remaining() <= 20.0:
            break
        h = _random_candidate(rng)
        refined_h, refined_score = _local_search(h, rng, max_iter=300)
        if refined_score < best_score:
            best_h = refined_h
            best_score = refined_score
    
    if best_h is None:
        return [1.0] * N
    
    # Ensure nonnegativity and convert to list
    best_h = np.clip(best_h, 0, None)
    return [float(v) for v in best_h]
