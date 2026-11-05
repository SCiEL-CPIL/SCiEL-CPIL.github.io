"""
Strategy: Structured combinatorial constructions + exact per-coordinate line search refinement.

Instead of smooth-surrogate gradient/coordinate descent, we:
1. Generate many diverse initial candidates from explicit low-peak autoconvolution structures
   (Barker-like binary, chirp/quadratic phase, random phase, alternating blocks, sign patterns).
2. Refine each candidate using EXACT line search on individual coordinates:
   for coordinate i, the autoconvolution is quadratic in h[i], so we can find the exact
   minimizer of max(conv) (or a smoothed proxy) by evaluating a small set of candidate values.
3. Keep the best candidate by the true task value.

This is structurally different from the current program:
- No log-sum-exp surrogate or L-BFGS.
- Uses explicit combinatorial/structured initializations.
- Uses exact (or near-exact) line search per coordinate rather than gradient-based updates.
"""

import numpy as np
import time
import itertools

N = 400
_DEADLINE = 145.0  # Safety margin before 150s kill


def _evaluate(h):
    """Compute the task value exactly."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _normalize(h, target_sum=None):
    """Normalize h so sum = target_sum (default N)."""
    if target_sum is None:
        target_sum = N
    s = np.sum(h)
    if abs(s) < 1e-12:
        return h
    return h * target_sum / s


def _autoconv_peak(h):
    """Return max of autoconvolution of h (after normalizing sum to N)."""
    h = _normalize(h)
    conv = np.convolve(h, h)
    return np.max(conv)


def _line_search_coordinate(h, i, n_samples=200):
    """
    For coordinate i, find a better value by sampling.
    The autoconvolution is quadratic in h[i], so we can evaluate max(conv) at
    several candidate values and pick the best.
    """
    h_orig = h[i]
    # Sample a range of values around h_orig
    # Scale: typical h values are O(1) after normalization
    span = max(1.0, abs(h_orig) * 2.0)
    candidates = np.linspace(h_orig - span, h_orig + span, n_samples)
    
    best_val = _autoconv_peak(h)
    best_h = h_orig
    
    for new_val in candidates:
        h[i] = new_val
        v = _autoconv_peak(h)
        if v < best_val:
            best_val = v
            best_h = new_val
    
    h[i] = best_h
    return best_h


def _local_refine(h, n_passes=3, n_samples=100, deadline=None):
    """
    Refine h using exact line search on coordinates.
    """
    h = h.copy()
    h = _normalize(h)
    best_val = _autoconv_peak(h)
    
    for p in range(n_passes):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Random order to avoid systematic bias
        order = np.random.permutation(N)
        
        for idx in order:
            if deadline is not None and time.time() > deadline - 5:
                break
            
            old_h = h[idx]
            h[idx] = _line_search_coordinate(h, idx, n_samples=n_samples)
            
            # Check improvement
            v = _autoconv_peak(h)
            if v < best_val:
                best_val = v
            else:
                # Revert if no improvement
                h[idx] = old_h
    
    return h, best_val


def _barker_like_candidate(rng, n_modes=5):
    """
    Generate a Barker-like binary candidate with some smoothness.
    Barker sequences have low autocorrelation peaks.
    """
    # Start with random signs
    h = np.sign(rng.standard_normal(N))
    
    # Apply some smoothing to reduce high-frequency content
    # Use a simple moving average
    window = 3
    kernel = np.ones(window) / window
    h_smooth = np.convolve(h, kernel, mode='same')
    
    # Quantize back to binary-like values
    h = np.sign(h_smooth)
    
    # Add some variation
    for _ in range(n_modes):
        idx = rng.integers(0, N)
        h[idx] *= -1
    
    return h


def _chirp_candidate(rng, freq=1.0, phase=0.0):
    """
    Generate a chirp/quadratic phase candidate.
    """
    x = np.linspace(-0.5, 0.5, N)
    h = np.cos(2 * np.pi * freq * x * x + phase)
    return h


def _random_phase_candidate(rng):
    """
    Generate a random phase candidate.
    """
    phases = rng.uniform(0, 2 * np.pi, N)
    h = np.cos(phases)
    return h


def _alternating_blocks_candidate(rng, n_blocks=10):
    """
    Generate alternating block candidate.
    """
    block_size = N // n_blocks
    h = np.zeros(N)
    sign = 1.0
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, N)
        h[start:end] = sign
        sign *= -1
    
    # Add some random perturbation
    h += rng.standard_normal(N) * 0.1
    
    return h


def _sign_pattern_candidate(rng):
    """
    Generate a sign pattern candidate with low-frequency bias.
    """
    # Low-frequency sign pattern
    x = np.linspace(-0.5, 0.5, N)
    h = np.sign(np.sin(2 * np.pi * x))
    
    # Add random perturbation
    h += rng.standard_normal(N) * 0.2
    
    return h


def _generate_candidates(rng, n_candidates=20):
    """
    Generate diverse initial candidates.
    """
    candidates = []
    
    # Barker-like
    for i in range(5):
        candidates.append(_barker_like_candidate(rng, n_modes=5 + i))
    
    # Chirp
    for freq in [0.5, 1.0, 2.0, 3.0]:
        for phase in [0.0, np.pi / 2, np.pi]:
            candidates.append(_chirp_candidate(rng, freq=freq, phase=phase))
    
    # Random phase
    for i in range(3):
        candidates.append(_random_phase_candidate(rng))
    
    # Alternating blocks
    for n_blocks in [5, 10, 20]:
        candidates.append(_alternating_blocks_candidate(rng, n_blocks=n_blocks))
    
    # Sign patterns
    for i in range(3):
        candidates.append(_sign_pattern_candidate(rng))
    
    # Truncate to requested number
    return candidates[:n_candidates]


def construct():
    """
    Construct step heights using structured combinatorial search + local refinement.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Generate diverse candidates
    candidates = _generate_candidates(rng, n_candidates=30)
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline - 30:
            break
        
        cand = _normalize(cand)
        v = _evaluate(cand)
        
        if v < best_val:
            best_val = v
            best_h = cand.copy()
    
    # Local refinement on the best candidate
    if time.time() < deadline - 20:
        h_refined, _ = _local_refine(best_h, n_passes=3, n_samples=100, deadline=deadline)
        v_refined = _evaluate(h_refined)
        
        if v_refined < best_val:
            best_val = v_refined
            best_h = h_refined
    
    # Try a few more random perturbations + refinement
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            
            h_perturbed = best_h + rng.standard_normal(N) * 0.1
            h_perturbed = _normalize(h_perturbed)
            
            h_refined, _ = _local_refine(h_perturbed, n_passes=1, n_samples=50, deadline=deadline)
            v_refined = _evaluate(h_refined)
            
            if v_refined < best_val:
                best_val = v_refined
                best_h = h_refined
    
    # Final check
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h


# EVOLVE-BLOCK-END
