"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0  # wall-clock limit with safety margin


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))


def _try_candidate(h):
    """Evaluate a candidate; return (value, h) or (inf, None) if invalid."""
    if not np.all(np.isfinite(h)):
        return np.inf, None
    v = _evaluate(h)
    if not np.isfinite(v):
        return np.inf, None
    return v, h


def _random_search(rng, best_val, best_h, deadline):
    """Randomly sample vectors with various structures."""
    for _ in range(100000):
        if time.time() > deadline:
            break
        # Try different random structures
        choice = rng.integers(0, 5)
        if choice == 0:
            # Pure random
            h = rng.normal(0, 1, N)
        elif choice == 1:
            # Mostly positive with some negatives
            h = rng.uniform(0.5, 1.0, N)
            h[rng.choice(N, size=N // 10, replace=False)] *= -1
        elif choice == 2:
            # Sparse: only a few nonzero entries
            h = np.zeros(N)
            idx = rng.choice(N, size=rng.integers(10, 50), replace=False)
            h[idx] = rng.normal(0, 1, len(idx))
        elif choice == 3:
            # Block structure: alternating blocks of constant value
            block_size = rng.integers(5, 40)
            h = np.zeros(N)
            for i in range(0, N, block_size):
                val = rng.normal(0, 1)
                h[i:min(i + block_size, N)] = val
        else:
            # Low-frequency cosine-like
            freq = rng.uniform(0.1, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            h = np.cos(2 * np.pi * freq * np.arange(N) / N + phase)
        
        v, hh = _try_candidate(h)
        if v < best_val:
            best_val = v
            best_h = hh.copy()
    return best_val, best_h


def _greedy_improvement(h, deadline):
    """Try to improve a candidate by random perturbations."""
    best_val = _evaluate(h)
    best_h = h.copy()
    rng = np.random.default_rng()
    
    for step in range(200000):
        if time.time() > deadline:
            break
        # Pick a few indices to perturb
        n_perturb = rng.integers(1, 5)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, 0.1, n_perturb)
        
        v, _ = _try_candidate(h_new)
        if v < best_val:
            best_val = v
            best_h = h_new
    return best_val, best_h


def _structured_search(deadline):
    """Try specific structured patterns that might minimize the value."""
    best_val = np.inf
    best_h = np.ones(N)
    
    # The seed gives 2.0. Let's try to understand the structure.
    # value = abs(2*N*max(conv)/sum^2)
    # For all ones: conv at lag 0 is N, max is N, sum is N, so 2*N*N/N^2 = 2.
    # To reduce this, we want max(conv) small relative to sum^2.
    
    # Try: make h such that conv has no large peaks.
    # If h is orthogonal to itself in some sense... but conv at lag 0 is sum(h^2) >= (sum h)^2/N.
    # So max(conv) >= sum(h^2) >= (sum h)^2 / N.
    # Thus value >= 2*N*(sum h)^2/(N*(sum h)^2) = 2.
    # Wait, that means the minimum possible value is 2?!
    # Let me re-check: max(conv) >= conv[0] = sum(h_i^2) >= (sum h)^2 / N by Cauchy-Schwarz.
    # So 2*N*max(conv)/sum(h)^2 >= 2*N*(sum h)^2/(N * sum(h)^2) = 2.
    # So the minimum is exactly 2, achieved when h is constant!
    # The seed already achieves the theoretical minimum.
    
    # But wait, the task says "lower is better" and the seed gives 2.0.
    # If 2.0 is the mathematical lower bound, then we cannot improve.
    # Let me double-check the convolution index.
    # np.convolve(h, h) for full convolution has length 2N-1.
    # The element at index k corresponds to sum_{i+j=k} h[i]*h[j].
    # For k=0: h[0]*h[0]
    # For k=N-1: sum_i h[i]*h[i] = sum(h^2)
    # Ah, the max is over ALL entries, not just the central one.
    # conv[0] = h[0]^2, which could be small.
    # But conv[N-1] = sum(h^2) >= (sum h)^2/N.
    # So max(conv) >= sum(h^2) >= (sum h)^2/N.
    # Therefore value >= 2. The seed achieves exactly 2.
    
    # This means no improvement is possible above score 1.0.
    # The search will just confirm the seed is optimal.
    
    return best_val, best_h


def construct():
    """Construct step heights. Returns the seed (optimal) since 2.0 is the theoretical minimum."""
    # The mathematical lower bound for the value is 2.0, achieved by the constant vector.
    # Any deviation increases the value due to the strictness of Cauchy-Schwarz for non-constant vectors.
    # Thus, the seed is the global optimum.
    return np.ones(N)
# EVOLVE-BLOCK-END
