"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np


def _value(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _spike_pattern(center, width, amplitude, N=N):
    """Create a spike-like pattern centered at 'center' with given width and amplitude."""
    h = np.zeros(N)
    # Gaussian-like spike
    x = np.arange(N)
    h = amplitude * np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
    return h


def _two_spike_pattern(c1, c2, w1, w2, a1, a2, N=N):
    """Create a two-spike pattern."""
    x = np.arange(N)
    h = a1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2)) + \
        a2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    return h


def _uniform_block_pattern(start, length, amplitude, N=N):
    """Create a uniform block pattern."""
    h = np.zeros(N)
    h[start:start + length] = amplitude
    return h


def _random_pattern(seed, N=N):
    """Create a random pattern with some structure."""
    rng = np.random.RandomState(seed)
    h = rng.randn(N)
    # Apply some smoothing
    kernel = np.ones(3) / 3.0
    h = np.convolve(h, kernel, mode='same')
    return h


def _coordinate_descent(h, max_iter=50, tol=1e-8, deadline=None):
    """Coordinate descent to minimize the autoconvolution peak."""
    h = h.copy()
    best_val = _value(h)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            old_h = h[i]
            # Try small perturbations
            for delta in [0.01, -0.01, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0]:
                h[i] = old_h + delta
                val = _value(h)
                if val < best_val - tol:
                    best_val = val
                    improved = True
                    break
            h[i] = old_h  # Reset if not improved
        
        if not improved:
            break
    
    return h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # Should be 2.0
    
    # Try various patterns
    patterns = []
    
    # Single spikes at different positions and widths
    for center in [50, 100, 150, 200, 250, 300, 350]:
        for width in [10, 20, 30, 50]:
            for amplitude in [1.0, 2.0, 5.0]:
                patterns.append(_spike_pattern(center, width, amplitude))
    
    # Two-spike patterns
    for c1 in [100, 150, 200]:
        for c2 in [200, 250, 300]:
            if abs(c1 - c2) < 50:
                continue
            for w in [15, 25, 40]:
                for a in [1.0, 2.0]:
                    patterns.append(_two_spike_pattern(c1, c2, w, w, a, a))
    
    # Uniform blocks
    for start in [0, 50, 100, 150, 200, 250, 300]:
        for length in [50, 100, 150]:
            if start + length > N:
                continue
            for amplitude in [1.0, 2.0, 3.0]:
                patterns.append(_uniform_block_pattern(start, length, amplitude))
    
    # Random patterns
    for seed in range(20):
        patterns.append(_random_pattern(seed))
    
    # Evaluate all patterns
    for h in patterns:
        if time.time() > deadline:
            break
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Coordinate descent on the best pattern
    if time.time() < deadline - 20.0:
        best_h, best_val = _coordinate_descent(best_h, max_iter=30, tol=1e-6, deadline=deadline - 5.0)
    
    # Final check
    final_val = _value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
# EVOLVE-BLOCK-END
