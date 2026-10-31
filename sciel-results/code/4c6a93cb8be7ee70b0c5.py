import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def autoconv_max_abs(x):
        """Compute max absolute autoconvolution using FFT for speed."""
        if np.all(x == 0):
            return 0.0
        ac = fftconvolve(x, x, mode='full')
        return np.max(np.abs(ac))

    def score(x):
        """max|autoconv| / (sum)^2 - lower is better."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return autoconv_max_abs(x) / (s * s)

    def local_optimize(h, max_passes=3, deadline=None):
        """Local optimization: try flipping signs and small value tweaks."""
        h = h.copy()
        best_val = score(h)
        for _ in range(max_passes):
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                # Try flipping sign
                h[i] = -h[i]
                new_val = score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = -h[i]  # revert
            if not improved:
                break
        return h, best_val

    best_h = np.ones(N)
    best_val = score(best_h)

    rng = np.random.default_rng(42)

    # PRIMARY: Deterministic tapered windows with sign patterns
    # These structurally minimize peak autoconvolution
    
    # 1. Raised cosine (Hanning) window
    x = np.arange(N)
    hann = 0.5 * (1 + np.cos(2 * np.pi * x / (N - 1)))
    # Try with all positive
    h = hann.copy()
    s = np.sum(h)
    if s > 1e-6:
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Try with sign modulation: positive in center, negative at edges
    for sign_pattern in [
        np.ones(N),
        np.sign(np.cos(np.pi * x / (N - 1))),  # single sign flip
        np.where(x < N//4, -1, np.where(x >= 3*N//4, -1, 1)),  # negative edges
    ]:
        h = hann * sign_pattern
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 2. Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * x / (N - 1))
    for sign_pattern in [np.ones(N), np.where(x < N//4, -1, np.where(x >= 3*N//4, -1, 1))]:
        h = hamming * sign_pattern
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 3. Triangular window
    tri = 1 - np.abs(2 * x / (N - 1) - 1)
    for sign_pattern in [np.ones(N), np.where(x < N//4, -1, np.where(x >= 3*N//4, -1, 1))]:
        h = tri * sign_pattern
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 4. Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * x / (N - 1)) + 0.08 * np.cos(4 * np.pi * x / (N - 1))
    for sign_pattern in [np.ones(N), np.where(x < N//4, -1, np.where(x >= 3*N//4, -1, 1))]:
        h = blackman * sign_pattern
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 5. Various smooth sign-modulated windows
    # Use cosine modulation at different frequencies
    for freq in [1, 2, 3, 5, 8]:
        mod = np.cos(2 * np.pi * freq * x / N)
        h = hann * mod
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        val = score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # SECONDARY: Local optimization on best tapered window
    if time.time() < deadline:
        h_opt, val_opt = local_optimize(best_h, max_passes=5, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # TERTIARY: Diverse random initializations with local sign-flip
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r % 5 == 0:
            # Random ±1
            h = rng.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            # Mostly +1 with some random -1
            h = np.ones(N)
            n_neg = rng.integers(N//4, N//2)
            neg_idx = rng.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 5 == 2:
            # Chirp-like pattern
            h = np.sign(np.cos(2 * np.pi * x * (x + 1) / (2 * N)))
        elif r % 5 == 3:
            # Random continuous
            h = rng.uniform(-1, 1, size=N)
        else:
            # Block pattern
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = rng.integers(1, N // 10 + 1)
                sign = rng.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            h = h + 0.5
            h = np.sign(h)
        
        h_opt, val_opt = local_optimize(h, max_passes=3, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    return best_h
