"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

The metric is: score = max_k |autoconv[k]| / (sum f)^2
where autoconv[k] = sum_i f[i] * f[k - i] (linear autoconvolution).

Key structural change: For autoconvolution (not autocorrelation), signed
sequences can dramatically reduce the peak relative to the squared sum.
The constant vector is NOT optimal here. We search for balanced signed
sequences (±1) where the self-convolution is as flat as possible.

Strategy: Use simulated annealing / greedy local search over ±1 sequences
to minimize max|autoconv| / (sum f)^2.
"""
import numpy as np
import time
import random

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def _autoconv(f):
    """Linear autoconvolution of f with itself."""
    return np.convolve(f, f, mode='full')


def _score(f):
    """Compute max|autoconv| / (sum f)^2. Returns inf if sum f == 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(f)
    return np.max(np.abs(ac)) / (s * s)


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    """
    # Start with a random ±1 sequence
    f = np.random.choice([-1.0, 1.0], size=N)
    best_f = f.copy()
    best_score = _score(best_f)
    
    # Also try a few structured initializations
    candidates = [
        np.random.choice([-1.0, 1.0], size=N),
        np.random.choice([-1.0, 1.0], size=N),
        np.random.choice([-1.0, 1.0], size=N),
        # Alternating
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        # Block structure
        np.concatenate([np.ones(N//4), -np.ones(N//4), np.ones(N//4), -np.ones(N//4)]),
    ]
    
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_f = c.copy()
    
    # Simulated annealing local search
    rng = np.random.RandomState(42)
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    # Precompute for speed: use FFT-based autoconvolution
    def _autoconv_fft(f):
        n = len(f)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        F = np.fft.rfft(f, size)
        ac = np.fft.irfft(F * F, size)
        return ac[:2*n-1]
    
    def _score_fast(f):
        s = np.sum(f)
        if abs(s) < 1e-12:
            return np.inf
        ac = _autoconv_fft(f)
        return np.max(np.abs(ac)) / (s * s)
    
    best_score = _score_fast(best_f)
    f = best_f.copy()
    cur_score = best_score
    
    iterations = 0
    while _remaining() > 5.0 and T > T_min:
        # Flip a random position
        idx = rng.randint(N)
        old_val = f[idx]
        f[idx] = -f[idx]
        
        new_score = _score_fast(f)
        
        if new_score < cur_score or rng.random() < np.exp(-(new_score - cur_score) / T):
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            f[idx] = old_val
        
        T *= cooling
        iterations += 1
        if iterations % 5000 == 0:
            # Occasionally try multi-flip moves
            for _ in range(5):
                idx1 = rng.randint(N)
                idx2 = rng.randint(N)
                if idx1 == idx2:
                    continue
                f[idx1] = -f[idx1]
                f[idx2] = -f[idx2]
                new_score = _score_fast(f)
                if new_score < cur_score:
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_f = f.copy()
                else:
                    f[idx1] = -f[idx1]
                    f[idx2] = -f[idx2]
    
    # Final: try to improve by flipping individual bits greedily
    f = best_f.copy()
    cur_score = _score_fast(f)
    improved = True
    while improved and _remaining() > 2.0:
        improved = False
        for idx in range(0, N, 1):
            f[idx] = -f[idx]
            new_score = _score_fast(f)
            if new_score < cur_score:
                cur_score = new_score
                improved = True
            else:
                f[idx] = -f[idx]
        if cur_score < best_score:
            best_score = cur_score
            best_f = f.copy()
    
    return best_f
