"""Improved program using chirp-sign structure for autoconvolution minimization."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

def _autoconv_max_abs(x):
    """Compute max absolute value of autoconvolution away from center (or all)."""
    # Autoconvolution via FFT
    n = len(x)
    n_fft = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n_fft)
    ac = np.fft.irfft(X * np.conj(X), n_fft)[:n]
    # ac[0] is the peak (n * mean^2 roughly), we want to minimize the sidelobe max
    # The "upper bound on C3" likely involves the max off-peak autocorrelation
    # Let's use max of |ac[k]| for k != 0
    if len(ac) > 1:
        return np.max(np.abs(ac[1:]))
    return 0.0

def _chirp_sign_signal(w1, w2, phase1=0.0, phase2=0.0, n=N):
    """Generate signed step function as sign of sum of two chirps."""
    t = np.arange(n, dtype=np.float64)
    sig1 = np.cos(w1 * t + phase1)
    sig2 = np.cos(w2 * t * t + phase2)
    s = sig1 + sig2
    # Sign function: +1 or -1
    return np.sign(s).astype(np.float64)

def _objective(params):
    w1, w2, p1, p2 = params
    x = _chirp_sign_signal(w1, w2, p1, p2)
    return _autoconv_max_abs(x)

def construct():
    start_time = time.time()
    
    # Initial: all ones (seed behavior)
    best_x = np.ones(N)
    best_val = _autoconv_max_abs(best_x)
    
    # Try a grid of chirp parameters first (fast)
    # w1 in [0, 2*pi], w2 in [0, 2*pi/N] (quadratic phase)
    t = np.arange(N, dtype=np.float64)
    
    # Quick scan over a few promising frequency pairs
    w1_candidates = np.linspace(0, 2*np.pi, 20)
    w2_candidates = np.linspace(0, 4*np.pi/N, 20)
    
    for w1 in w1_candidates:
        if time.time() - start_time > DEADLINE - 30:
            break
        for w2 in w2_candidates:
            if time.time() - start_time > DEADLINE - 30:
                break
            x = _chirp_sign_signal(w1, w2)
            val = _autoconv_max_abs(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    
    # Local refinement with differential evolution
    if time.time() - start_time < DEADLINE - 20:
        try:
            bounds = [(0, 2*np.pi), (0, 4*np.pi/N), (0, 2*np.pi), (0, 2*np.pi)]
            result = differential_evolution(
                _objective, bounds,
                maxiter=50, popsize=15,
                tol=1e-6, seed=42,
                polish=True
            )
            if time.time() - start_time < DEADLINE - 5:
                w1, w2, p1, p2 = result.x
                x = _chirp_sign_signal(w1, w2, p1, p2)
                val = _autoconv_max_abs(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
        except Exception:
            pass
    
    # Final local search: flip individual bits to reduce max autocorrelation
    if time.time() - start_time < DEADLINE - 10:
        x = best_x.copy()
        val = _autoconv_max_abs(x)
        improved = True
        while improved and time.time() - start_time < DEADLINE - 10:
            improved = False
            # Try flipping random positions
            for _ in range(200):
                if time.time() - start_time > DEADLINE - 10:
                    break
                idx = np.random.randint(N)
                x[idx] *= -1
                new_val = _autoconv_max_abs(x)
                if new_val < val:
                    val = new_val
                    best_x = x.copy()
                    best_val = val
                    improved = True
                else:
                    x[idx] *= -1  # revert
    
    return best_x

# EVOLVE-BLOCK-START
# (construct is defined above, this block is for structural compatibility)
# EVOLVE-BLOCK-END
