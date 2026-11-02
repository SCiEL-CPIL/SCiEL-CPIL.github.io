import numpy as np
import time
from scipy.signal import fftconvolve

N = 400
DEADLINE = 145.0  # seconds, with safety margin

def _compute_c3prime(x):
    """Compute C3' = max(|x*x|) / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-15:
        return np.inf
    conv = fftconvolve(x, x, mode='full')
    return np.max(np.abs(conv)) / (s * s)

def _autoconv_fft(x):
    """Compute autoconvolution via FFT (full mode)."""
    n = len(x)
    size = 2 * n - 1
    X = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(X * X, n=size)
    return conv

def _max_abs_autoconv(x):
    """Return max |autoconv| and the sum."""
    conv = _autoconv_fft(x)
    return np.max(np.abs(conv)), np.sum(x)

def _flip_delta(x, idx):
    """
    Compute the new max|autoconv| and new sum if we flip x[idx].
    Uses FFT: flipping x[idx] changes X by -2*x[idx]*exp(-2πi*k*idx/size).
    We recompute the full autoconv via FFT for correctness but this is fast.
    """
    x_new = x.copy()
    x_new[idx] = -x_new[idx]
    s_new = np.sum(x_new)
    if abs(s_new) < 1e-15:
        return np.inf, s_new
    conv_new = _autoconv_fft(x_new)
    return np.max(np.abs(conv_new)), s_new

def construct():
    """
    Constructs a signed step function (±1 values) minimizing C3'.
    Uses discrete local search on the hypercube with simulated annealing
    and multi-start from diverse initializations.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_score = _compute_c3prime(best_x)
    
    def _time_ok(margin=5.0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    # Multi-start local search with simulated annealing
    n_starts = 8
    for start in range(n_starts):
        if not _time_ok(10.0):
            break
        
        # Diverse initializations
        if start == 0:
            x = np.ones(N)
        elif start == 1:
            x = np.ones(N)
            x[rng.choice(N, size=N//4, replace=False)] = -1
        elif start == 2:
            x = np.ones(N)
            x[rng.choice(N, size=N//2, replace=False)] = -1
        elif start == 3:
            x = np.ones(N)
            x[rng.choice(N, size=3*N//4, replace=False)] = -1
        else:
            x = rng.choice([-1, 1], size=N)
        
        s = np.sum(x)
        if abs(s) < 1e-15:
            continue
        conv = _autoconv_fft(x)
        max_abs = np.max(np.abs(conv))
        score = max_abs / (s * s)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Simulated annealing local search
        T = 0.5
        T_min = 0.01
        cooling = 0.999
        no_improve = 0
        max_no_improve = 2000
        
        for it in range(50000):
            if not _time_ok(5.0):
                break
            if no_improve > max_no_improve:
                break
            
            idx = rng.integers(0, N)
            new_max_abs, s_new = _flip_delta(x, idx)
            
            if abs(s_new) < 1e-15:
                no_improve += 1
                continue
            
            new_score = new_max_abs / (s_new * s_new)
            delta = new_score - score
            
            if delta < 0:
                # Accept improvement
                x[idx] = -x[idx]
                score = new_score
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_x = x.copy()
            else:
                # Accept with probability exp(-delta/T)
                if np.log(rng.random()) > -delta / T:
                    x[idx] = -x[idx]
                    score = new_score
                no_improve += 1
            
            T = max(T * cooling, T_min)
        
        # After SA, do a final greedy polish
        improved = True
        while improved and _time_ok(5.0):
            improved = False
            for idx in range(N):
                if not _time_ok(5.0):
                    break
                new_max_abs, s_new = _flip_delta(x, idx)
                if abs(s_new) < 1e-15:
                    continue
                new_score = new_max_abs / (s_new * s_new)
                if new_score < score:
                    x[idx] = -x[idx]
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_x = x.copy()
    
    return best_x
