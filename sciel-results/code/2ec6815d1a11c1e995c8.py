"""
Phase-Only Retrieval (POR) Strategy:
1. Fix the power spectrum to a constant magnitude (spectral flatness).
2. Optimize the complex phase vector using Stochastic Gradient Descent (SGD)
   with a smooth max surrogate to minimize the maximum absolute autoconvolution.
3. Ensure the resulting time-domain signal is real and has a non-zero sum.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _soft_max_abs(conv, beta=100.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _smooth_metric(x, beta=50.0):
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)

def _try_phase_only_retrieval(deadline, rng):
    """
    Optimize phases phi[k] for k in 1..K-1.
    Fix m[k] = 1 for all k (flat spectrum).
    X[k] = 1 * exp(i * phi[k]).
    X[0] is fixed to be real and positive to ensure sum(x) > 0.
    """
    K = _FFT_SIZE // 2 + 1
    
    # Initialize phases randomly
    phi = rng.uniform(-np.pi, np.pi, size=K)
    phi[0] = 0.0  # Fix DC phase to 0
    
    # Fixed magnitudes
    m = np.ones(K)
    
    best_phi = phi.copy()
    best_val = float('inf')
    
    # SGD parameters
    lr = 0.01
    beta = 50.0
    batch_size = 10
    max_iter = 20000
    
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Construct X
        X = m * np.exp(1j * phi)
        
        # Compute autoconv
        conv = np.fft.irfft(X * X, _FFT_SIZE)[:2 * N - 1]
        
        # Compute smooth max
        abs_conv = np.abs(conv)
        max_val = np.max(abs_conv)
        shifted = abs_conv - max_val
        soft_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
        
        # Compute integral
        s = np.sum(np.fft.irfft(X, _FFT_SIZE)[:N])
        
        # Metric
        val = soft_max / (s * s)
        
        if val < best_val:
            best_val = val
            best_phi = phi.copy()
            
        # Compute gradient w.r.t. phi
        # d(soft_max)/d(phi[k]) = sum_j (beta * exp(beta * (abs_conv[j] - max_val)) / sum(exp(...))) * d(abs_conv[j])/d(phi[k])
        # This is complex. Let's use finite differences for simplicity in SGD.
        # Actually, finite differences are expensive. Let's try a few random perturbations.
        
        # Random coordinate descent / SGD step
        idxs = rng.integers(1, K, size=batch_size)
        
        for idx in idxs:
            # Try small perturbation
            delta = rng.uniform(-0.1, 0.1)
            phi_new = phi.copy()
            phi_new[idx] += delta
            
            X_new = m * np.exp(1j * phi_new)
            conv_new = np.fft.irfft(X_new * X_new, _FFT_SIZE)[:2 * N - 1]
            abs_conv_new = np.abs(conv_new)
            max_val_new = np.max(abs_conv_new)
            shifted_new = abs_conv_new - max_val_new
            soft_max_new = max_val_new + np.log(np.sum(np.exp(beta * shifted_new))) / beta
            s_new = np.sum(np.fft.irfft(X_new, _FFT_SIZE)[:N])
            val_new = soft_max_new / (s_new * s_new)
            
            if val_new < val:
                phi = phi_new
                val = val_new
            else:
                # If it got worse, revert (or try opposite direction)
                phi_new[idx] -= 2 * delta
                X_new2 = m * np.exp(1j * phi_new)
                conv_new2 = np.fft.irfft(X_new2 * X_new2, _FFT_SIZE)[:2 * N - 1]
                abs_conv_new2 = np.abs(conv_new2)
                max_val_new2 = np.max(abs_conv_new2)
                shifted_new2 = abs_conv_new2 - max_val_new2
                soft_max_new2 = max_val_new2 + np.log(np.sum(np.exp(beta * shifted_new2))) / beta
                s_new2 = np.sum(np.fft.irfft(X_new2, _FFT_SIZE)[:N])
                val_new2 = soft_max_new2 / (s_new2 * s_new2)
                
                if val_new2 < val:
                    phi = phi_new
                    val = val_new2

    # Construct final x
    X_final = m * np.exp(1j * best_phi)
    x = np.fft.irfft(X_final, _FFT_SIZE)[:N]
    
    # Ensure realness (it should be, but just in case)
    x = np.real(x)
    
    return x, _metric(x)

def _try_local_search_fast(x, deadline, rng, max_rounds=100000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 150:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val

def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.05, 0.05
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Phase-Only Retrieval
    x_por, val_por = _try_phase_only_retrieval(deadline - 30, rng)
    
    # Phase 2: Local Search
    if time.time() < deadline - 15:
        x_por, val_por = _try_local_search_fast(x_por, deadline - 15, rng, max_rounds=50000)
    
    # Phase 3: Final polish
    if time.time() < deadline - 2:
        x_por, val_por = _final_polish(x_por, deadline - 2, rng)
    
    return x_por
# EVOLVE-BLOCK-END
