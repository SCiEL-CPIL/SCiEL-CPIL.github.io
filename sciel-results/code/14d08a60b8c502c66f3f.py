import numpy as np
import time

N = 400

def _initial_autocorr(h):
    """Compute full autocorrelation r[0..N-1] where r[k] = sum_i h[i]*h[i+k]."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    return r[:N]  # r[0..N-1]

def _peak(r):
    """Max |r[k]| for k=1..N-1."""
    return np.max(np.abs(r[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = None
    best_peak = np.inf
    
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        np.random.seed(42 + n_restarts)
        h = np.random.choice([-1, 1], N).astype(np.float64)
        
        r = _initial_autocorr(h)
        current_peak = _peak(r)
        if current_peak < best_peak:
            best_peak = current_peak
            best_h = h.copy()
        
        # Local search with incremental updates
        improved = True
        no_improve_count = 0
        while improved and time.time() < deadline - 5.0:
            improved = False
            no_improve_count += 1
            if no_improve_count > 50:
                break
            
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                old_val = h[i]
                h[i] = -old_val
                
                # Update r[k] for all k
                # Delta r[k] = -2*old_val*(h[i+k] + h[i-k]) with bounds
                # r[k] = sum_{j} h[j]*h[j+k]
                # When h[i] changes by -2*old_val, terms involving h[i] change
                # For lag k: terms where j=i (h[i]*h[i+k]) and j=i-k (h[i-k]*h[i])
                # So delta r[k] = -2*old_val * (h[i+k] if i+k<N else 0) + (-2*old_val)*(h[i-k] if i-k>=0 else 0)
                
                delta = np.zeros(N)
                # Terms where h[i] is the first factor: j=i, h[i]*h[i+k]
                if i + 1 < N:
                    delta[1:N-i] += -2 * old_val * h[i+1:i+N]
                # Terms where h[i] is the second factor: j=i-k, h[i-k]*h[i], so k = i-j, j from 0 to i-1
                if i >= 1:
                    delta[1:i+1] += -2 * old_val * h[i-1::-1]
                
                r_new = r + delta
                new_peak = _peak(r_new)
                
                if new_peak < current_peak - 1e-12:
                    current_peak = new_peak
                    r = r_new
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_h = h.copy()
                    improved = True
                else:
                    h[i] = old_val
        
        if n_restarts >= 20:
            break
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
