"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _overlap_max(h):
    """Compute maximum overlap of h with 1-h over all shifts."""
    n = len(h)
    # h is a step function on [0, 2] with N steps, each of width 2/N
    # Overlap at shift k (in steps): sum_i h[i] * (1 - h[(i+k) % n]) * (2/N)
    # = (2/N) * sum_i h[i] - (2/N) * sum_i h[i]*h[(i+k)%n]
    # Since sum h = 1 (total integral), sum_i h[i] = N/2
    # Overlap at shift k = (2/N)*(N/2) - (2/N)*autocorr[k] = 1 - (2/N)*autocorr[k]
    # Wait, let me reconsider. The integral of h over [0,2] is 1.
    # Each step has width 2/N. So sum_i h[i] * (2/N) = 1, meaning sum h[i] = N/2.
    # Overlap at shift k: integral of h(x) * (1-h(x+k)) dx over [0,2]
    # = integral h(x) dx - integral h(x)h(x+k) dx
    # = 1 - (2/N) * sum_i h[i] * h[(i+k)%n]
    # So overlap[k] = 1 - (2/N) * autocorr[k]
    # To minimize max overlap, maximize min autocorr.
    
    autocorr = np.correlate(h, h, mode='full')[n-1:]  # lags 0 to n-1
    # autocorr[k] = sum_i h[i] * h[(i+k) % n] for k=0..n-1
    # Actually np.correlate gives different indexing. Let me use FFT.
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[i] * h[(i-k) % n] = sum_i h[i] * h[(i+k) % n] by symmetry
    overlap = 1.0 - (2.0 / n) * autocorr
    return np.max(overlap)

def _construct_from_params(params, N):
    """Construct h from parameters: base 0.5 + sum of sinusoidal modes."""
    h = np.full(N, 0.5)
    # params: [amp1, phase1, amp2, phase2, amp3, phase3, ...]
    for i in range(len(params) // 2):
        freq = i + 1
        amp = params[2*i]
        phase = params[2*i + 1]
        h += amp * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
    # Clamp to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    # Rescale so sum = N/2
    s = h.sum()
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0.0, 1.0)
    # Final adjustment: if clipping changed sum, do a simple redistribution
    diff = N / 2.0 - h.sum()
    if abs(diff) > 1e-10:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try different numbers of modes
    n_modes_list = [1, 2, 3, 4]
    
    for n_modes in n_modes_list:
        if time.time() > deadline - 10:
            break
        
        n_params = 2 * n_modes
        bounds = [(-0.5, 0.5)] * n_modes + [(0, 2*np.pi)] * n_modes
        
        # Multi-start with random initializations
        n_starts = 8
        for start in range(n_starts):
            if time.time() > deadline - 5:
                break
            
            if start == 0:
                x0 = np.array([0.1 * (1 if i % 2 == 0 else 0.5) for i in range(n_params)])
            else:
                x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            
            def objective(params):
                h = _construct_from_params(params, N)
                return _overlap_max(h)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                h = _construct_from_params(res.x, N)
                score = _overlap_max(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
            except Exception:
                pass
        
        # Also try differential evolution for small n_modes
        if n_modes <= 2 and time.time() < deadline - 20:
            try:
                def objective(params):
                    h = _construct_from_params(params, N)
                    return _overlap_max(h)
                
                res = differential_evolution(objective, bounds, maxiter=100, 
                                             popsize=15, tol=1e-8, seed=42)
                h = _construct_from_params(res.x, N)
                score = _overlap_max(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
            except Exception:
                pass
    
    return best_h
# EVOLVE-BLOCK-END
