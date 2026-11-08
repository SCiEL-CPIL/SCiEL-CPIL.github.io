import numpy as np
from scipy.optimize import minimize
import time

N = 600
dx = 0.5 / N  # width of each interval

def compute_value(h):
    """Compute the C1 upper bound: max(autoconv) / (integral)^2"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    integral = np.sum(h) * dx
    if integral < 1e-10:
        return 1e10
    # Compute autoconvolution via FFT
    size = 2 * N
    f = np.zeros(size)
    f[:N] = h
    F = np.fft.fft(f)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    max_conv = np.max(autocorr) * dx
    return max_conv / (integral * integral)

def soft_max(arr, beta):
    """Smooth approximation to max"""
    arr = np.asarray(arr, dtype=float)
    m = np.max(arr)
    if m < 1e-10:
        return m
    return m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (arr - m))))

def objective_smooth(h, beta):
    """Smooth surrogate objective for optimization"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    integral = np.sum(h) * dx
    if integral < 1e-10:
        return 1e10
    size = 2 * N
    f = np.zeros(size)
    f[:N] = h
    F = np.fft.fft(f)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    max_conv_smooth = soft_max(autocorr, beta) * dx
    return max_conv_smooth / (integral * integral)

def peak_reduction(h, n_steps=50, deadline=None):
    """Targeted refinement: reduce contributions to the highest autocorrelation lags"""
    h = np.maximum(h.copy(), 0)
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Compute autocorrelation
        size = 2 * N
        f = np.zeros(size)
        f[:N] = h
        F = np.fft.fft(f)
        autocorr = np.fft.ifft(F * np.conj(F)).real
        
        # Find the peak lag
        k_max = np.argmax(autocorr)
        
        # Gradient of autocorr[k_max] w.r.t. h[j] = h[j + k_max] + h[j - k_max]
        # (with wrap-around handled by the FFT structure)
        # For non-circular, we need to be careful about boundaries
        g = np.zeros(N)
        for j in range(N):
            # Contribution from h[j] * h[j+k_max] and h[j+k_max] * h[j]
            j1 = j + k_max
            j2 = j - k_max
            if 0 <= j1 < N:
                g[j] += h[j1]
            if 0 <= j2 < N:
                g[j] += h[j2]
        
        # Also consider the symmetric peak at -k_max (which is N-k_max in circular)
        # But since autocorr is real and even, k_max and N-k_max give same magnitude
        # The gradient already accounts for this via the two terms
        
        # Subgradient step: reduce h where gradient is high
        eta = 0.05
        h_new = h - eta * g
        h_new = np.maximum(h_new, 0)
        
        # Renormalize to keep integral constant (since objective is scale-invariant)
        s = np.sum(h_new)
        if s < 1e-10:
            break
        h_new = h_new / s * np.sum(h)
        
        # Check if improved
        val_new = compute_value(h_new)
        val_old = compute_value(h)
        if val_new < val_old:
            h = h_new
        else:
            eta *= 0.5
            if eta < 1e-6:
                break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    bounds = [(0, None)] * N
    
    # Initial candidates
    candidates = [
        np.ones(N),
        np.exp(-10 * (np.linspace(-0.25, 0.25, N, endpoint=False) + dx/2)**2),
        np.cos(np.linspace(0, np.pi, N))**2,
        np.random.exponential(1.0, N)
    ]
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: L-BFGS-B with smooth surrogates
    betas = [20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
        
        for h0 in candidates:
            if time.time() > deadline - 30:
                break
            
            try:
                obj = lambda h: objective_smooth(h, beta)
                result = minimize(obj, h0, method='L-BFGS-B',
                                 bounds=bounds,
                                 options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-8})
                
                if time.time() > deadline - 30:
                    break
                
                h_candidate = np.maximum(result.x, 0)
                score = compute_value(h_candidate)
                
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
            except Exception:
                continue
    
    # Phase 2: Targeted peak reduction on the best candidate
    if best_h is not None and time.time() < deadline - 15:
        h_refined = peak_reduction(best_h, n_steps=100, deadline=deadline)
        score_refined = compute_value(h_refined)
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined
    
    # Phase 3: Final polish with L-BFGS-B on exact objective
    if best_h is not None and time.time() < deadline - 10:
        try:
            obj_exact = lambda h: compute_value(h)
            result = minimize(obj_exact, best_h, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-10})
            h_final = np.maximum(result.x, 0)
            score_final = compute_value(h_final)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
        except Exception:
            pass
    
    if best_h is None:
        best_h = np.ones(N)
    
    return [float(x) for x in best_h]
