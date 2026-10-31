"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_overlap(h, shift):
    """Compute overlap of h with shifted 1-h for given circular shift."""
    shifted_h = np.roll(h, shift)
    return np.sum(h * (1 - shifted_h))

def _max_overlap(h):
    """Maximum overlap over all shifts."""
    overlaps = np.array([_compute_overlap(h, s) for s in range(N)])
    return max(overlaps)

def _fft_max_overlap(h):
    """Fast computation of max overlap using FFT."""
    # overlap(s) = sum_i h[i] * (1 - h[(i-s) mod N])
    #           = sum_i h[i] - sum_i h[i]*h[(i-s) mod N]
    #           = N/2 - autocorrelation(s)
    # where autocorrelation(s) = sum_i h[i]*h[(i-s) mod N]
    # Use FFT for circular autocorrelation
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[s] = sum_i h[i]*h[(i+s) mod N] for s=0..N-1
    # We need sum_i h[i]*h[(i-s) mod N] = autocorr[N-s mod N]
    # Actually: autocorr[k] = sum_i h[i] * h[(i+k) mod N]
    # We want C[s] = sum_i h[i] * h[(i-s) mod N] = autocorr[N-s] for s>0, autocorr[0] for s=0
    # So overlaps[s] = N/2 - autocorr[N-s mod N]
    sum_h = np.sum(h)
    overlaps = sum_h - autocorr[np.mod(-np.arange(N), N)]
    return max(overlaps)

def _fourier_h(coefficients):
    """Construct h from Fourier coefficients (DC + a few harmonics)."""
    h = np.zeros(N)
    # coefficients[0] is DC (should be 0.5 for mean 0.5)
    for k, amp in enumerate(coefficients):
        if k == 0:
            h += amp
        else:
            h += amp * np.cos(2 * np.pi * k * np.arange(N) / N)
    return h

def construct():
    deadline = time.time() + 105  # Safety margin before 150s limit
    
    # Start with uniform
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _fft_max_overlap(h0)
    
    # Phase 1: Try structured Fourier-based constructions
    # h[i] = 0.5 + sum_k a_k * cos(2*pi*k*i/N + phi_k)
    # We want to minimize max_s sum_i h[i]*(1-h[(i-s)mod N])
    
    # Try single harmonic: h[i] = 0.5 + a*cos(2*pi*i/N + phi)
    # This is a simple wave. Let's optimize amplitude and phase.
    def _eval_fourier_single(a, phi):
        h = 0.5 + a * np.cos(2 * np.pi * np.arange(N) / N + phi)
        h = np.clip(h, 0, 1)
        # Renormalize to have sum exactly N/2
        h = h + (N/2 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        h = h + (N/2 - np.sum(h)) / N
        return _fft_max_overlap(h), h
    
    # Grid search over amplitude and phase for single harmonic
    for k in [1, 2, 3, 4, 5]:
        for a in np.linspace(0.0, 0.49, 20):
            for phi in np.linspace(0, 2*np.pi, 20, endpoint=False):
                val, h = _eval_fourier_single(a, phi)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
    
    if time.time() > deadline - 60:
        # Ensure exact sum and return
        best_h = best_h + (N/2 - np.sum(best_h)) / N
        best_h = np.clip(best_h, 0, 1)
        best_h = best_h + (N/2 - np.sum(best_h)) / N
        return best_h
    
    # Phase 2: Try two-harmonic constructions
    # h[i] = 0.5 + a1*cos(2*pi*i/N + phi1) + a2*cos(4*pi*i/N + phi2)
    def _eval_two_harmonic(a1, phi1, a2, phi2):
        h = (0.5 + a1 * np.cos(2 * np.pi * np.arange(N) / N + phi1) +
             a2 * np.cos(4 * np.pi * np.arange(N) / N + phi2))
        h = np.clip(h, 0, 1)
        h = h + (N/2 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        h = h + (N/2 - np.sum(h)) / N
        return _fft_max_overlap(h), h
    
    # Coarse grid for two harmonics
    for a1 in [0.1, 0.2, 0.3]:
        for phi1 in [0, np.pi/2, np.pi, 3*np.pi/2]:
            for a2 in [0.1, 0.2, 0.3]:
                for phi2 in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    val, h = _eval_two_harmonic(a1, phi1, a2, phi2)
                    if val < best_val:
                        best_val = val
                        best_h = h.copy()
    
    if time.time() > deadline - 30:
        best_h = best_h + (N/2 - np.sum(best_h)) / N
        best_h = np.clip(best_h, 0, 1)
        best_h = best_h + (N/2 - np.sum(best_h)) / N
        return best_h
    
    # Phase 3: Local optimization from best Fourier solution
    def objective(h_flat):
        h = np.clip(h_flat, 0, 1)
        h = h + (N/2 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        # Soft-max for smoothness
        overlaps = np.array([_compute_overlap(h, s) for s in range(N)])
        beta = 30.0
        return np.logsumexp(beta * overlaps) / beta
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    bounds = [(0, 1)] * N
    
    try:
        res = minimize(objective, best_h, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})
        if res.success:
            h_res = res.x
            h_res = np.clip(h_res, 0, 1)
            h_res = h_res + (N/2 - np.sum(h_res)) / N
            h_res = np.clip(h_res, 0, 1)
            val = _fft_max_overlap(h_res)
            if val < best_val:
                best_h = h_res.copy()
                best_val = val
    except Exception:
        pass
    
    # Phase 4: Try a few random starts with SLSQP
    np.random.seed(123)
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        
        h_init = np.random.uniform(0, 1, N)
        h_init = (h_init - h_init.mean()) + 0.5
        h_init = np.clip(h_init, 0, 1)
        h_init += (N/2 - np.sum(h_init)) / N
        h_init = np.clip(h_init, 0, 1)
        h_init += (N/2 - np.sum(h_init)) / N
        
        try:
            res = minimize(objective, h_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_res = res.x
                h_res = np.clip(h_res, 0, 1)
                h_res = h_res + (N/2 - np.sum(h_res)) / N
                h_res = np.clip(h_res, 0, 1)
                val = _fft_max_overlap(h_res)
                if val < best_val:
                    best_h = h_res.copy()
                    best_val = val
        except Exception:
            pass
    
    # Final: ensure exact sum
    best_h = best_h + (N/2 - np.sum(best_h)) / N
    best_h = np.clip(best_h, 0, 1)
    # One more tiny correction
    best_h = best_h + (N/2 - np.sum(best_h)) / N
    
    return best_h
# EVOLVE-BLOCK-END
