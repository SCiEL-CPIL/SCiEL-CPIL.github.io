"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # safety margin before 120s kill

# EVOLVE-BLOCK-START
def _overlap_values(h, shifts):
    """Return overlap for each shift: sum_i h_i * (1 - h_{(i+shift) mod N})."""
    # h shape (N,), shifts shape (S,)
    # For each shift s, overlap = sum_i h[i] * (1 - h[(i+s) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i+s)%N]
    # Use FFT for cross-correlation: sum_i h[i]*h[(i+s)%N] is circular cross-corr
    H = np.fft.fft(h)
    # circular cross correlation c[s] = sum_i h[i] * h[(i+s) % N]
    # = sum_i h[i] * conj(h[i]) shifted... use ifft(fft(h) * conj(fft(h)))? 
    # Actually autocorrelation: r[s] = sum_i h[i+s] * h[i] (circular)
    # We want sum_i h[i] * h[(i+s)%N]. Let j = (i+s)%N => i = (j-s)%N
    # = sum_j h[(j-s)%N] * h[j] = autocorrelation at lag s.
    r = np.real(np.fft.ifft(np.abs(H)**2))  # r[s] = sum_j h[j]*h[(j+s)%N] circular
    total = np.sum(h)
    return total - r  # overlap[s] = sum h[i] - sum h[i]*h[(i+s)%N]

def _max_overlap_smooth(h, beta):
    """Smooth max of overlap via log-sum-exp."""
    ov = _overlap_values(h, np.arange(N))
    m = np.max(ov)
    return m + np.log(np.sum(np.exp(beta*(ov - m)))) / beta

def _objective(x, beta):
    h = np.clip(x, 0.0, 1.0)
    # enforce sum constraint softly or via projection; we'll use equality in minimize
    return _max_overlap_smooth(h, beta)

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    # Start from a smooth ramp: 0 to 1 over the domain, then adjust to mean 0.5
    # A good initial guess: h[i] = 0.5 + 0.5*sin(2*pi*i/N) gives mean 0.5, in [0,1]
    i = np.arange(N)
    h0 = 0.5 + 0.5*np.sin(2*np.pi*i/N)
    h0 = np.clip(h0, 0.0, 1.0)
    # adjust to exact sum N/2
    h0 = h0 - np.mean(h0) + 0.5
    h0 = np.clip(h0, 0.0, 1.0)
    # final correction for exact sum
    h0 = h0 + (N/2.0 - np.sum(h0))/N
    h0 = np.clip(h0, 0.0, 1.0)
    
    best_h = h0.copy()
    best_score = _max_overlap_smooth(best_h, 50.0)
    
    # Try multiple betas and restarts within time budget
    for beta in [10.0, 30.0, 50.0, 100.0]:
        if time.time() - t0 > _DEADLINE - 20:
            break
        # L-BFGS-B with equality constraint sum(h)=N/2
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        try:
            res = minimize(_objective, h0.copy(), args=(beta,), method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-9})
            if res.success or res.fun < best_score:
                h_new = np.clip(res.x, 0.0, 1.0)
                # exact sum correction
                h_new = h_new + (N/2.0 - np.sum(h_new))/N
                h_new = np.clip(h_new, 0.0, 1.0)
                s_new = _max_overlap_smooth(h_new, 50.0)
                if s_new < best_score:
                    best_h = h_new.copy()
                    best_score = s_new
                    h0 = h_new.copy()
        except Exception:
            pass
    
    # Final high-precision polish with high beta
    if time.time() - t0 < _DEADLINE - 10:
        try:
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            res = minimize(_objective, best_h.copy(), args=(100.0,), method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 500, 'ftol': 1e-12})
            h_new = np.clip(res.x, 0.0, 1.0)
            h_new = h_new + (N/2.0 - np.sum(h_new))/N
            h_new = np.clip(h_new, 0.0, 1.0)
            s_new = _max_overlap_smooth(h_new, 100.0)
            if s_new < best_score:
                best_h = h_new.copy()
        except Exception:
            pass
    
    # Ensure exact sum and valid bounds
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = best_h + (N/2.0 - np.sum(best_h))/N
    best_h = np.clip(best_h, 0.0, 1.0)
    # final tiny correction to make sum exactly N/2 within float precision
    diff = N/2.0 - np.sum(best_h)
    best_h[0] += diff
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
