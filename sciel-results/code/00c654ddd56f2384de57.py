import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

def _compute_score(h):
    h = np.asarray(h, dtype=float)
    if len(h) == 0:
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _h_from_mask(mask_indices, amplitudes, phases):
    """Compute h = |IDFT(M)|^2 where M is sparse."""
    M = np.zeros(N, dtype=complex)
    for idx, amp, ph in zip(mask_indices, amplitudes, phases):
        M[int(idx) % N] = amp * np.exp(1j * ph)
    h = np.abs(np.fft.ifft(M))**2
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N) / N
    best_score = _compute_score(best_h)
    
    # Try known best
    h_known = np.zeros(N)
    h_known[0] = 2.0/3.0
    h_known[1] = 1.0/3.0
    s = _compute_score(h_known)
    if s > best_score:
        best_score = s
        best_h = h_known.copy()
    
    # Search over sparse frequency masks
    # For each number of active bins K, search over which bins and their params
    for K in range(2, 7):
        if time.time() > deadline - 30:
            break
        
        # Try different bin combinations
        # Use symmetry: only consider bins in first half to reduce redundancy
        # But let's just try random combinations and local optimize
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            # Choose K distinct frequency bins
            bins = np.random.choice(N, size=K, replace=False)
            # Initial amplitudes and phases
            amps0 = np.random.rand(K) * 0.5 + 0.5
            phs0 = np.random.rand(K) * 2 * np.pi
            
            def objective(params):
                amps = np.abs(params[:K])
                phs = params[K:2*K]
                h = _h_from_mask(bins, amps, phs)
                return -_compute_score(h)
            
            x0 = np.concatenate([amps0, phs0])
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.success or res.fun < -best_score:
                    h_new = _h_from_mask(bins, np.abs(res.x[:K]), res.x[K:2*K])
                    s_new = _compute_score(h_new)
                    if s_new > best_score:
                        best_score = s_new
                        best_h = h_new
            except:
                pass
    
    # Also try direct local optimization on the best found
    def objective_direct(h_flat):
        h = np.maximum(h_flat, 0)
        s = np.sum(h)
        if s < 1e-15:
            return 0
        h = h / s
        return -_compute_score(h)
    
    x0 = best_h.copy()
    try:
        res = minimize(objective_direct, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
        h_new = np.maximum(res.x, 0)
        s_sum = np.sum(h_new)
        if s_sum > 1e-15:
            h_new = h_new / s_sum
            s_new = _compute_score(h_new)
            if s_new > best_score:
                best_score = s_new
                best_h = h_new
    except:
        pass
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_score(h))
