"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _overlap_max(h, n):
    """Compute max cyclic overlap sum_i h[i]*(1-h[i-k]) over all shifts k."""
    # Overlap at shift k: sum_i h[i] * (1 - h[(i-k) % n])
    # = sum_i h[i] - sum_i h[i]*h[(i-k)%n]
    # The first term is sum(h) = n/2 (fixed)
    # So we want to maximize min_k sum_i h[i]*h[(i-k)%n]
    # i.e., maximize the minimum autocorrelation
    # overlap_k = sum(h) - autocorr_k, so max overlap = sum(h) - min autocorr
    # We want to minimize max overlap = maximize min autocorr
    autocorr = np.fft.ifft(np.abs(np.fft.fft(h))**2).real
    # autocorr[k] = sum_i h[i]*h[(i-k)%n]
    min_ac = np.min(autocorr)
    return np.sum(h) - min_ac

def _smooth_max_overlap(h, n, beta):
    """Smooth approximation of max overlap using log-sum-exp on overlaps."""
    # overlaps_k = sum(h) - autocorr_k
    autocorr = np.fft.ifft(np.abs(np.fft.fft(h))**2).real
    overlaps = np.sum(h) - autocorr
    # log-sum-exp for smooth max
    max_o = np.max(overlaps)
    if beta < 1e6:
        lse = max_o + np.log(np.sum(np.exp(beta * (overlaps - max_o)))) / beta
    else:
        lse = max_o
    return lse

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    n = N
    # Initialize with alternating pattern perturbed
    h = np.zeros(n)
    for i in range(n):
        h[i] = 0.5 + 0.49 * (1 if i % 2 == 0 else -1)
    h = np.clip(h, 0.0, 1.0)
    
    best_h = h.copy()
    best_score = _overlap_max(h, n)
    
    # Try multiple initializations
    inits = [
        'alternating',
        'random',
        'gaussian_mix',
        'fourier_modes'
    ]
    
    rng = np.random.default_rng(42)
    
    init_list = []
    # Alternating
    h_alt = np.zeros(n)
    for i in range(n):
        h_alt[i] = 0.5 + 0.49 * (1 if i % 2 == 0 else -1)
    init_list.append(np.clip(h_alt, 0, 1))
    
    # Random
    h_rand = rng.uniform(0, 1, n)
    h_rand = (h_rand - h_rand.mean()) / (h_rand.std() + 1e-10) * 0.3 + 0.5
    init_list.append(np.clip(h_rand, 0, 1))
    
    # Gaussian mix
    h_gauss = 0.5 + 0.3 * np.sin(2*np.pi*2*np.arange(n)/n) + 0.2 * np.sin(2*np.pi*3*np.arange(n)/n)
    init_list.append(np.clip(h_gauss, 0, 1))
    
    # Fourier modes - try to maximize min autocorr
    h_four = np.zeros(n)
    for i in range(n):
        h_four[i] = 0.5 + 0.35 * np.cos(2*np.pi*i/n) + 0.25 * np.cos(2*np.pi*3*i/n)
    init_list.append(np.clip(h_four, 0, 1))
    
    for h0 in init_list:
        if time.time() > deadline - 10:
            break
        
        # Project to satisfy sum constraint
        h0 = h0 - (h0.mean() - 0.5)
        h0 = np.clip(h0, 0, 1)
        # Re-project
        h0 = h0 - (h0.mean() - 0.5)
        h0 = np.clip(h0, 0, 1)
        
        # Optimization using smooth max
        def obj(x):
            if time.time() > deadline - 5:
                return 1e10
            h_x = x - (x.mean() - 0.5)
            h_x = np.clip(h_x, 0, 1)
            return _smooth_max_overlap(h_x, n, 50.0)
        
        def jac(x):
            h_x = x - (x.mean() - 0.5)
            h_x = np.clip(h_x, 0, 1)
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros(n)
            for i in range(n):
                xp = h_x.copy()
                xp[i] += eps
                xm = h_x.copy()
                xm[i] -= eps
                grad[i] = (obj(xp) - obj(xm)) / (2*eps)
            return grad
        
        try:
            res = minimize(obj, h0, method='L-BFGS-B', jac=jac, 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            h_opt = res.x - (res.x.mean() - 0.5)
            h_opt = np.clip(h_opt, 0, 1)
            # Final projection
            h_opt = h_opt - (h_opt.mean() - 0.5)
            h_opt = np.clip(h_opt, 0, 1)
            
            score = _overlap_max(h_opt, n)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Polish: local search around best
    if time.time() < deadline - 15:
        # Try small perturbations
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            h_try = best_h + rng.normal(0, 0.05, n)
            h_try = h_try - (h_try.mean() - 0.5)
            h_try = np.clip(h_try, 0, 1)
            h_try = h_try - (h_try.mean() - 0.5)
            h_try = np.clip(h_try, 0, 1)
            score = _overlap_max(h_try, n)
            if score < best_score:
                best_score = score
                best_h = h_try.copy()
        
        # Gradient descent polish
        h_cur = best_h.copy()
        lr = 0.01
        for it in range(1000):
            if time.time() > deadline - 5:
                break
            # Compute gradient numerically
            eps = 1e-6
            grad = np.zeros(n)
            for i in range(n):
                hp = h_cur.copy()
                hp[i] += eps
                hp = hp - (hp.mean() - 0.5)
                hp = np.clip(hp, 0, 1)
                hm = h_cur.copy()
                hm[i] -= eps
                hm = hm - (hm.mean() - 0.5)
                hm = np.clip(hm, 0, 1)
                grad[i] = (_overlap_max(hp, n) - _overlap_max(hm, n)) / (2*eps)
            
            h_new = h_cur - lr * grad
            h_new = h_new - (h_new.mean() - 0.5)
            h_new = np.clip(h_new, 0, 1)
            h_new = h_new - (h_new.mean() - 0.5)
            h_new = np.clip(h_new, 0, 1)
            
            if _overlap_max(h_new, n) < best_score:
                best_score = _overlap_max(h_new, n)
                best_h = h_new
                h_cur = h_new
            else:
                lr *= 0.9
    
    # Ensure exact sum
    best_h = best_h - (best_h.mean() - 0.5)
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
