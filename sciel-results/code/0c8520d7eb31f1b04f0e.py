"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _value(h):
    """Compute the task value: abs(2*N*max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))

def _soft_max(x, tau=1.0):
    """Soft maximum via log-sum-exp."""
    m = np.max(x)
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))

def _objective(params, tau=1.0):
    """
    Objective: minimize soft-max of autocorrelation / sum(h)^2.
    params: (2*K+1,) array of Fourier coefficients.
    We construct h as a sum of cosines (real-valued) with zero mean.
    """
    K = len(params) // 2
    freqs = np.arange(1, K + 1)
    # Build h from Fourier coefficients: h = sum_k a_k * cos(2*pi*k*n/N) + b_k * sin(...)
    # For simplicity, use only cosine terms (even symmetry) to keep it simple.
    # Actually, let's use a general real Fourier series.
    n = np.arange(N)
    h = np.zeros(N)
    for k in range(1, K + 1):
        a = params[2*(k-1)]
        b = params[2*(k-1)+1]
        h += a * np.cos(2 * np.pi * k * n / N) + b * np.sin(2 * np.pi * k * n / N)
    
    # Enforce zero mean by subtracting the DC component
    h -= np.mean(h)
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10  # Penalize zero sum
    
    conv = np.convolve(h, h, mode='full')
    # Use soft max for smoothness
    sm = _soft_max(conv, tau=tau)
    
    return sm / (s ** 2)

def _refine(h, max_iters=200, tol=1e-8):
    """
    Local refinement in h-space using projected gradient or L-BFGS-B.
    We optimize: minimize max(conv(h)) subject to sum(h)=1.
    Since the objective is abs(2*N*max(conv)/sum^2), and we can scale h,
    the scale-invariant part is max(conv) / (sum(h)/N)^2 * (1/N) ... 
    Actually, let's just minimize max(conv) subject to sum(h)=1, sum(h^2)=N (normalized).
    Or simpler: minimize max(conv) / (sum(h))^2 directly.
    
    Let's use L-BFGS-B with a soft-max surrogate.
    """
    # Normalize to sum=1
    s = np.sum(h)
    if abs(s) < 1e-10:
        return h
    
    # We'll optimize in the subspace where sum(h) is fixed.
    # Use a parameterization: h = h0 + M @ z, where M has rows summing to 0.
    # Simpler: use L-BFGS-B with a penalty for sum(h) != 1.
    
    def obj(z):
        h_local = z.copy()
        # Enforce sum=1 by projecting
        h_local += (1.0 - np.sum(h_local)) / N
        if abs(np.sum(h_local)) < 1e-10:
            return 1e10
        conv = np.convolve(h_local, h_local, mode='full')
        sm = _soft_max(conv, tau=0.1)
        return sm / (np.sum(h_local) ** 2)
    
    res = minimize(obj, h.copy(), method='L-BFGS-B', options={'maxiter': max_iters, 'ftol': tol})
    if res.success:
        return res.x
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Multi-start optimization in Fourier space
    np.random.seed(42)
    
    # Try different numbers of Fourier modes
    for K in [10, 20, 50]:
        if time.time() > deadline:
            break
            
        num_starts = 5
        for start in range(num_starts):
            if time.time() > deadline:
                break
            
            # Random initialization in Fourier space
            params = np.random.randn(2 * K) * 0.1
            
            # Optimize in Fourier space
            res = minimize(_objective, params, method='L-BFGS-B', 
                          args=(1.0,), options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                # Convert back to h
                K_opt = len(res.x) // 2
                n = np.arange(N)
                h = np.zeros(N)
                for k in range(1, K_opt + 1):
                    a = res.x[2*(k-1)]
                    b = res.x[2*(k-1)+1]
                    h += a * np.cos(2 * np.pi * k * n / N) + b * np.sin(2 * np.pi * k * n / N)
                h -= np.mean(h)
                
                # Local refinement in h-space
                h_refined = _refine(h, max_iters=100)
                
                val = _value(h_refined)
                if val < best_val:
                    best_val = val
                    best_h = h_refined
    
    # Final check
    if time.time() > deadline - 5:
        return best_h
    
    # One more round of refinement with tighter tolerance
    h_final = _refine(best_h, max_iters=200, tol=1e-10)
    val_final = _value(h_final)
    if val_final < best_val:
        best_h = h_final
        best_val = val_final
    
    return best_h
# EVOLVE-BLOCK-END
