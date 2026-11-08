import numpy as np
import time
from scipy.optimize import minimize

N = 600
DELTA = 1.0 / 1200.0

def objective(h):
    # h is a vector of length N
    # Compute discrete convolution
    conv = np.convolve(h, h, mode='full')
    ss = np.sum(h**2)
    if ss < 1e-10:
        return 1e10
    return np.max(conv) / ss

def refine(h, time_limit=2.0):
    """
    Use L-BFGS-B with a very tight time limit to polish the initial guess.
    """
    start_time = time.time()
    try:
        # L-BFGS-B is scale invariant for the ratio objective, but we can help it
        # by fixing the norm to 1? No, let it handle bounds.
        # However, L-BFGS-B can be slow. We use a strict time check.
        # We wrap the objective to check time.
        
        def obj_with_time(x):
            if time.time() > time_limit:
                raise TimeoutError()
            return objective(x)
            
        res = minimize(obj_with_time, h, method='L-BFGS-B', 
                       bounds=[(0, None)]*N, 
                       options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6, 'maxfun': 200})
        if res.success or res.fun < objective(h):
            return res.x
    except Exception:
        pass
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 1.0
    best_h = np.ones(N)
    
    # 1. Baseline: Uniform
    h0 = np.ones(N)
    s0 = objective(h0)
    if s0 < best_score:
        best_score = s0
        best_h = h0.copy()
        
    # 2. Spectral/Analytical Initial Guesses
    # The optimal function for minimizing max autoconvolution / L2 norm^2 
    # is related to the prolate spheroidal wave functions or Gaussian-like shapes.
    # A Gaussian window $e^{-\alpha x^2}$ is a good approximation.
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA/2 # Center of intervals
    
    candidates = []
    
    # Try various Gaussian widths
    for alpha in [50, 100, 200, 500, 1000]:
        h_gauss = np.exp(-alpha * x**2)
        candidates.append(h_gauss)
        
    # Try a "raised cosine" or "Hann" window variant
    h_hann = 0.5 * (1 + np.cos(2 * np.pi * x)) # Note: x is in [-0.25, 0.25]
    # Standard Hann is 0.5(1-cos(2pi n/N)). Here we map to domain.
    # Let's use a window that tapers to 0 at boundaries.
    h_taper = np.cos(np.pi * x * 2) ** 2 # Cosine squared, 1 at center, 0 at +-0.25
    candidates.append(h_taper)
    
    # Try a "box" with tapered edges (e.g. 80% box, 20% taper)
    h_box_taper = np.ones(N)
    taper_len = int(0.1 * N)
    for i in range(taper_len):
        frac = i / taper_len
        h_box_taper[i] = frac
        h_box_taper[N-1-i] = frac
    candidates.append(h_box_taper)
    
    # Random smooth candidates (low frequency noise)
    np.random.seed(42)
    for _ in range(10):
        # Generate smooth random function
        freqs = np.random.uniform(0, 5, N)
        phases = np.random.uniform(0, 2*np.pi, N)
        h_rand = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * x + phases)
        # Ensure non-negative
        h_rand = np.maximum(h_rand, 0)
        candidates.append(h_rand)
        
    # 3. Evaluate and Refine
    for h_cand in candidates:
        if time.time() > deadline - 10: # Stop if close to deadline
            break
            
        # Normalize to avoid numerical issues in optimizer (though objective is scale invariant)
        # Actually, L-BFGS-B works better if magnitudes are reasonable.
        # Let's normalize L2 norm to 1.
        norm = np.linalg.norm(h_cand)
        if norm < 1e-10:
            continue
        h_norm = h_cand / norm
        
        # Quick score
        s_cand = objective(h_norm)
        if s_cand < best_score:
            best_score = s_cand
            best_h = h_norm.copy()
            
        # Refine if we have time
        if time.time() < deadline - 5:
            h_refined = refine(h_norm, time_limit=2.0)
            s_ref = objective(h_refined)
            if s_ref < best_score:
                best_score = s_ref
                best_h = h_refined.copy()
                
    # 4. Final global check with a few more random restarts if time permits
    # Since refine is expensive, we only do this if we have lots of time left.
    if time.time() < deadline - 30:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            # Random smooth perturbation of best_h
            h_pert = best_h.copy()
            noise = np.random.normal(0, 0.1, N)
            # Smooth the noise
            from scipy.ndimage import gaussian_filter1d
            noise = gaussian_filter1d(noise, sigma=5)
            h_pert = np.maximum(h_pert + noise, 0)
            norm = np.linalg.norm(h_pert)
            if norm < 1e-10: continue
            h_pert /= norm
            
            s_pert = objective(h_pert)
            if s_pert < best_score:
                best_score = s_pert
                best_h = h_pert.copy()
                
            if time.time() < deadline - 5:
                h_refined = refine(h_pert, time_limit=1.5)
                s_ref = objective(h_refined)
                if s_ref < best_score:
                    best_score = s_ref
                    best_h = h_refined.copy()

    return list(best_h)
