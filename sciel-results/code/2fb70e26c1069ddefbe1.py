import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _score_normalized(h):
    """Score assuming h is non-negative. Returns 0 if invalid."""
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _gaussian_profile(center, width, height=1.0):
    """Generate a Gaussian-like profile."""
    x = np.arange(N)
    h = height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    return h

def _lorentzian_profile(center, width, height=1.0):
    """Generate a Lorentzian-like profile."""
    x = np.arange(N)
    h = height / (1 + ((x - center) / width) ** 2)
    return h

def _raised_cosine_profile(center, width, height=1.0):
    """Generate a raised cosine bump."""
    x = np.arange(N)
    dist = np.abs(x - center)
    h = np.zeros(N)
    mask = dist <= width
    h[mask] = height * (1 + np.cos(np.pi * dist[mask] / width)) / 2
    return h

def _power_law_profile(center, width, exponent, height=1.0):
    """Generate a power-law decay profile."""
    x = np.arange(N)
    dist = np.abs(x - center)
    h = np.zeros(N)
    mask = dist <= width
    h[mask] = height * (1 - dist[mask] / width) ** exponent
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Try structured parametric families
    # The objective is scale-invariant, so we can fix the scale later or just maximize the ratio directly.
    # Let's evaluate the raw score for various shapes.
    
    profiles = []
    
    # Gaussian
    for center in np.arange(0, N, 0.5):
        for width in np.arange(0.5, N, 0.5):
            profiles.append(_gaussian_profile(center, width))
            
    # Lorentzian
    for center in np.arange(0, N, 0.5):
        for width in np.arange(0.5, N, 0.5):
            profiles.append(_lorentzian_profile(center, width))
            
    # Raised Cosine
    for center in np.arange(0, N, 0.5):
        for width in np.arange(0.5, N, 0.5):
            profiles.append(_raised_cosine_profile(center, width))
            
    # Power Law
    for center in np.arange(0, N, 0.5):
        for width in np.arange(0.5, N, 0.5):
            for exp in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
                profiles.append(_power_law_profile(center, width, exp))
                
    # Evaluate all
    for p in profiles:
        if time.time() > deadline - 20:
            break
        s = _score_normalized(p)
        if s > best_score:
            best_score = s
            best_h = p.copy()
            
    # 2. Refine the best found using local optimization in the shape space
    # We can't easily do gradient descent on the discrete grid, but we can do a fine grid search
    # around the best parameters if we stored them. 
    # Alternatively, use a general local search on the vector h itself but guided by the structure.
    
    # Let's do a simple coordinate descent on the vector h to polish the best candidate.
    # Since the function is non-smooth (due to max in inf norm), we use a stochastic approach.
    
    current_h = best_h.copy()
    current_score = best_score
    
    # Coordinate descent with random restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            h = best_h.copy()
        else:
            # Random restart with sparse structure
            h = np.zeros(N)
            k = np.random.randint(1, 10)
            idx = np.random.choice(N, k, replace=False)
            h[idx] = np.random.uniform(0.1, 1.0, k)
        
        s = _score_normalized(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        
        # Local steps
        for step in range(200):
            if time.time() > deadline - 5:
                break
                
            # Perturb a few random indices
            n_perturb = np.random.randint(1, 4)
            indices = np.random.choice(N, n_perturb, replace=False)
            h_new = h.copy()
            for idx in indices:
                # Random walk in log-space for positive values, or just multiplicative
                if h_new[idx] > 1e-6:
                    h_new[idx] *= np.random.uniform(0.5, 1.5)
                else:
                    h_new[idx] = np.random.uniform(0, 0.5)
            
            s_new = _score_normalized(h_new)
            if s_new > s:
                h = h_new
                s = s_new
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
            else:
                # With some probability, accept worse solutions (simulated annealing style)
                # But simple hill climbing is often sufficient for this type of problem
                pass
        
        # Coordinate descent
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            idx = np.random.randint(N)
            old_val = h[idx]
            # Try a few values for this coordinate
            for factor in [0.5, 0.8, 1.2, 1.5, 2.0]:
                h[idx] = max(0, old_val * factor)
                s_try = _score_normalized(h)
                if s_try > s:
                    s = s_try
                    if s > best_score:
                        best_score = s
                        best_h = h.copy()
                    break
            else:
                h[idx] = old_val
                
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    uniform = np.ones(N)
    print("Uniform score:", _score(uniform))
    
    spike = np.zeros(N)
    spike[N//2] = 1.0
    print("Spike score:", _score(spike))
    
    result = construct()
    print("Best score:", _score(np.array(result)))
    print("Result sum:", sum(result))
