"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

def _score(h):
    """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf)"""
    h = np.asarray(h, dtype=float)
    sum_sq = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    # Circular autocorrelation: (f*f)(k) proportional to sum_j h_j * h_{j+k}
    n = len(h)
    max_corr = 0.0
    for k in range(n):
        corr = np.dot(h, np.roll(h, k))
        if corr > max_corr:
            max_corr = corr
    return sum_sq / (sum_h * max_corr)

def _optimize():
    """Optimize the nonnegative height vector to maximize the ratio."""
    deadline = time.time() + 100  # Safety margin
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try various parametric families and local optimization
    np.random.seed(42)
    
    # Strategy 1: Low-frequency cosine patterns (clipped to nonnegative)
    for freq in range(1, 10):
        for phase in np.linspace(0, 2*np.pi, 20, endpoint=False):
            t = np.arange(N)
            h = np.maximum(0.1, np.cos(2*np.pi*freq*t/N + phase))
            # Normalize
            h = h / np.sum(h) * N
            s = _score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
    
    # Strategy 2: Random initialization + gradient-free local search
    for restart in range(50):
        if time.time() > deadline:
            break
        # Random smooth initialization
        t = np.arange(N)
        if restart % 3 == 0:
            # Low freq sinusoid
            f = np.random.uniform(1, 5)
            p = np.random.uniform(0, 2*np.pi)
            h = np.maximum(0.05, np.cos(2*np.pi*f*t/N + p))
        elif restart % 3 == 1:
            # Random smooth
            coeffs = np.random.randn(5) * 0.3
            h = np.maximum(0.05, 1.0 + sum(c*np.cos(2*np.pi*k*t/N) for k, c in enumerate(coeffs)))
        else:
            # Perturbed uniform
            h = np.ones(N) + 0.3*np.random.randn(N)
            h = np.maximum(0.01, h)
        
        h = h / np.sum(h) * N
        
        # Local search with random perturbations
        for step in range(200):
            if time.time() > deadline:
                break
            s = _score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
            
            # Random perturbation
            delta = np.random.randn(N) * 0.05
            h_new = np.maximum(0.01, h + delta)
            h_new = h_new / np.sum(h_new) * N
            h = h_new
    
    # Strategy 3: Try specific structured patterns
    patterns = [
        # Ramp
        np.linspace(0.1, 2.0, N),
        # Triangle
        np.abs(np.linspace(-1, 1, N)),
        # Two peaks
        np.maximum(0.1, np.exp(-((np.arange(N) - N//4)**2)/5) + np.exp(-((np.arange(N) - 3*N//4)**2)/5)),
        # Single peak (Gaussian)
        np.maximum(0.1, np.exp(-((np.arange(N) - N//2)**2)/10)),
        # Cosine window
        np.maximum(0.1, np.cos(np.pi * (np.arange(N) - N/2) / N)),
    ]
    for pat in patterns:
        h = pat / np.sum(pat) * N
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Final polish: coordinate descent
    if time.time() < deadline - 5:
        h = best_h.copy()
        for _ in range(10000):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try a few values
                for candidate in [old*0.9, old*0.8, old*1.1, old*1.2, 0.01, 0.1, 1.0, 2.0]:
                    h[i] = max(0.01, candidate)
                    s = _score(h)
                    if s > best_score:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
                        break
                h[i] = old
            if not improved:
                break
    
    return list(best_h)

# EVOLVE-BLOCK-START
def construct():
    return _optimize()
# EVOLVE-BLOCK-END
