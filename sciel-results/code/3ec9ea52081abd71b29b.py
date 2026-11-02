"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, leave margin
    
    best_h = np.ones(N)
    best_ratio = _ratio(best_h)
    
    # Try various structured configurations
    candidates = []
    
    # Single spike
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # Two spikes at various distances
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            candidates.append(h)
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Ramp
    candidates.append(np.linspace(1, 0, N))
    candidates.append(np.linspace(0, 1, N))
    
    # Triangle
    h = np.zeros(N)
    mid = N // 2
    h[mid] = 1.0
    for k in range(1, mid+1):
        if mid-k >= 0:
            h[mid-k] = 1.0 - k/mid
        if mid+k < N:
            h[mid+k] = 1.0 - k/mid
    candidates.append(h)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    for sigma in [0.2, 0.3, 0.5, 0.7, 1.0]:
        h = np.exp(-x**2 / (2*sigma**2))
        h = np.clip(h, 0, None)
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        r = _ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Local optimization using gradient-free method
    # Use Nelder-Mead style search on the best candidate
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Simple random perturbation search
        np.random.seed(42)
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            # Perturb best_h
            h_trial = best_h + np.random.randn(N) * 0.1 * np.max(best_h)
            h_trial = np.clip(h_trial, 0, None)
            # Normalize to avoid overflow/underflow
            if np.sum(h_trial) > 0:
                h_trial = h_trial / np.sum(h_trial) * np.sum(best_h)
            r = _ratio(h_trial)
            if r > best_ratio:
                best_ratio = r
                best_h = h_trial.copy()
        
        # Coordinate ascent
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                # Try adjusting coordinate i
                for delta in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 2.0]:
                    h_trial = best_h.copy()
                    h_trial[i] = max(0, h_trial[i] + delta)
                    r = _ratio(h_trial)
                    if r > best_ratio:
                        best_ratio = r
                        best_h = h_trial.copy()
                        improved = True
                        break
            if not improved:
                break
    
    # Return as list
    return list(best_h)
# EVOLVE-BLOCK-END
