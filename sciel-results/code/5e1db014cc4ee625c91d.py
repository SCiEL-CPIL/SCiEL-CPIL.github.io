import numpy as np
import time

N = 600
_DEADLINE = 140.0  # safety margin before 150s kill

def compute_score(h):
    """
    Computes the normalized maximum of the autoconvolution.
    h: array of nonnegative heights.
    Score = max_k (h * h)_k / (sum h)^2
    """
    if np.sum(h) == 0:
        return float('inf')
    # Autoconvolution
    # h is length N. h*h is length 2N-1.
    # We only need the max, which is at index N-1 (shift 0) for non-negative h.
    # c_0 = sum(h^2)
    c_max = np.sum(h**2)
    norm = np.sum(h)**2
    return c_max / norm

def construct():
    start = time.time()
    
    # Candidate 1: Uniform distribution
    h_uniform = np.ones(N)
    
    # Candidate 2: Triangular distribution (peak in center)
    h_tri = np.linspace(1, 0, N//2)
    h_tri = np.concatenate([h_tri, h_tri[:-1][::-1]])
    # Ensure non-negative
    h_tri = np.maximum(h_tri, 0)
    
    # Candidate 3: Gaussian-like (discretized)
    x = np.linspace(-1, 1, N)
    h_gauss = np.exp(-x**2 * 2)
    
    # Candidate 4: "Flat top" with slight taper
    h_taper = np.ones(N)
    taper_len = N // 10
    h_taper[:taper_len] = np.linspace(0.5, 1.0, taper_len)
    h_taper[-taper_len:] = np.linspace(1.0, 0.5, taper_len)
    
    candidates = [h_uniform, h_tri, h_gauss, h_taper]
    
    best_h = h_uniform
    best_score = float('inf')
    
    for h in candidates:
        # Normalize to same sum for fair comparison? 
        # The score is scale-invariant: (c * h * c * h) / (c * sum h)^2 = score(h)
        # So normalization doesn't change the score, but helps avoid overflow/underflow.
        h_norm = h / np.sum(h) * N # Scale so sum is N, like uniform
        score = compute_score(h_norm)
        
        if score < best_score:
            best_score = score
            best_h = h_norm
            
    # Check time
    if time.time() - start > _DEADLINE:
        return [float(x) for x in h_uniform]
        
    return [float(x) for x in best_h]
