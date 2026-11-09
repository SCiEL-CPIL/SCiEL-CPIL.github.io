import numpy as np
from scipy.optimize import minimize
import time

def _score_from_heights(h):
    """
    Computes the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    We assume circular convolution for the autoconvolution, as this allows
    the peak to be smaller than ||h||_2^2 for asymmetric h, potentially
    leading to scores > 1.0.
    """
    h = np.asarray(h, dtype=float)
    N = len(h)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    
    l2_sq = np.sum(h ** 2)
    
    # Compute circular autoconvolution using FFT for speed
    # h*h_circ = IFFT(FFT(h) * FFT(h))
    H = np.fft.fft(h)
    ac = np.fft.ifft(H * H).real
    
    # Due to floating point errors, real part might have tiny negatives, 
    # but for non-negative h, the true value is non-negative.
    peak = np.max(ac)
    
    if peak <= 0:
        return 0.0
        
    return l2_sq / (l1 * peak)

def _neg_ratio(h):
    """Objective function for minimization: negative of the score."""
    return -_score_from_heights(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    N = 50
    best_score = -np.inf
    best_h = None
    
    # We optimize the shape.
    # Parameterization: h_i = exp(y_i) / sum(exp(y_j))
    # This ensures h_i >= 0 and sum(h_i) = 1.
    
    def objective(y):
        h = np.exp(y)
        h = h / np.sum(h)
        return -_score_from_heights(h)
    
    max_restarts = 50
    
    # Diverse initializations
    # 1. Random
    # 2. Spikes (one high, others low)
    # 3. Asymmetric blocks
    # 4. Comb patterns
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Generate y0
        if i % 5 == 0:
            # Random normal
            y0 = np.random.normal(0, 1, N)
        elif i % 5 == 1:
            # Spike: one large value
            y0 = np.zeros(N)
            y0[np.random.randint(N)] = 5.0
        elif i % 5 == 2:
            # Asymmetric block: high on left, low on right
            y0 = np.zeros(N)
            y0[:N//2] = 1.0
            y0[N//2:] = -1.0
        elif i % 5 == 3:
            # Comb: alternating high/low
            y0 = np.zeros(N)
            y0[::2] = 1.0
            y0[1::2] = -1.0
        else:
            # Smooth random
            y0 = np.cumsum(np.random.normal(0, 0.5, N))
            y0 = y0 - np.mean(y0)
            
        try:
            # Use Nelder-Mead for robustness in non-smooth/non-convex landscapes
            # Or Powell. Nelder-Mead is often good for these "peaky" functions.
            res = minimize(objective, y0, method='Nelder-Mead',
                           options={'maxiter': 500, 'fatol': 1e-8, 'xatol': 1e-8})
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_y = res.x
                best_h = np.exp(best_y)
                best_h = best_h / np.sum(best_h)
                
        except Exception:
            continue
        
        if time.time() > deadline:
            break

    if best_h is None:
        # Fallback: uniform
        best_h = np.ones(N) / N
    else:
        # Ensure non-negative and normalized
        best_h = np.maximum(best_h, 0)
        s = np.sum(best_h)
        if s > 0:
            best_h = best_h / s
        else:
            best_h = np.ones(N) / N
            
    return list(best_h)
