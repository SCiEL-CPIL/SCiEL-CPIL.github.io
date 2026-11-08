import numpy as np
import time

N = 50

def _ratio(h):
    """
    Compute the ratio (sum h^2) / (sum h * max autoconv)
    This is proportional to the continuous ratio.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    # Discrete autoconvolution max
    # c[k] = sum_i h[i] h[i-k]
    # Using correlate
    c = np.correlate(h, h, mode='full')
    ac_max = c.max()
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def construct():
    # Generate candidate patterns
    candidates = []
    
    # 1. Single Spike (Theoretical Max)
    spike = np.zeros(N)
    spike[0] = 1.0
    candidates.append(spike)
    
    # 2. Uniform (Seed)
    uniform = np.ones(N)
    candidates.append(uniform)
    
    # 3. Two adjacent spikes
    two_spikes = np.zeros(N)
    two_spikes[0] = 1.0
    two_spikes[1] = 1.0
    candidates.append(two_spikes)
    
    # 4. Alternating spikes
    alt = np.zeros(N)
    alt[::2] = 1.0
    candidates.append(alt)
    
    # 5. Gaussian-like
    x = np.linspace(0, N-1, N)
    gauss = np.exp(-0.5 * ((x - N//2) / 5)**2)
    candidates.append(gauss)
    
    # Evaluate all candidates
    best_h = None
    best_score = -1.0
    
    for h in candidates:
        score = _ratio(h)
        if score > best_score:
            best_score = score
            best_h = h
            
    # Ensure nonnegative
    if best_h is not None:
        best_h = np.maximum(best_h, 0.0)
    else:
        best_h = np.ones(N)
        
    return list(best_h)
