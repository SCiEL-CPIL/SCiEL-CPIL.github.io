"""
Optimized program for the Third Autocorrelation Inequality task.

Structural change: Use Zadoff-Chu/CAZAC sequences as the base representation.
CAZAC sequences have near-zero non-zero autocorrelation lags, which should
minimize the maximum of the autoconvolution. We optimize sign-flip patterns
and DC offset to further reduce the peak.

Optimization strategy: 
1. Generate CAZAC base sequences for various roots
2. Optimize sign-flip patterns via local search
3. Fine-tune DC offset and amplitude scaling
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START

def _compute_value(h):
    """Compute the evaluator's exact value."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _zadoff_chu_base(root, N):
    """
    Generate a Zadoff-Chu sequence of length N.
    ZC sequences have constant amplitude and zero autocorrelation at non-zero lags
    (for prime N). For composite N, we use the closest prime or a modified version.
    """
    # Find the smallest prime >= N
    def next_prime(n):
        if n <= 2:
            return 2
        candidate = n if n % 2 == 1 else n + 1
        while True:
            if all(candidate % i != 0 for i in range(2, int(np.sqrt(candidate)) + 1)):
                return candidate
            candidate += 2
    
    p = next_prime(N)
    i = np.arange(p, dtype=np.float64)
    # Zadoff-Chu: exp(-j * pi * root * i * (i+1) / p)
    phase = -np.pi * root * i * (i + 1) / p
    zc = np.exp(1j * phase)
    
    # Take real part and normalize to unit amplitude
    h = np.real(zc)
    h = h / np.max(np.abs(h)) if np.max(np.abs(h)) > 0 else h
    
    # If p > N, truncate; if p < N, pad (shouldn't happen since p >= N)
    if len(h) > N:
        h = h[:N]
    elif len(h) < N:
        h = np.pad(h, (0, N - len(h)), mode='wrap')
    
    return h

def _cazac_with_signs(base, signs):
    """
    Apply sign flips to the CAZAC base sequence.
    signs is a binary array of length N.
    """
    return base * (2.0 * signs - 1.0)  # Map {0,1} to {-1,1}

def _soft_max(conv, temperature=50.0):
    """
    Smooth approximation to max using log-sum-exp.
    """
    m = np.max(conv)
    return m + np.log(np.sum(np.exp((conv - m) * temperature))) / temperature

def _obj_sign_flip(signs_binary, base, dc_offset, temperature=50.0):
    """
    Objective function for sign-flip optimization.
    signs_binary is a vector of 0s and 1s.
    """
    h = _cazac_with_signs(base, signs_binary) + dc_offset
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    sm = _soft_max(conv, temperature)
    return abs(2.0 * N * sm / (s * s))

def _local_search_signs(base, dc_offset, deadline, max_iterations=1000):
    """
    Local search over sign-flip patterns.
    Start from random signs and try flipping individual bits.
    """
    if time.time() > deadline:
        return np.zeros(N, dtype=int), _compute_value(base + dc_offset)
    
    best_signs = np.zeros(N, dtype=int)
    best_val = _obj_sign_flip(best_signs, base, dc_offset)
    
    # Try random initializations
    for init in range(5):
        if time.time() > deadline - 10:
            break
        signs = np.random.randint(0, 2, N)
        val = _obj_sign_flip(signs, base, dc_offset)
        if val < best_val:
            best_val = val
            best_signs = signs.copy()
    
    # Local search: try flipping each bit
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try flipping random subset of bits
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Pick a random position to flip
            idx = np.random.randint(N)
            new_signs = best_signs.copy()
            new_signs[idx] = 1 - new_signs[idx]
            
            new_val = _obj_sign_flip(new_signs, base, dc_offset)
            if new_val < best_val - 1e-10:
                best_val = new_val
                best_signs = new_signs
                improved = True
        
        if not improved:
            # Try simultaneous flips
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Flip 2-3 random positions
                n_flips = np.random.randint(2, 4)
                indices = np.random.choice(N, size=n_flips, replace=False)
                new_signs = best_signs.copy()
                new_signs[indices] = 1 - new_signs[indices]
                
                new_val = _obj_sign_flip(new_signs, base, dc_offset)
                if new_val < best_val - 1e-10:
                    best_val = new_val
                    best_signs = new_signs
                    improved = True
        
        if not improved:
            break
    
    return best_signs, best_val

def _optimize_dc_offset(base, signs, deadline):
    """
    Optimize DC offset for a given sign pattern.
    """
    if time.time() > deadline:
        return 0.0
    
    def obj_dc(dc):
        h = _cazac_with_signs(base, signs) + dc
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        sm = _soft_max(conv, temperature=30.0)
        return abs(2.0 * N * sm / (s * s))
    
    try:
        result = minimize(obj_dc, x0=0.0, method='Nelder-Mead', 
                         options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
        return result.x[0]
    except Exception:
        return 0.0

def construct():
    """
    Returns the optimal step heights.
    Uses Zadoff-Chu/CAZAC sequences with optimized sign-flip patterns.
    """
    deadline = START_TIME + TIME_LIMIT - 15.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Try different Zadoff-Chu roots
    roots = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    for root in roots:
        if time.time() > deadline - 20:
            break
        
        base = _zadoff_chu_base(root, N)
        
        # Optimize sign flips
        signs, val_signs = _local_search_signs(base, 0.0, deadline - 10)
        
        if val_signs < best_val:
            # Optimize DC offset
            dc_opt = _optimize_dc_offset(base, signs, deadline - 5)
            h_candidate = _cazac_with_signs(base, signs) + dc_opt
            val_candidate = _compute_value(h_candidate)
            
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate.copy()
    
    # Final safety check
    if time.time() > deadline - 5:
        return np.ones(N)
    
    final_val = _compute_value(best_h)
    if final_val < 2.0:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
