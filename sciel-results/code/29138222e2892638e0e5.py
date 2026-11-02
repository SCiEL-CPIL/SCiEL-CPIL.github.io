"""Optimized signed step function for autoconvolution upper bound minimization."""
import numpy as np
import time

N = 400

def _autoconv_max_abs(s):
    """Compute max absolute value of the autoconvolution of sign vector s."""
    # Full linear convolution of s with itself
    conv = np.convolve(s, s, mode='full')
    return np.max(np.abs(conv))

def _sine_init(freq=1.0, phase=0.0):
    """Initialize sign vector from a low-frequency sine wave."""
    t = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
    s = np.sign(np.sin(t + phase))
    # Ensure no zeros
    s[s == 0] = 1
    return s

def _block_flip(s, start, length, rng):
    """Flip a block of signs."""
    end = min(start + length, N)
    s[start:end] = -s[start:end]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Try multiple sine-wave initializations with different frequencies/phases
    best_s = None
    best_val = np.inf
    
    # First, try a set of analytical sine-based patterns
    for freq in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0, 64.0, 80.0, 100.0, 128.0, 160.0, 200.0]:
        for phase in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
            s = _sine_init(freq, phase)
            val = _autoconv_max_abs(s)
            if val < best_val:
                best_val = val
                best_s = s.copy()
    
    if time.time() > deadline:
        return best_s
    
    # Simulated annealing with block flips
    current_s = best_s.copy()
    current_val = best_val
    
    # Annealing schedule
    T_start = 0.1
    T_end = 1e-6
    total_iterations = 50000
    
    # Precompute for speed
    for it in range(total_iterations):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        progress = it / total_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Choose block size and position
        # Mix of small and large blocks
        if rng.random() < 0.5:
            block_len = rng.integers(4, 17)  # 4-16
        else:
            block_len = rng.integers(16, 65)  # 16-64
        
        if block_len > N:
            block_len = N
        
        start = rng.integers(0, N - block_len + 1)
        
        # Compute delta in autoconvolution max abs is expensive, 
        # so we just flip and recompute (or use incremental if possible)
        old_s = current_s[start:start+block_len].copy()
        current_s[start:start+block_len] = -current_s[start:start+block_len]
        
        new_val = _autoconv_max_abs(current_s)
        
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = current_s.copy()
        else:
            # Revert
            current_s[start:start+block_len] = old_s
    
    return best_s
