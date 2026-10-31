import numpy as np
import time
from scipy.signal import fftconvolve

def _autoconv_max_abs(x):
    """Compute max |autoconvolution| of x."""
    n = len(x)
    if n == 0:
        return 0.0
    # Full autoconvolution via FFT
    c = fftconvolve(x, x, mode='full')
    return np.max(np.abs(c))

def _score(x):
    """Compute the C3' score: max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return _autoconv_max_abs(x) / (s * s)

def _local_search(x, max_iters=5000, deadline=None):
    """Local search: flip individual signs to reduce score."""
    x = x.copy()
    n = len(x)
    current_score = _score(x)
    
    # Precompute autoconvolution for efficiency
    c = fftconvolve(x, x, mode='full')
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping each position
        best_gain = 0.0
        best_pos = -1
        for i in range(n):
            # Compute new autoconvolution if we flip x[i]
            # x_new = x with x[i] negated
            # delta = -2*x[i] at position i
            # New autoconv = c + delta * x + x * delta + delta * delta
            # = c - 2*x[i]*(x shifted by i) - 2*x[i]*(x shifted by -i) + 4*x[i]^2 * delta_i
            # This is complex; let's just compute directly for candidate flips
            
            # Skip if no time
            if deadline is not None and time.time() > deadline:
                break
            
            # Try flipping position i
            x[i] = -x[i]
            new_c = fftconvolve(x, x, mode='full')
            new_max = np.max(np.abs(new_c))
            new_sum = np.sum(x)
            if abs(new_sum) < 1e-12:
                new_score = np.inf
            else:
                new_score = new_max / (new_sum * new_sum)
            x[i] = -x[i]  # revert
            
            if new_score < current_score:
                gain = current_score - new_score
                if gain > best_gain:
                    best_gain = gain
                    best_pos = i
        
        if best_pos >= 0:
            x[best_pos] = -x[best_pos]
            current_score = current_score - best_gain
            improved = True
    
    return x, current_score

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    
    Uses binary sign sequences {±1} with local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Try different sequence lengths
    # For binary sequences, N around 100-500 seems reasonable for optimization
    best_x = None
    best_score = np.inf
    
    # Start with various initial conditions and lengths
    for N in [200, 300, 400, 500, 600, 800]:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        n_trials = 3
        for trial in range(n_trials):
            if time.time() > deadline - 20:
                break
            
            # Random initial sign sequence
            np.random.seed(42 + trial * 1000 + N)
            x = np.random.choice([-1.0, 1.0], size=N)
            
            # Run local search
            x_opt, score = _local_search(x, max_iters=2000, deadline=deadline - 15)
            
            if score < best_score:
                best_score = score
                best_x = x_opt.copy()
        
        if time.time() > deadline - 15:
            break
    
    # If no good result, fallback to constant sequence
    if best_x is None or best_score >= 1.0 / 10000:
        # Use a large constant sequence as fallback
        N_fallback = 100000
        best_x = np.ones(N_fallback)
    
    return best_x
