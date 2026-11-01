"""Program to find a signed step function minimizing the autoconvolution upper bound C3."""
import numpy as np
import time
from scipy.optimize import linprog
from scipy.fft import fft, ifft

N = 400
TIME_LIMIT = 110.0  # Safety margin before 120s limit

def _compute_c3(h):
    """Compute the autoconvolution upper bound C3 for a signed step function h.
    
    The autoconvolution at lag 0 is sum(h_i^2) = N for signed steps.
    The upper bound on C3 involves the maximum of the autoconvolution away from lag 0,
    or a specific norm. Based on the task context (third autocorrelation inequality),
    C3 is typically related to the L-infinity norm of the autocorrelation sidelobes
    or a specific combination. 
    
    A common formulation for "autoconvolution upper bound" in such contexts is:
    C3 = max_k |sum_i h_i h_{i+k}| for k != 0, normalized or not.
    However, the "upper bound on C3" might refer to the value used in an inequality.
    
    Let's assume the score is the maximum absolute value of the circular autocorrelation
    at non-zero lags, or perhaps the global maximum including lag 0?
    Since sum h_i^2 = N, the peak at lag 0 is N.
    If the metric is "upper bound on C3", and C3 is an inequality constant,
    often we want to minimize the ratio of sidelobes to main lobe, or just the sidelobe peak.
    
    Let's look at the seed: np.ones(N).
    Autocorrelation of ones: R[k] = N - |k| for circular? No, linear autocorrelation of a box of ones.
    If h = [1]*N, linear autocorrelation R[k] = N-|k| for |k|<N.
    Max sidelobe is N-1.
    
    If we use signed steps, we can reduce the sidelobes.
    The metric is likely the maximum absolute autocorrelation value at non-zero lags,
    or perhaps the L2 norm of sidelobes.
    Given "upper bound on C3", and "lower is better", minimizing the peak sidelobe level (PSL)
    is a standard problem (minimum PSL sequence design).
    
    Let's define the score as the maximum absolute value of the linear autocorrelation
    at non-zero lags.
    """
    # Compute linear autocorrelation
    # h is length N. Autocorrelation R[k] = sum_{i} h[i] h[i+k]
    # Using FFT for efficiency
    n_fft = 2 * N
    H = fft(h, n=n_fft)
    R = ifft(H * np.conj(H))
    R = np.real(R)
    
    # R[0] is the energy (N)
    # R[k] for k=1..N-1 are positive lags
    # R[n_fft-k] correspond to negative lags
    # We want max |R[k]| for k != 0
    sidelobes = np.concatenate([R[1:N], R[n_fft-N+1:n_fft]])
    return np.max(np.abs(sidelobes))

def _evaluate_and_refine(h, time_limit):
    """Refine the sign pattern using local search (flipping signs)."""
    start_time = time.time()
    current_h = h.copy()
    current_score = _compute_c3(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    # Local search: try flipping each sign
    # To be fast, we can update autocorrelation incrementally or just recompute if N is small enough.
    # N=400, FFT is fast.
    
    for i in range(N):
        if time.time() - start_time > time_limit:
            break
        
        # Flip sign of element i
        old_val = current_h[i]
        current_h[i] = -old_val
        
        new_score = _compute_c3(current_h)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        else:
            current_h[i] = old_val
            
    return best_h, best_score

def construct():
    start_time = time.time()
    
    # Initialize with random signs
    np.random.seed(42) # Reproducibility for debugging, but maybe random is better?
    # Let's try a few random initializations
    best_h = None
    best_score = np.inf
    
    # Number of random restarts based on time
    # Each local search pass takes roughly N * (FFT cost)
    # FFT of size 800 is very fast.
    # Let's do multiple restarts.
    
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Random initialization
        h = np.random.choice([-1, 1], size=N)
        
        # Refine
        h_refined, score = _evaluate_and_refine(h, TIME_LIMIT - (time.time() - start_time) - 5)
        
        if score < best_score:
            best_score = score
            best_h = h_refined
            
        # If we've found a very low score, we can stop early
        # What is a good target? For N=400, random PSL is ~sqrt(N) ~ 20.
        # Optimized can be lower.
        
    # If no valid result, fallback to all ones (though signed is required, task says "signed step function")
    # "Choose a signed step function" implies values should be +/- 1?
    # "sequence of signed step heights" -> heights are likely +/- 1.
    
    if best_h is None:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft

N = 400
TIME_LIMIT = 110.0

def _compute_c3(h):
    n_fft = 2 * N
    H = fft(h, n=n_fft)
    R = ifft(H * np.conj(H))
    R = np.real(R)
    sidelobes = np.concatenate([R[1:N], R[n_fft-N+1:n_fft]])
    return np.max(np.abs(sidelobes))

def _evaluate_and_refine(h, time_limit):
    start_time = time.time()
    current_h = h.copy()
    current_score = _compute_c3(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    # Iterate through flips until no improvement or time out
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            old_val = current_h[i]
            current_h[i] = -old_val
            new_score = _compute_c3(current_h)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = current_h.copy()
                improved = True
            else:
                current_h[i] = old_val
                
    return best_h, best_score

def construct():
    start_time = time.time()
    np.random.seed(0)
    
    best_h = None
    best_score = np.inf
    
    # Try a few random restarts
    for r in range(20):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        h = np.random.choice([-1, 1], size=N)
        h_refined, score = _evaluate_and_refine(h, TIME_LIMIT - (time.time() - start_time) - 10)
        
        if score < best_score:
            best_score = score
            best_h = h_refined
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
