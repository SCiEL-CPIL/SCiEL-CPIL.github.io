"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autoconv_max(x):
    """Compute the maximum absolute autoconvolution value for a ±1 step function.
    
    The autoconvolution at lag k (in interval units) for a step function with
    heights x_i on N equal intervals is:
    (autoconv)(k) = sum_{i} x_i * x_{i+k} for k >= 0 (with appropriate bounds)
    
    For the continuous autoconvolution of a step function, the value at offset
    t = k * (interval_width) + fraction involves partial overlaps. The maximum
    occurs at integer lags for ±1 step functions, and equals the autocorrelation.
    
    We compute the full autocorrelation via FFT and return the max absolute value
    excluding lag 0 (or including, depending on the bound).
    """
    # Full autocorrelation via FFT
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    acf = np.fft.irfft(X * np.conj(X), n=n_fft)
    # acf[0] is the zero-lag (sum of squares = N)
    # acf[1:N] are positive lags, acf[n_fft-N+1:] are negative lags
    # For a real signal, acf is symmetric
    # The autoconvolution of the step function: the continuous autoconvolution
    # at offset t has maximum at the peaks. For ±1 steps, the discrete autocorrelation
    # gives the values at integer offsets. The continuous autoconvolution between
    # integer offsets is linear interpolation, so the max is at integer offsets.
    
    # Get all lags: 0, 1, ..., N-1 (positive) and the function is symmetric
    # The autoconvolution value at lag k is the sum x_i * x_{i+k}
    # We want the maximum of this (or max absolute, depending on the bound)
    
    # For the C3 autocorrelation inequality, the bound typically involves
    # the maximum of the autocorrelation. Let's use max of |acf| for lags 1..N-1
    # Actually, let me reconsider: the "upper bound on C3" - if C3 is a constant
    # in an inequality, minimizing the upper bound means minimizing this quantity.
    
    # The autoconvolution of f with f: (f*f)(t) = ∫f(x)f(x-t)dx
    # For step functions with ±1 values on N intervals of width w=1/200:
    # At t = k*w, (f*f)(k*w) = w * sum_{i: i, i+k in [0,N)} x_i * x_{i+k}
    # The maximum over t is at some integer k (for ±1 steps, the function is piecewise
    # linear between integer lags, with the max at the endpoints).
    
    # So our objective is max_k |w * acf[k]| for k = 1, ..., N-1
    # Since w is constant, we can just use max |acf[k]| for k >= 1
    
    # acf[0] = N (zero lag)
    # acf[1], acf[2], ..., acf[N-1] are the positive lags
    # By symmetry, negative lags are the same
    
    # Return the max absolute value for non-zero lags
    nonzero_acf = np.concatenate([acf[1:N], acf[n_fft-N+1:n_fft]])
    return np.max(np.abs(nonzero_acf))


def _greedy_coordinate_descent(x, max_iterations=5000, time_limit=100.0):
    """Greedy coordinate descent: flip one coordinate at a time if it improves the objective."""
    start_time = time.time()
    current_obj = _compute_autoconv_max(x)
    best_x = x.copy()
    best_obj = current_obj
    
    # Precompute for efficiency: when we flip x[i], the change in autocorrelation
    # at lag k is: -2*x[i]*x[i+k] for lag k, and -2*x[i]*x[i-k] for lag -k
    # So the change in acf[k] for k>0 is -2*x[i]*x[i+k] if i+k < N, else 0
    # And for the symmetric part
    
    # For efficiency, let's use the direct formula for the change
    N_local = len(x)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(N_local):
            # Try flipping x[i]
            old_val = x[i]
            x[i] = -old_val
            
            new_obj = _compute_autoconv_max(x)
            
            if new_obj < current_obj - 1e-12:
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_x = x.copy()
                improved = True
            else:
                x[i] = old_val  # revert
        
        if not improved:
            break
    
    return best_x, best_obj


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Try multiple random initializations
    best_x = None
    best_obj = np.inf
    
    rng = np.random.default_rng(42)
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > deadline - 20.0:
            break
        
        # Random ±1 initialization
        x = rng.choice([-1.0, 1.0], size=N)
        
        # Run greedy coordinate descent
        time_remaining = deadline - (time.time() - start_time) - 5.0
        if time_remaining < 1.0:
            break
        
        x_opt, obj = _greedy_coordinate_descent(x, max_iterations=3000, time_limit=time_remaining)
        
        if obj < best_obj:
            best_obj = obj
            best_x = x_opt
    
    # If we didn't find anything, use a default
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
