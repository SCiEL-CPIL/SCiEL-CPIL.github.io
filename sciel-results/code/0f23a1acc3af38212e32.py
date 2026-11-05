"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145  # safety margin before 150s kill


def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    For a step function f(x) with N equal intervals on [-1/4, 1/4],
    the autoconvolution (f*f)(t) is computed via FFT-based convolution.
    The bound is dx * max_k |conv[k]| where conv is the discrete autoconvolution.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n  # total length 0.5, n intervals
    
    # Pad h for linear autoconvolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Linear autoconvolution via FFT
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    # The continuous autoconvolution at grid points is dx * conv[k]
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)


def _logistic_sequence(n, r, x0):
    """
    Generate a signed sequence from the logistic map in a chaotic regime.
    x_{k+1} = r * x_k * (1 - x_k).
    Sign is +1 if x > 0.5, else -1.
    """
    x = x0
    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        seq[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
    return seq


def _local_search(h, time_budget=10.0):
    """
    Local search: try flipping individual bits (changing sign) to reduce the C3 bound.
    Uses first-improvement strategy with time-based stopping.
    """
    h = np.array(h, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < 5000:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        
        # Try flipping each position
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            new_bound = _compute_c3_bound(h)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound


def construct():
    start_time = time.time()
    
    # Initialize with multiple starting points
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try Thue-Morse sequence
    tm = _thue_morse(N)
    tm_bound = _compute_c3_bound(tm)
    if tm_bound < best_bound:
        best_bound = tm_bound
        best_h = tm.copy()
    
    # Try logistic map sequences with various parameters
    # Optimize r to maximize Lyapunov exponent (chaotic regime)
    # r in (3.57, 4.0) is chaotic
    np.random.seed(42)
    r_values = np.linspace(3.57, 3.999, 50)
    x0_values = np.linspace(0.01, 0.99, 20)
    
    for r in r_values:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        for x0 in x0_values:
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            h = _logistic_sequence(N, r, x0)
            bound = _compute_c3_bound(h)
            if bound < best_bound:
                best_bound = bound
                best_h = h.copy()
    
    # Try random initializations
    for trial in range(10):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        local_time = min(remaining - 10, 60)
        best_h, best_bound = _local_search(best_h, time_budget=local_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Alternate between random and logistic starts
            if trial % 2 == 0:
                h = np.random.choice([-1.0, 1.0], size=N)
            else:
                r = np.random.uniform(3.57, 3.999)
                x0 = np.random.uniform(0.01, 0.99)
                h = _logistic_sequence(N, r, x0)
            
            bound = _compute_c3_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 20)
                h, bound = _local_search(h, time_budget=local_time)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final verification
    final_bound = _compute_c3_bound(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
