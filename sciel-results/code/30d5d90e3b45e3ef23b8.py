"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute the max absolute autocorrelation at non-zero lags."""
    n = len(h)
    # Full autocorrelation via FFT
    fft_size = 2 * n
    H = np.fft.fft(h, n=fft_size)
    acf = np.fft.ifft(np.abs(H)**2).real
    # acf[0] is the DC term (sum of squares), we want non-zero lags
    # The autocorrelation at lag k is acf[k] for k>0 and acf[fft_size-k] for k<0
    max_val = 0.0
    for k in range(1, n):
        val = abs(acf[k])
        if val > max_val:
            max_val = val
    return max_val


def _score(h):
    """Score = upper bound on C3 (lower is better)."""
    return _autocorr_peak(h)


def _soft_max_abs(x, beta=10.0):
    """
    Soft approximation of max(|x|) using log-sum-exp.
    This provides a smooth surrogate for the nonsmooth max-abs objective.
    """
    abs_x = np.abs(x)
    # Avoid log(0) issues
    max_val = np.max(abs_x)
    if max_val == 0:
        return 0.0
    # log-sum-exp for stability
    shifted = abs_x - max_val
    sum_exp = np.sum(np.exp(beta * shifted))
    return max_val + (1.0 / beta) * np.log(sum_exp)


def _soft_score(h, beta=10.0):
    """
    Soft surrogate for the max-abs autocorrelation peak.
    Uses log-sum-exp to smooth the objective for gradient-based or simulated annealing approaches.
    """
    n = len(h)
    fft_size = 2 * n
    H = np.fft.fft(h, n=fft_size)
    acf = np.fft.ifft(np.abs(H)**2).real
    # Extract non-zero lags
    lags = np.concatenate([acf[1:n], acf[fft_size-n:fft_size]])
    return _soft_max_abs(lags, beta)


def _b_spline_init(order=4, base_freq=2.0, phase=0.0):
    """
    Generate a smooth periodic waveform using a sum of B-spline-like basis functions
    (approximated by smoothed step functions or splines), then threshold the derivative
    to create a signed step function.
    
    The idea: smooth spectral roll-off suppresses high-frequency autoconvolution peaks.
    """
    x = np.linspace(0, 1, N, endpoint=False)
    
    # Construct a smooth signal from a few harmonics with B-spline-like smoothing
    # We use a sum of cosines with decreasing amplitude (spectral roll-off)
    signal = np.zeros(N)
    for k in range(1, 10):
        # B-spline-like spectral decay: 1/k^2
        amp = 1.0 / (k ** 2)
        signal += amp * np.cos(2 * np.pi * k * base_freq * x + phase * k)
    
    # Compute the derivative (finite difference) to get a "signed step" structure
    # The derivative of a smooth low-autocorrelation signal, when thresholded,
    # should yield a sequence with low autoconvolution peaks
    deriv = np.zeros(N)
    deriv[:-1] = np.diff(signal)
    deriv[-1] = deriv[0]  # periodic
    
    # Threshold the derivative to create ±1 or 0 values
    # Use a small threshold to keep most values active
    threshold = 0.05 * np.std(deriv)
    h = np.zeros(N)
    h[deriv > threshold] = 1.0
    h[deriv < -threshold] = -1.0
    
    return h


def _local_search_signs(h, deadline, max_iterations=500):
    """
    Local search by flipping individual signs.
    Uses a greedy approach with simulated annealing-like acceptance.
    """
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    active_idx = np.nonzero(current_h)[0]
    if len(active_idx) == 0:
        return best_h, best_score
    
    rng = np.random.default_rng(12345)
    temperature = 0.01
    cooling = 0.999
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random active position to flip
        idx = active_idx[rng.integers(len(active_idx))]
        old_val = current_h[idx]
        current_h[idx] = -old_val
        new_score = _score(current_h)
        
        delta = new_score - current_score
        
        # Metropolis acceptance
        if delta < 0 or (temperature > 1e-6 and rng.random() < np.exp(-delta / temperature)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        else:
            current_h[idx] = old_val
        
        temperature *= cooling
    
    return best_h, best_score


def _swap_search(h, deadline, max_swaps=200):
    """
    Local search by swapping active positions with inactive ones.
    """
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    active_idx = np.nonzero(current_h)[0]
    inactive_idx = np.setdiff1d(np.arange(N), active_idx)
    
    if len(inactive_idx) == 0 or len(active_idx) == 0:
        return best_h, best_score
    
    rng = np.random.default_rng(67890)
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        # Pick a random active to remove and inactive to add
        i_remove = rng.integers(len(active_idx))
        i_add = rng.integers(len(inactive_idx))
        
        idx_remove = active_idx[i_remove]
        idx_add = inactive_idx[i_add]
        old_val = current_h[idx_remove]
        
        # Perform swap
        current_h[idx_remove] = 0
        current_h[idx_add] = old_val
        
        new_score = _score(current_h)
        
        if new_score < current_score:
            current_score = new_score
            # Update active/inactive sets
            active_idx = np.delete(active_idx, i_remove)
            active_idx = np.append(active_idx, idx_add)
            inactive_idx = np.delete(inactive_idx, i_add)
            inactive_idx = np.append(inactive_idx, idx_remove)
            
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        else:
            # Revert
            current_h[idx_add] = 0
            current_h[idx_remove] = old_val
    
    return best_h, best_score


def construct():
    """
    Construct a signed step function with N=400 intervals
    that minimizes the autoconvolution upper bound (C3).
    
    Strategy:
    1. Initialize using B-spline-inspired smooth signal thresholding.
    2. Refine with combined sign-flip and position-swap local search.
    3. Use soft surrogate for exploration in early stages.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_h = np.ones(N)  # Fallback: the seed program
    best_score = _score(best_h)
    
    # Phase 1: B-spline-inspired initialization with different parameters
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        
        # Vary base frequency and phase
        base_freq = 1.0 + 0.5 * trial
        phase = trial * 0.7
        
        h = _b_spline_init(order=4, base_freq=base_freq, phase=phase)
        
        # Ensure we have a reasonable number of non-zero entries
        # If too sparse or too dense, adjust
        n_nonzero = np.count_nonzero(h)
        if n_nonzero < 50:
            # Add some random ±1 values
            rng = np.random.default_rng(trial)
            zero_idx = np.nonzero(h == 0)[0]
            if len(zero_idx) > 0:
                add_count = min(100, len(zero_idx))
                add_idx = rng.choice(zero_idx, size=add_count, replace=False)
                h[add_idx] = rng.choice([-1, 1], size=add_count)
        elif n_nonzero > 350:
            # Remove some values to make it sparser
            rng = np.random.default_rng(trial + 1000)
            nonzero_idx = np.nonzero(h != 0)[0]
            remove_count = n_nonzero - 250
            remove_idx = rng.choice(nonzero_idx, size=remove_count, replace=False)
            h[remove_idx] = 0
        
        current_score = _score(h)
        
        # Phase 2: Local search refinement
        h, current_score = _local_search_signs(h, deadline - 15)
        h, current_score = _swap_search(h, deadline - 15, max_swaps=100)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
    
    # Phase 3: Final polish with more iterations on the best solution
    if time.time() < deadline - 10:
        best_h, best_score = _local_search_signs(best_h, deadline - 5, max_iterations=1000)
        best_h, best_score = _swap_search(best_h, deadline - 5, max_swaps=300)
    
    return best_h
# EVOLVE-BLOCK-END
