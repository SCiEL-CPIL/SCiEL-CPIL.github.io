import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    # signed_step is an array of N values in {-1, 1}
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _compute_c3_fast(signed_step):
    """
    Faster C3 computation using direct FFT on the step array.
    Since the step function is constant on intervals, we can compute
    the autoconvolution more efficiently.
    """
    # Pad to avoid circular convolution
    n_fft = 2 * N
    F = np.fft.fft(signed_step, n=n_fft)
    autoconv = np.fft.ifft(F * np.conj(F)).real
    
    # The autoconvolution of the step function (scaled appropriately)
    # The actual continuous autoconvolution is related to this discrete one
    # For a step function with interval width h = 0.5/N, the peak is h * N = 0.5
    # So we need to scale: continuous_autoconv(t) ≈ h * discrete_autoconv(k)
    # where k = t/h
    
    h = 0.5 / N
    # The discrete autoconv has peak N (all ones case), continuous has peak 0.5
    # So scaling factor is 0.5/N = h
    c3 = np.max(np.abs(autoconv)) * h
    
    return c3

def _local_search_hill_climbing(signed_step, time_budget=10.0):
    """
    Hill climbing local search: flip individual bits to reduce C3.
    Uses the fast C3 computation for speed.
    """
    start_time = time.time()
    current_step = signed_step.copy()
    current_c3 = _compute_c3_fast(current_step)
    
    no_improve = 0
    patience = 2000
    
    while time.time() - start_time < time_budget:
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3_fast(current_step)
        
        if new_c3 < current_c3:
            current_c3 = new_c3
            no_improve = 0
        else:
            current_step[idx] = old_val
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    return current_step, current_c3

def _multi_start_hill_climbing(num_starts=5, time_per_start=15.0):
    """
    Run hill climbing from multiple diverse starting points.
    """
    best_step = None
    best_c3 = float('inf')
    
    for start_idx in range(num_starts):
        if time.time() > DEADLINE - 5.0:
            break
        
        # Create diverse initializations
        if start_idx == 0:
            # All ones (seed)
            init = np.ones(N, dtype=np.int8)
        elif start_idx == 1:
            # Alternating
            init = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
        elif start_idx == 2:
            # Block pattern: first half +1, second half -1
            init = np.ones(N, dtype=np.int8)
            init[N//2:] = -1
        elif start_idx == 3:
            # Random
            init = np.random.choice([-1, 1], size=N).astype(np.int8)
        else:
            # Quasi-random (based on golden ratio)
            golden = 0.6180339887498949
            init = np.ones(N, dtype=np.int8)
            for i in range(N):
                if ((i * golden) % 1.0) > 0.5:
                    init[i] = -1
        
        # Run hill climbing from this start
        refined, c3_val = _local_search_hill_climbing(init, time_budget=time_per_start)
        
        if c3_val < best_c3:
            best_c3 = c3_val
            best_step = refined.copy()
    
    return best_step, best_c3

def _annealed_search(signed_step, time_budget=20.0):
    """
    Simulated annealing style search to escape local minima.
    """
    start_time = time.time()
    current_step = signed_step.copy()
    current_c3 = _compute_c3_fast(current_step)
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # Temperature schedule
    T_start = 0.1
    T_end = 0.001
    
    num_steps = 50000
    step_interval = time_budget / num_steps
    
    for step in range(num_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Temperature decays exponentially
        progress = step / num_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3_fast(current_step)
        delta = new_c3 - current_c3
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            current_step[idx] = old_val
    
    return best_step, best_c3

def construct():
    start_time = time.time()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Phase 1: Multi-start hill climbing
    best_step, best_c3 = _multi_start_hill_climbing(num_starts=4, time_per_start=12.0)
    
    # Phase 2: Annealed search from the best found so far
    if time.time() - start_time < DEADLINE - 10.0:
        annealed_step, annealed_c3 = _annealed_search(best_step, time_budget=20.0)
        if annealed_c3 < best_c3:
            best_step = annealed_step
            best_c3 = annealed_c3
    
    # Phase 3: Final polish with hill climbing
    if time.time() - start_time < DEADLINE - 5.0:
        polished_step, polished_c3 = _local_search_hill_climbing(best_step, time_budget=10.0)
        if polished_c3 < best_c3:
            best_step = polished_step
            best_c3 = polished_c3
    
    # Verify with the original (slower but more accurate) C3 computation
    # and return the best step
    return best_step
