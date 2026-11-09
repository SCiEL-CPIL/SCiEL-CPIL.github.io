"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_all_overlaps(h):
    """Compute overlap for all shifts using FFT. Returns array of N overlaps."""
    n = len(h)
    # overlap(k) = sum_i h[i] * (1 - h[(i+k) % n]) = sum(h) - sum_i h[i]*h[(i+k)%n]
    # The second term is circular cross-correlation of h with h
    # Using FFT: correlation = ifft(fft(h) * conj(fft(h))) gives autocorrelation
    # Actually we need sum_i h[i] * h[(i+k) % n] which is circular cross-correlation
    # corr[k] = sum_i h[i] * h[(i+k) % n]
    # This is ifft(fft(h) * conj(fft(roll(h, -k))))... 
    # Simpler: corr = ifft(fft(h) * conj(fft(h))) gives sum_i h[i+k]*h[i]
    # Let me just use the direct approach but vectorized for speed
    
    # For N=100, direct computation is fast enough with numpy
    overlaps = np.zeros(n)
    for k in range(n):
        if k == 0:
            overlaps[k] = np.sum(h * (1.0 - h))
        else:
            shifted = np.roll(h, -k)
            overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _compute_max_overlap(h):
    """Compute the maximum overlap."""
    n = len(h)
    max_ov = 0.0
    for k in range(1, n):
        shifted = np.roll(h, -k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _compute_max_overlap_fast(h):
    """Faster computation using precomputed structure."""
    n = len(h)
    # overlap(k) = sum(h) - circular_corr_h_h(k)
    # where circular_corr_h_h(k) = sum_i h[i] * h[(i+k) % n]
    sum_h = np.sum(h)
    
    # Compute circular autocorrelation using FFT
    # autocorr[k] = sum_i h[i+k] * h[i] (with circular indexing)
    # = ifft(fft(h) * conj(fft(h)))
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[(i+k) % n] * h[i]
    # So overlap(k) = sum_h - autocorr[k]
    overlaps = sum_h - autocorr
    # We want max over k=1..n-1 (shift by 0 is trivial: sum(h*(1-h)))
    if n > 1:
        return np.max(overlaps[1:n])
    return 0.0

def _fix_sum_binary(h, n_ones):
    """Ensure h has exactly n_ones ones and the rest zeros, with sum = N/2."""
    n = len(h)
    # h should be binary with exactly n_ones = n/2 ones
    # If not, adjust
    current_ones = np.sum(h > 0.5)
    if current_ones == n_ones:
        return h.copy()
    
    # Convert to binary
    h_bin = (h > 0.5).astype(float)
    current_ones = np.sum(h_bin > 0.5)
    
    if current_ones > n_ones:
        # Need to turn off some ones
        ones_idx = np.where(h_bin > 0.5)[0]
        # Turn off the "least useful" ones (arbitrary for now)
        to_turn_off = np.random.choice(ones_idx, current_ones - n_ones, replace=False)
        h_bin[to_turn_off] = 0.0
    else:
        # Need to turn on some zeros
        zeros_idx = np.where(h_bin <= 0.5)[0]
        to_turn_on = np.random.choice(zeros_idx, n_ones - current_ones, replace=False)
        h_bin[to_turn_on] = 1.0
    
    return h_bin

def _local_search_binary(h, time_limit, seed=42):
    """Local search on binary sequences using 1-flips and 2-swaps."""
    rng = np.random.RandomState(seed)
    n = len(h)
    n_ones = n // 2
    
    h = h.copy()
    current_score = _compute_max_overlap_fast(h)
    best_score = current_score
    best_h = h.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            break
        
        # Choose operation: 1-flip (turn on a 0, turn off a 1) or 2-swap
        # For binary with fixed count, we need to swap a 1 with a 0
        ones = np.where(h > 0.5)[0]
        zeros = np.where(h <= 0.5)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        # Try multiple random swaps
        improved = False
        for _ in range(20):
            i = ones[rng.randint(len(ones))]
            j = zeros[rng.randint(len(zeros))]
            
            h[i] = 0.0
            h[j] = 1.0
            
            new_score = _compute_max_overlap_fast(h)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                no_improve_count = 0
                improved = True
            else:
                # Revert
                h[i] = 1.0
                h[j] = 0.0
        
        if not improved:
            no_improve_count += 1
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n_ones = N // 2
    
    # Start with a good initial pattern
    # Try alternating pattern
    h_init = np.zeros(N)
    h_init[::2] = 1.0
    # Adjust to have exactly n_ones
    if np.sum(h_init > 0.5) != n_ones:
        h_init = np.zeros(N)
        indices = np.random.RandomState(42).permutation(N)
        h_init[indices[:n_ones]] = 1.0
    
    best_h = h_init.copy()
    best_score = _compute_max_overlap_fast(best_h)
    
    # Try several structured initial patterns
    structured_patterns = []
    
    # Block patterns
    for block_size in range(2, 20):
        if N % block_size == 0:
            n_blocks = N // block_size
            ones_per_block = block_size // 2
            if ones_per_block == 0:
                continue
            h_pat = np.zeros(N)
            for b in range(n_blocks):
                start = b * block_size
                h_pat[start:start+ones_per_block] = 1.0
            if np.sum(h_pat > 0.5) == n_ones:
                structured_patterns.append(h_pat)
    
    # Sinusoidal threshold patterns
    for m in range(1, 15):
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            h_pat = 0.5 + 0.49 * np.sin(2 * np.pi * m * np.arange(N) / N + phase)
            h_bin = (h_pat > 0.5).astype(float)
            if np.sum(h_bin > 0.5) == n_ones:
                structured_patterns.append(h_bin)
    
    # Evaluate structured patterns
    for h_pat in structured_patterns:
        score = _compute_max_overlap_fast(h_pat)
        if score < best_score:
            best_score = score
            best_h = h_pat.copy()
    
    # Run local search from multiple starting points
    np.random.seed(42)
    n_restarts = 0
    max_restarts = 30
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        # Create random initial solution
        h_rand = np.zeros(N)
        indices = np.random.permutation(N)
        h_rand[indices[:n_ones]] = 1.0
        
        # Run local search with remaining time
        remaining = deadline - time.time() - 2
        if remaining < 1.0:
            break
        
        run_time = min(remaining, 3.0)  # Cap each run at 3s
        h_res, score_res = _local_search_binary(h_rand, run_time, seed=42 + n_restarts)
        
        if score_res < best_score:
            best_score = score_res
            best_h = h_res.copy()
        
        n_restarts += 1
    
    # Final polish: try local search from the best solution found
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        if remaining > 1.0:
            h_res, score_res = _local_search_binary(best_h, min(remaining, 5.0), seed=999)
            if score_res < best_score:
                best_score = score_res
                best_h = h_res.copy()
    
    return best_h.astype(float)
# EVOLVE-BLOCK-END
