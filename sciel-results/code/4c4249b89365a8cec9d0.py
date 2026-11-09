"""Spectral optimization for Erdos minimum overlap."""
import time
import numpy as np

N = 100
DEADLINE = 120.0
SAFETY_MARGIN = 10.0  # seconds before hard limit


def _compute_overlap_max(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts."""
    # h is a sequence of step heights
    # 1-h is the complement
    # Overlap at shift k: sum_i h[i] * (1-h)[(i-k) % N]
    # = sum_i h[i] * (1 - h[(i-k) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i-k) % N]
    # = sum(h) - autocorrelation of h at shift k
    
    h_sum = np.sum(h)
    # Compute autocorrelation for all shifts
    # For shift k, autocorr[k] = sum_i h[i] * h[(i-k) % N]
    # This is the circular cross-correlation of h with itself
    # Using FFT for efficiency
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[i] * h[(i+k) % N] (need to check convention)
    # Actually, ifft(|H|^2)[k] = sum_j h[j] * h[(j+k) mod N]
    # We want sum_i h[i] * h[(i-k) mod N] which is autocorr at shift -k
    # But since we take max over all shifts, it's the same set of values
    
    overlap_max = h_sum - np.min(autocorr)
    return overlap_max


def _evaluate(h):
    """Evaluate the objective: maximum overlap of h with shifted (1-h)."""
    if not np.all((h >= 0) & (h <= 1)):
        return float('inf')
    if abs(np.sum(h) - N / 2) > 1e-6:
        return float('inf')
    return _compute_overlap_max(h)


def _construct_sinusoidal(amplitude, phase, baseline=0.5):
    """Construct h as baseline + amplitude * sin(2*pi*phase*i/N)."""
    i = np.arange(N)
    h = baseline + amplitude * np.sin(2 * np.pi * phase * i / N)
    # Clip to [0, 1]
    h = np.clip(h, 0, 1)
    # Adjust to make sum exactly N/2
    # If after clipping the sum is not N/2, we need to adjust
    current_sum = np.sum(h)
    if current_sum > 0:
        # Scale or add constant to fix sum
        # Simple approach: add a constant offset
        offset = (N / 2 - current_sum) / N
        h = h + offset
        h = np.clip(h, 0, 1)
        # May need iterative adjustment
        for _ in range(10):
            current_sum = np.sum(h)
            diff = N / 2 - current_sum
            if abs(diff) < 1e-10:
                break
            offset = diff / N
            h = h + offset
            h = np.clip(h, 0, 1)
    return h


def _construct_high_freq(amplitude, freq, phase=0.0, baseline=0.5):
    """Construct h with a high-frequency sinusoidal component."""
    i = np.arange(N)
    h = baseline + amplitude * np.sin(2 * np.pi * freq * i / N + phase)
    h = np.clip(h, 0, 1)
    current_sum = np.sum(h)
    if current_sum > 0:
        offset = (N / 2 - current_sum) / N
        h = h + offset
        h = np.clip(h, 0, 1)
        for _ in range(10):
            current_sum = np.sum(h)
            diff = N / 2 - current_sum
            if abs(diff) < 1e-10:
                break
            offset = diff / N
            h = h + offset
            h = np.clip(h, 0, 1)
    return h


def _local_search(h, max_iters=200, step=0.1):
    """Simple local search to refine the solution."""
    best_h = h.copy()
    best_score = _evaluate(best_h)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        improved = False
        for i in range(N):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                    break
            for delta in [-step, step]:
                new_h = best_h.copy()
                new_h[i] = np.clip(new_h[i] + delta, 0, 1)
                # Fix sum
                diff = N / 2 - np.sum(new_h)
                if abs(diff) > 1e-10:
                    # Distribute the correction
                    mask = (new_h > 0) & (new_h < 1)
                    if np.sum(mask) > 0:
                        new_h[mask] += diff / np.sum(mask)
                        new_h = np.clip(new_h, 0, 1)
                
                score = _evaluate(new_h)
                if score < best_score - 1e-10:
                    best_h = new_h
                    best_score = score
                    improved = True
        
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    
    return best_h, best_score


def construct():
    start_time = time.time()
    
    # Strategy: Use spectral analysis to find good initial candidates
    # Then refine with local search
    
    best_h = np.full(N, 0.5)
    best_score = _evaluate(best_h)  # Should be N/2 = 50
    
    # Try various sinusoidal constructions with different frequencies
    # Key insight: high-frequency perturbations reduce cross-correlation
    # for non-zero shifts
    
    # First, try the single-period sinusoid that gave score 1.085
    # h = 0.5 + A*sin(2*pi*i/N)
    # The overlap max is approximately N/4 + A^2/2
    # To minimize, we want small A, but we need the sum to be N/2
    # With a pure sinusoid, sum is already N/2 (integral of sin over full period is 0)
    
    # Try different amplitudes for the fundamental frequency
    for amp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        h = _construct_sinusoidal(amp, 0.0)
        score = _evaluate(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try higher frequencies - these may have lower max overlap
    # For frequency f, the autocorrelation structure changes
    for freq in [1, 2, 3, 4, 5, 7, 10, 13, 17, 23]:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        for amp in [0.3, 0.5, 0.7]:
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            h = _construct_high_freq(amp, freq)
            score = _evaluate(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Try binary-like patterns (0 and 1) with specific structures
    # A pattern that is "well-mixed" should have low autocorrelation
    # Try alternating blocks
    
    # Try a few more structured patterns
    patterns = []
    # Block pattern: groups of 0s and 1s
    for block_size in [1, 2, 3, 4, 5, 7, 10]:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        # Create a pattern with blocks
        h = np.zeros(N)
        # Alternate blocks of 1s and 0s, adjusting to get sum = N/2
        pos = 0
        val = 1
        while pos < N:
            end = min(pos + block_size, N)
            h[pos:end] = val
            pos = end
            val = 1 - val
        # Fix sum
        diff = N / 2 - np.sum(h)
        if abs(diff) > 1e-10:
            # Need to adjust some elements
            # Change some 1s to 0s or vice versa
            if diff > 0:
                # Need more 1s
                idx = np.where(h == 0)[0]
                if len(idx) > 0:
                    h[idx[:int(diff)]] = 1
            else:
                idx = np.where(h == 1)[0]
                if len(idx) > 0:
                    h[idx[:int(-diff)]] = 0
        
        if abs(np.sum(h) - N / 2) < 1e-6:
            patterns.append(h)
    
    for h in patterns:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        score = _evaluate(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Local search refinement
    if time.time() < DEADLINE - SAFETY_MARGIN:
        best_h, best_score = _local_search(best_h, max_iters=100, step=0.2)
    
    # Final check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Return the best we have
        pass
    
    # Ensure valid output
    best_h = np.clip(best_h, 0, 1)
    # Final sum fix
    diff = N / 2 - np.sum(best_h)
    if abs(diff) > 1e-10:
        mask = (best_h > 0) & (best_h < 1)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
