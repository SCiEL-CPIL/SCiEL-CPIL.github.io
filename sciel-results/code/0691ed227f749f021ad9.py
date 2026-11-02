"""Erdos minimum overlap - optimized via autocorrelation shaping."""
import numpy as np
import time

N = 100

def _compute_overlap(h):
    """Compute max overlap of h with shifted 1-h."""
    # overlap(s) = sum_i h[i] * (1-h)[(i-s) mod N] = N/2 - autocorr(s)
    # We want to minimize max_s overlap(s) = N/2 - min_s autocorr(s)
    # So we want to maximize min_s autocorr(s)
    h = np.asarray(h, dtype=np.float64)
    # Autocorrelation via FFT
    H = np.fft.fft(h)
    power = np.abs(H)**2
    autocorr = np.fft.ifft(power).real
    # autocorr[s] = sum_i h[i]*h[(i-s) mod N]
    # The max overlap = N/2 - min(autocorr)
    # But autocorr[0] = sum h[i]^2 which is the largest typically
    # We care about all shifts including 0? Let's assume all shifts.
    min_ac = np.min(autocorr)
    return N / 2.0 - min_ac

def _project_to_simplex_with_bounds(h, target_sum=None):
    """Project h onto [0,1]^N with sum = target_sum (default N/2)."""
    if target_sum is None:
        target_sum = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    # Adjust to meet sum constraint while staying in [0,1]
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    if diff > 0:
        # Need to increase sum; add to elements with most room
        room = 1.0 - h
        order = np.argsort(-room)
        for i in order:
            if diff <= 0:
                break
            add = min(diff, room[i])
            h[i] += add
            diff -= add
    else:
        # Need to decrease sum; subtract from elements with most value
        order = np.argsort(h)
        for i in order:
            if diff >= 0:
                break
            sub = min(-diff, h[i])
            h[i] -= sub
            diff += sub
    return h

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap(best_h)
    
    # Try several structural initializations
    candidates = []
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Periodic patterns with period p
    for p in [2, 3, 4, 5, 7, 10]:
        if N % p == 0:
            # Try various duty cycles
            for duty in [0.3, 0.4, 0.5, 0.6, 0.7]:
                pattern = np.zeros(p)
                k = int(round(duty * p))
                pattern[:k] = 1.0
                # Scale to mean 0.5
                if np.mean(pattern) > 0:
                    pattern = pattern / np.mean(pattern) * 0.5
                h = np.tile(pattern, N // p)
                candidates.append(h)
    
    # 3. Sinusoidal variations
    for freq in [1, 2, 3, 4, 5]:
        t = np.arange(N)
        for amp in [0.1, 0.2, 0.3]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * t / N)
            h = np.clip(h, 0, 1)
            # Normalize sum
            if np.sum(h) > 0:
                h = h * (N / 2.0) / np.sum(h)
                h = np.clip(h, 0, 1)
            candidates.append(h)
    
    # 4. Random with structure
    np.random.seed(42)
    for _ in range(20):
        h = np.random.rand(N) * 0.5 + 0.25
        h = _project_to_simplex_with_bounds(h)
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        h = _project_to_simplex_with_bounds(h)
        score = _compute_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Local optimization: coordinate descent / gradient steps
    h = best_h.copy()
    lr = 0.01
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Compute gradient of -min(autocorr) w.r.t. h
        # min(autocorr) is non-smooth; use subgradient or smooth approximation
        # autocorr(s) = sum_i h[i]*h[(i-s)%N]
        # d(autocorr(s))/dh[j] = h[(j+s)%N] + h[(j-s)%N]
        
        H = np.fft.fft(h)
        power = np.abs(H)**2
        autocorr = np.fft.ifft(power).real
        
        # Find which lag has minimum autocorr
        s_min = np.argmin(autocorr)
        
        # Gradient of autocorr(s_min) w.r.t. h
        # autocorr(s) = sum_i h[i] * h[(i-s)%N]
        # d/dh[j] = h[(j+s)%N] + h[(j-s)%N] (for the two terms involving h[j])
        # But autocorr is circular, so:
        grad = np.zeros(N)
        for j in range(N):
            idx1 = (j + s_min) % N
            idx2 = (j - s_min) % N
            grad[j] = h[idx1] + h[idx2]
        
        # We want to maximize autocorr(s_min), so update h += lr * grad
        # But also need to maintain sum = N/2 and bounds
        h_new = h + lr * grad
        
        # Project
        h_new = _project_to_simplex_with_bounds(h_new)
        
        score_new = _compute_overlap(h_new)
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            h = h_new.copy()
            # Decrease learning rate slightly
            lr *= 0.999
        else:
            h = h.copy()
            lr *= 0.95
            if lr < 1e-6:
                break
    
    # Final projection to ensure exact sum
    best_h = _project_to_simplex_with_bounds(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
