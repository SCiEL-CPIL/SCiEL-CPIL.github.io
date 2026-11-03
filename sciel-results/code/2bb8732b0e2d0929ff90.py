import time
import numpy as np

N = 600
_START_TIME = time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s kill

def _compute_score(h):
    """Compute normalized max of autoconvolution. Lower is better."""
    h = np.asarray(h, dtype=np.float64)
    total = h.sum()
    if total == 0:
        return float('inf')
    n = len(h)
    # Autocorrelation via FFT
    H = np.fft.fft(h, n=2*n)
    power = np.abs(H)**2
    c = np.fft.ifft(power).real
    peak = c.max()
    return peak / (total**2)

def _local_search(h, max_iters=30, rng=None):
    """Greedy local search with binary flips."""
    if rng is None:
        rng = np.random.default_rng()
    current_score = _compute_score(h)
    for _ in range(max_iters):
        if time.time() - _START_TIME > _DEADLINE:
            break
        best_gain = 0
        best_i = -1
        # Try a random subset of positions
        candidates = rng.choice(N, size=min(80, N), replace=False)
        for i in candidates:
            old_val = h[i]
            h[i] = 1.0 - old_val
            new_score = _compute_score(h)
            gain = current_score - new_score
            if gain > best_gain:
                best_gain = gain
                best_i = i
            h[i] = old_val
        if best_i >= 0:
            h[best_i] = 1.0 - h[best_i]
            current_score -= best_gain
        else:
            break
    return h, current_score

# EVOLVE-BLOCK-START
def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    # Initialize: 50% on, evenly spaced
    h = np.zeros(N)
    h[::2] = 1.0
    
    # Local search to get good starting point
    h, current_score = _local_search(h, max_iters=20, rng=rng)
    
    # Simulated annealing with binary flips
    T = 0.05
    T_min = 0.0005
    cooling = 0.997
    best_h = h.copy()
    best_score = current_score
    no_improve = 0
    
    while T > T_min and time.time() - _START_TIME < _DEADLINE:
        # Propose flip at random position
        i = rng.integers(0, N)
        old_val = h[i]
        h[i] = 1.0 - old_val
        new_score = _compute_score(h)
        
        delta = new_score - current_score
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            h[i] = old_val
            no_improve += 1
        
        T *= cooling
        
        # Periodic local search to escape plateaus
        if no_improve > 200 and time.time() - _START_TIME < _DEADLINE - 5:
            h, current_score = _local_search(h, max_iters=15, rng=rng)
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            no_improve = 0
    
    # Final polish
    if time.time() - _START_TIME < _DEADLINE - 2:
        h_final = best_h.copy()
        h_final, _ = _local_search(h_final, max_iters=50, rng=rng)
        final_score = _compute_score(h_final)
        if final_score < best_score:
            best_h = h_final
            best_score = final_score
    
    return list(best_h)
# EVOLVE-BLOCK-END
