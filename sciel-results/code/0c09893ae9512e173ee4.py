"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr(x):
    """Compute full autocorrelation of x (circular, via FFT)."""
    n = len(x)
    X = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(X * np.conj(X)).real[:n]
    return acf

def _peak_sidelobe(x):
    """Return the maximum absolute autocorrelation value excluding lag 0."""
    acf = _autocorr(x)
    # Exclude lag 0 (which is n)
    return np.max(np.abs(acf[1:]))

def _greedy_polish(x, max_iters=50000, deadline=None):
    """Greedy flip: flip the bit that most reduces peak sidelobe."""
    x = x.copy()
    n = len(x)
    best_peak = _peak_sidelobe(x)
    
    # Precompute: for each position, what's the effect of flipping?
    # Flipping x[i] changes acf by -2*x[i]*x shifted
    # acf_new[k] = acf[k] - 2*x[i]*sum_j x[j]*x[j+k] ... this is complex
    # Simpler: just try each flip and compute new peak
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if best_peak < 1:
            break
            
        best_gain = 0
        best_idx = -1
        
        # Sample a subset of positions for speed, or do all if fast enough
        # For N=400, computing acf takes ~O(N log N), so 400 flips * O(N log N) per iter
        # That's ~400 * 400*log(400) ≈ 400*3200 = 1.28M ops per iter - might be slow
        # Let's use a smarter approach: compute the delta acf analytically
        
        # Flipping x[i] from s to -s changes acf[k] by:
        # delta_acf[k] = -2*s * (sum_j x[j] * x[j+k] where j involves i)
        # Actually: acf[k] = sum_j x[j]*x[j+k]
        # After flipping x[i]: x'[i] = -x[i]
        # acf'[k] = acf[k] - 2*x[i]*x[i+k] - 2*x[i]*x[i-k] + ... 
        # Wait, let me think more carefully.
        
        # acf[k] = sum_{j=0}^{n-1-k} x[j]*x[j+k] (linear autocorrelation)
        # But we're using circular. Let me use circular autocorrelation.
        # acf_circ[k] = sum_j x[j]*x[(j+k) mod n]
        # Flipping x[i]: x'[i] = -x[i]
        # acf'_circ[k] = acf_circ[k] - 2*x[i]*x[(i+k) mod n] - 2*x[i]*x[(i-k) mod n] + 4*x[i]*x[i] * (if k=0)
        # Wait no. Let me redo:
        # acf_circ[k] = sum_j x[j]*x[(j+k)%n]
        # The terms involving x[i] are: x[i]*x[(i+k)%n] (when j=i) and x[(i-k)%n]*x[i] (when j+k=i, i.e., j=i-k)
        # So acf_circ[k] has two terms with x[i]: x[i]*x[(i+k)%n] and x[(i-k)%n]*x[i]
        # After flipping: these become -x[i]*x[(i+k)%n] and -x[(i-k)%n]*x[i]
        # So delta = -2*x[i]*x[(i+k)%n] - 2*x[(i-k)%n]*x[i] = -2*x[i]*(x[(i+k)%n] + x[(i-k)%n])
        
        # So delta_acf[k] = -2*x[i]*(x[(i+k)%n] + x[(i-k)%n])
        
        # We want to minimize max_k |acf[k] + delta_acf[k]| for k != 0
        
        # This is still expensive to evaluate for all k. Let's just do a few random tries.
        
        # Try a random subset of positions
        candidates = np.random.choice(n, size=min(50, n), replace=False)
        for i in candidates:
            # Compute new peak after flipping position i
            # Use the delta formula
            s = x[i]
            # delta_acf[k] = -2*s*(x[(i+k)%n] + x[(i-k)%n]) for k=1..n-1
            # We need acf + delta, then find max abs
            # This requires computing for all k, which is O(n)
            
            # Quick estimate: just compute the new acf via FFT (fast)
            x[i] = -s
            new_peak = _peak_sidelobe(x)
            x[i] = s
            
            if new_peak < best_peak - 1e-10:
                best_gain = best_peak - new_peak
                best_idx = i
        
        if best_idx >= 0:
            x[best_idx] = -x[best_idx]
            best_peak = _peak_sidelobe(x)
        else:
            # No improvement found in this round, try random flips
            for _ in range(5):
                i = np.random.randint(n)
                x[i] = -x[i]
                new_peak = _peak_sidelobe(x)
                if new_peak < best_peak:
                    best_peak = new_peak
                    break
                else:
                    x[i] = -x[i]  # revert
    
    return x, best_peak

def _anneal(x, deadline=None, max_iters=200000):
    """Simulated annealing to minimize peak sidelobe."""
    x = x.copy()
    n = len(x)
    best_x = x.copy()
    best_peak = _peak_sidelobe(x)
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        i = np.random.randint(n)
        x[i] = -x[i]
        new_peak = _peak_sidelobe(x)
        delta = new_peak - best_peak  # Wait, we should compare to current, not best
        
        # Actually, let's track current peak separately
        # Let me restructure
        
        x[i] = -x[i]  # revert for now, we'll decide
        
        # Compute current peak
        curr_peak = _peak_sidelobe(x)
        
        x[i] = -x[i]
        new_peak = _peak_sidelobe(x)
        delta = new_peak - curr_peak
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, T_min)):
            curr_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_x = x.copy()
        else:
            x[i] = -x[i]  # revert
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_x, best_peak

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    best_x = None
    best_peak = np.inf
    
    # Try multiple random initializations
    for trial in range(10):
        if time.time() > deadline:
            break
        
        # Random initialization
        x = np.random.choice([-1, 1], size=N)
        
        # Greedy polish
        x_polished, peak = _greedy_polish(x, max_iters=20000, deadline=deadline)
        
        if peak < best_peak:
            best_peak = peak
            best_x = x_polished.copy()
        
        if time.time() > deadline:
            break
        
        # Annealing from best so far
        x_annealed, peak_a = _anneal(best_x, deadline=deadline, max_iters=100000)
        
        if peak_a < best_peak:
            best_peak = peak_a
            best_x = x_annealed.copy()
    
    if best_x is None:
        best_x = np.random.choice([-1, 1], size=N)
    
    return best_x
# EVOLVE-BLOCK-END
