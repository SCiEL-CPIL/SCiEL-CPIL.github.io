"""
Program for the Erdos minimum-overlap task using Simulated Annealing in the spatial domain.

Strategy:
1. Represent h[i] as a vector of length N=100 with values in [0, 1].
2. The objective is to minimize the maximum overlap C_max = max_k C[k], where C[k] is the circular cross-correlation of h and 1-h.
3. Instead of just minimizing the max, we use a "soft" objective that penalizes the top 5% of overlap values heavily to smooth out the peaks.
4. We use Simulated Annealing (SA) to search the space.
   - Moves: Randomly perturb a few indices of h.
   - Constraint Handling: After each move, we project h back to the feasible region (sum=50, bounds [0,1]).
   - Acceptance: Accept worse moves with probability exp(-delta_E / T), where T is the current temperature.
5. This global search approach is structurally different from the spectral amplitude optimization which is a local gradient-based method on a different parameterization.

Why this might work better:
- The landscape of the max-overlap function is non-smooth and has many local minima.
- SA can jump over barriers to find better global minima.
- The "top-k" smoothing helps guide the search towards flattening the worst peaks, which is more effective than just looking at the single max peak which can be noisy.
"""

import time
import numpy as np

N = 100
_DEADLINE = 140.0
_t0 = time.time()

def _overlap_profile(h):
    """
    Calculate the circular cross-correlation of h and 1-h.
    C[k] = sum_i h[i] * (1 - h[i-k])
    Using FFT:
    H = FFT(h)
    OMH = FFT(1-h)
    C = IFFT(H * conj(OMH)).real
    """
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    OMH = np.fft.fft(one_minus_h)
    r = np.fft.ifft(H * np.conj(OMH)).real
    return r

def _score(h):
    """
    Calculate the maximum overlap value.
    """
    h = np.asarray(h, dtype=float)
    return float(np.max(_overlap_profile(h)))

def _fix_sum(h):
    """
    Project h to satisfy sum(h) = N/2 and 0 <= h <= 1.
    This is a simple heuristic projection.
    """
    h = np.asarray(h, dtype=float).copy()
    # Clip to bounds
    h = np.clip(h, 0.0, 1.0)
    
    # Adjust sum
    target_sum = N / 2.0
    current_sum = h.sum()
    diff = target_sum - current_sum
    
    if abs(diff) < 1e-12:
        return h
    
    # Distribute the difference uniformly, then re-clip
    # This might not preserve the sum exactly if clipping occurs,
    # so we iterate a few times
    for _ in range(10):
        step = diff / N
        h = h + step
        h = np.clip(h, 0.0, 1.0)
        new_sum = h.sum()
        diff = target_sum - new_sum
        if abs(diff) < 1e-9:
            break
    
    # If still off, adjust the most "free" elements (closest to 0.5)
    if abs(h.sum() - target_sum) > 1e-9:
        indices = np.argsort(np.abs(h - 0.5))
        # Adjust the first few indices
        for idx in indices[:10]:
            if h.sum() < target_sum:
                h[idx] = min(1.0, h[idx] + (target_sum - h.sum()))
            else:
                h[idx] = max(0.0, h[idx] - (h.sum() - target_sum))
            if abs(h.sum() - target_sum) < 1e-9:
                break
    
    return h

def _objective(h):
    """
    Objective function for SA.
    We want to minimize the "peakiness" of the overlap profile.
    Using the sum of the top K values as a proxy for max is smoother and guides the search better.
    """
    C = _overlap_profile(h)
    # Sort C in descending order
    C_sorted = np.sort(C)[::-1]
    # Take the top 5% (5 values for N=100)
    top_k = 5
    return np.sum(C_sorted[:top_k])

def construct():
    global _t0
    rng = np.random.default_rng(42)
    
    # Initialize with a random valid configuration
    h = rng.uniform(0.0, 1.0, size=N)
    h = _fix_sum(h)
    
    best_h = h.copy()
    best_obj = _objective(h)
    best_score = _score(h)
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.001
    alpha = 0.995 # Cooling factor
    max_iter = 200000
    n_perturb = 5 # Number of indices to perturb per step
    
    T = T_start
    current_obj = best_obj
    
    while (time.time() - _t0) < _DEADLINE and T > T_end:
        # Perturb h
        h_new = h.copy()
        indices = rng.choice(N, size=n_perturb, replace=False)
        noise = rng.normal(0, 0.1, size=n_perturb)
        h_new[indices] += noise
        
        # Project to feasible region
        h_new = _fix_sum(h_new)
        
        new_obj = _objective(h_new)
        delta_obj = new_obj - current_obj
        
        # Acceptance criterion
        if delta_obj < 0 or rng.random() < np.exp(-delta_obj / T):
            h = h_new
            current_obj = new_obj
            
            # Update best
            if new_obj < best_obj:
                best_obj = new_obj
                best_h = h.copy()
                # Update best_score only when we find a new best in objective
                # This saves some computation
                best_score = _score(h)
        
        T *= alpha
        
        # Periodic restart if stuck? No, SA should handle it.
        # But we can add a small random restart probability if no improvement for a long time.
        # For simplicity, we stick to standard SA.
        
    # Final polish: Local search on the best solution found
    h = best_h.copy()
    eta = 0.01
    no_improve = 0
    while (time.time() - _t0) < _DEADLINE - 5 and no_improve < 50:
        C = _overlap_profile(h)
        # Find the index with the maximum overlap
        k_max = int(np.argmax(C))
        
        # Gradient of C[k_max] w.r.t h
        # C[k] = sum_i h[i] (1-h[i-k])
        # d C[k] / d h[j] = (1-h[(j-k)%N]) - h[(j+k)%N]
        shifted_minus = np.roll(h, k_max)
        shifted_plus = np.roll(h, -k_max)
        g = (1.0 - shifted_minus) - shifted_plus
        
        h_new = h - eta * g
        h_new = _fix_sum(h_new)
        
        sc_new = _score(h_new)
        
        if sc_new < best_score - 1e-10:
            best_score = sc_new
            best_h = h_new.copy()
            h = h_new.copy()
            no_improve = 0
        else:
            no_improve += 1
            eta *= 0.9
            # Perturb to escape local minimum
            h = best_h + rng.normal(0, 0.01, size=N)
            h = _fix_sum(h)
            
    return best_h
