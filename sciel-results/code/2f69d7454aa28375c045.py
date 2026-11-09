import numpy as np
import time

N = 400

def _initial_autocorr(h):
    """Compute full autocorrelation r[0..N-1] where r[k] = sum_i h[i]*h[i+k]."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    return r[:N]  # r[0..N-1]

def _peak(r):
    """Max |r[k]| for k=1..N-1."""
    return np.max(np.abs(r[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = None
    best_peak = np.inf
    
    # Phase 1: Rapid random restarts to find a good starting point
    # Use a fast incremental update for the initial evaluation? No, just FFT for simplicity in restarts
    # But we need to be fast. Let's do a few hundred quick restarts.
    np.random.seed(42)
    
    # Pre-allocate for speed
    h = np.zeros(N, dtype=np.float64)
    
    # Quick random search
    # A random sequence typically has peak ~ sqrt(N) ~ 20.
    # All ones has peak N-1 = 399.
    # So random is already a huge improvement.
    
    n_restarts = 0
    # Do enough restarts to get a decent start, but not too many to eat budget
    while time.time() < deadline - 60.0:
        n_restarts += 1
        np.random.choice([-1, 1], N, out=h)
        
        r = _initial_autocorr(h)
        current_peak = _peak(r)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_h = h.copy()
        
        if n_restarts % 100 == 0:
             # Check time occasionally
             pass
             
    if best_h is None:
        # Fallback
        return np.ones(N)
        
    # Phase 2: Intensive Local Search with Incremental Updates
    # This is the key structural change: using the O(N) delta update from the current program
    # but applying it in a robust, multi-pass manner potentially with randomization or ordering.
    
    current_h = best_h.copy()
    current_r = _initial_autocorr(current_h)
    current_peak = best_peak
    
    # We will perform passes. In each pass, we try to flip every bit.
    # If a flip improves the peak, we keep it.
    # We stop if no improvement in a full pass, or if time is up.
    
    # To escape local minima, we can shuffle the order of indices in each pass.
    # However, deterministic order (0 to N-1) is often fine for greedy descent on this metric.
    # Let's try a few passes with different orderings if time permits.
    
    passes = 0
    max_passes = 50 # Safety cap, but time is the real limit
    
    while time.time() < deadline - 2.0 and passes < max_passes:
        improved = False
        # Shuffle indices to potentially escape symmetries or stuck states?
        # Actually, for peak sidelobe, local minima are common.
        # Let's stick to 0..N-1 for now, it's simpler and often effective for greedy.
        # If we want to be more aggressive, we can randomize.
        # Let's try randomizing the order every pass to avoid bias.
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline - 2.0:
                break
                
            old_val = current_h[i]
            new_val = -old_val
            
            # Calculate delta r
            # r[k] = sum_{j=0}^{N-1-k} h[j] h[j+k]
            # Change h[i] from old to new. Delta = (new - old) = -2*old
            # Terms in r[k] involving h[i]:
            # 1. j = i: h[i] h[i+k]. Valid if i+k < N. Contribution: h[i+k] * Delta
            # 2. j = i-k: h[i-k] h[i]. Valid if i-k >= 0. Contribution: h[i-k] * Delta
            # Note: For k=0, both terms are h[i]h[i], so Delta * 2*h[i]?
            # Wait, r[0] = sum h[j]^2 = N. It's constant for +/-1. We don't care about r[0] for peak sidelobe (k>=1).
            # So we only care about k=1..N-1.
            
            # Let's compute delta for k=1..N-1
            # delta_r = np.zeros(N) # Too slow to allocate every time? N=400, it's fine.
            # But we can do it in place or just check the max.
            
            # Optimization: Instead of constructing the full delta array and adding to r,
            # and then finding the max, we can update r in place, find max, and revert if not improved.
            # But reverting requires storing old r or recalculating delta and subtracting.
            # Since delta is sparse (only 2*N non-zero entries roughly), we can track affected indices.
            
            # Affected lags:
            # From j=i: lags k such that i+k < N => k = 1..N-1-i
            # From j=i-k: lags k such that i-k >= 0 => k = 1..i
            
            # Let's build the delta vector efficiently.
            # delta[k] = -2 * old_val * ( (h[i+k] if i+k<N else 0) + (h[i-k] if i-k>=0 else 0) )
            
            # We will update current_r in place.
            # Save old values of current_r at affected indices to revert if needed.
            # Or, simpler: compute new_r = current_r + delta, check peak, if better, keep, else revert.
            # Reverting: current_r -= delta.
            
            # Let's try to do this without allocating a new array for delta if possible,
            # or just allocate a small one.
            
            # Indices affected:
            # Left part: k in 1..i (from h[i-k]*h[i])
            # Right part: k in 1..N-1-i (from h[i]*h[i+k])
            
            # If i=0: only right part k=1..N-1
            # If i=N-1: only left part k=1..N-1
            
            # Let's just create a zero array and fill it. N=400 is small.
            delta = np.zeros(N)
            
            if i > 0:
                # k goes from 1 to i
                # h[i-k] for k=1..i is h[i-1], h[i-2], ..., h[0]
                # So delta[k] += -2 * old_val * h[i-k]
                # Vectorized:
                # k_indices = 1..i
                # vals = h[i-1::-1] (length i)
                delta[1:i+1] += -2 * old_val * h[i-1::-1]
                
            if i < N - 1:
                # k goes from 1 to N-1-i
                # h[i+k] for k=1..N-1-i is h[i+1], ..., h[N-1]
                # delta[k] += -2 * old_val * h[i+k]
                # Vectorized:
                # k_indices = 1..N-1-i
                # vals = h[i+1:N] (length N-1-i)
                delta[1:N-i] += -2 * old_val * h[i+1:N]
                
            # Apply delta
            new_r = current_r + delta
            new_peak = _peak(new_r)
            
            if new_peak < current_peak - 1e-12:
                # Accept
                current_r = new_r
                current_h[i] = new_val
                current_peak = new_peak
                improved = True
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = current_h.copy()
            else:
                # Reject, h[i] stays old_val (we didn't change it yet in current_h)
                # Wait, we didn't change current_h[i] yet.
                # We only changed current_r virtually.
                # So we just discard new_r.
                pass
                
        passes += 1
        
        if not improved:
            # No improvement in this pass. 
            # We could try a different strategy now, e.g., random restarts from current state?
            # Or just stop if we've done enough passes.
            # Given the time budget, stopping is usually safe.
            # But maybe we can do one more pass with shuffled indices?
            # If we didn't improve in a full pass, we are at a local minimum for single flips.
            # Breaking is standard.
            break
            
    return best_h
