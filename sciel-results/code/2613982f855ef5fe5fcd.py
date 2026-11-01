import numpy as np
import time

N = 400
DEADLINE = 145.0  # Safety margin before 150s limit

def _compute_c3(h):
    """
    Compute the C3 upper bound: max_k |conv(h,h)[k]| / (sum_i |h_i|)^2
    Uses FFT for efficient autoconvolution.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-12:
        return float('inf')
    
    return np.max(np.abs(ac)) / (sum_abs_h ** 2)

def _fast_local_search(h_init, time_limit=10.0):
    """
    Fast discrete local search with incremental autocorrelation updates.
    """
    start_time = time.time()
    h = h_init.copy()
    
    # Initial autocorrelation
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(F * np.conj(F), n=n_fft)
    current_max = np.max(np.abs(ac))
    current_val = current_max / (N ** 2) # sum_abs_h is always N for +/-1
    
    best_h = h.copy()
    best_val = current_val
    
    # Precompute indices for fast update
    # When flipping h[i], ac[k] changes by -2*h[i]*h[k-i] (for valid k-i)
    # We can vectorize the update of ac for a given i.
    
    # To speed up, we can try flipping random indices and accept if it improves.
    # We can also try a "steepest descent" style: find the flip that gives the largest reduction.
    # For N=400, checking all 400 flips takes 400 * O(N) = 160k ops, which is fast.
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            break
            
        # Find the best flip
        best_delta = 0
        best_idx = -1
        
        # Sample a subset of indices to keep it fast, or all if fast enough
        # Let's try all indices first, it's only 400.
        # Vectorized update for all i is hard, so we loop i.
        for i in range(N):
            # Calculate change in ac if we flip h[i]
            # h[i] becomes -h[i]
            # delta_ac[k] = -2 * h[i] * (h[k-i] + h[i-k]) ? 
            # Actually, ac[k] = sum_j h[j] h[j-k]
            # If we flip h[i], the terms involving h[i] change.
            # Terms: h[i]h[i-k] and h[i+k]h[i]
            # New terms: -h[i]h[i-k] and h[i+k](-h[i])
            # Change: -2 h[i] (h[i-k] + h[i+k])
            
            # We need to update ac and check if max |ac| decreases.
            # Instead of updating full ac, we can estimate the impact on the max.
            # This is tricky. Let's just do a simple random walk with acceptance, 
            # or a greedy hill climb with random restarts within the loop.
            
            # Greedy hill climb: try flipping i, compute new max, if better, keep.
            # To make it faster, we can use a temperature or just accept any improvement.
            
            # Let's try a random subset of indices to avoid O(N^2) per step if N were larger.
            # Here N=400, so O(N^2) per "epoch" is 160,000 ops. 
            # 1000 epochs = 160M ops. NumPy can do ~100M ops/s. So 1.6s per 1000 epochs.
            # This is feasible.
            
            pass # The loop above is a placeholder. We implement the actual logic below.

        # Actual implementation: Random Walk with Greedy Acceptance
        # Pick a random index
        i = np.random.randint(N)
        old_sign = h[i]
        
        # Compute delta_ac
        # ac_new = ac - 2*old_sign*(h shifted)
        # We need to be careful with boundaries.
        # ac is length 2N-1.
        # h is length N.
        # ac[k] = sum_{j} h[j] h[j-k]
        
        # Let's just recompute the affected parts or use a fast approximation.
        # Given the speed of NumPy, let's try a "vectorized batch flip" approach? No, single flip is better for local search.
        
        # Efficient update:
        # delta_ac = np.zeros(2*N-1)
        # Indices j such that j-i is in [0, N-1] -> j in [i, i+N-1]
        # And j such that i-j is in [0, N-1] -> j in [i-(N-1), i]
        
        # This is complex to vectorize perfectly without slicing issues.
        # Alternative: Just use FFT for every 10th flip to correct drift, and incremental for others?
        # Or, since N=400 is small, maybe full FFT is fast enough for ~1000 flips?
        # FFT size 800. 1000 FFTs is ~0.1s. 
        # So we can do ~10,000 flips in 1s.
        
        # Let's stick to the full FFT recomputation but optimize the loop structure.
        # The previous program was slow because it did it in a Python loop with overhead.
        
        # Let's try a different strategy: 
        # 1. Random initialization
        # 2. Greedy local search: Iterate through all indices, flip if it improves. Repeat until no improvement.
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                # Flip h[i]
                h[i] = -h[i]
                
                # Recompute C3
                F = np.fft.rfft(h, n=n_fft)
                ac_new = np.fft.irfft(F * np.conj(F), n=n_fft)
                new_max = np.max(np.abs(ac_new))
                new_val = new_max / (N ** 2)
                
                if new_val < current_val:
                    current_val = new_val
                    ac = ac_new
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h.copy()
                else:
                    # Revert
                    h[i] = old_sign
        
        # If no improvement in a full pass, try a random perturbation
        if time.time() - start_time < time_limit:
            # Flip 5 random bits
            for _ in range(5):
                h[np.random.randint(N)] *= -1
            # Restart greedy phase
            # This logic is getting complex. Let's simplify.
            
    return best_h, best_val

def _local_search_v2(h_init, time_limit=10.0):
    """
    Optimized local search using incremental updates where possible, 
    or fast FFT recomputation.
    """
    start_time = time.time()
    h = h_init.copy()
    n_fft = 2 * N
    
    # Initial state
    F = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(F * np.conj(F), n=n_fft)
    current_max = np.max(np.abs(ac))
    current_val = current_max / (N ** 2)
    
    best_h = h.copy()
    best_val = current_val
    
    # Precompute shift indices for incremental update
    # When flipping h[i], the change in ac is:
    # delta_ac[k] = -2 * h[i] * (h[k-i] + h[i-k])
    # We can precompute the vectors h[k-i] and h[i-k] for each i? 
    # That's N vectors of length 2N. 400 * 800 = 320k floats. 1.2MB. Fine.
    
    # But we need to handle boundaries.
    # Let's just use the FFT method but optimize the Python loop overhead.
    # The bottleneck is the Python loop calling FFT 400 times per pass.
    # 400 FFTs of size 800 is fast (~1-2ms).
    # So one pass takes ~1s. 
    # We can do ~10 passes in 10s.
    
    for _ in range(20): # Max passes
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
                
            old_sign = h[i]
            h[i] = -old_sign
            
            F = np.fft.rfft(h, n=n_fft)
            ac_new = np.fft.irfft(F * np.conj(F), n=n_fft)
            new_max = np.max(np.abs(ac_new))
            new_val = new_max / (N ** 2)
            
            if new_val < current_val:
                current_val = new_val
                ac = ac_new
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i] = old_sign
                
        if not improved:
            # Try a random kick to escape local minimum
            for _ in range(10):
                h[np.random.randint(N)] *= -1
            # Recompute state after kick
            F = np.fft.rfft(h, n=n_fft)
            ac = np.fft.irfft(F * np.conj(F), n=n_fft)
            current_val = np.max(np.abs(ac)) / (N ** 2)
            
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Multiple random starts
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() >= deadline - 5.0:
            break
            
        # Random initialization
        h_init = np.random.choice([-1.0, 1.0], size=N)
        
        # Run local search with a time slice
        time_slice = min(5.0, deadline - time.time() - 5.0)
        if time_slice < 1.0:
            break
            
        h_res, val_res = _local_search_v2(h_init, time_limit=time_slice)
        
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()
            
    # Phase 2: Polish the best result with a longer run
    if best_h is not None and time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        h_polished, val_polished = _local_search_v2(best_h, time_limit=remaining)
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished.copy()
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
