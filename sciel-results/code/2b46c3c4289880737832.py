# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _compute_conv(h):
    """Compute full convolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _objective_value(h):
    """
    Compute the raw upper bound value: max_k |conv(h,h)[k]| / (sum |h_i|)^2
    """
    conv = _compute_conv(h)
    max_abs = np.max(np.abs(conv))
    sum_abs = np.sum(np.abs(h))
    return max_abs / (sum_abs * sum_abs)

def _local_search_signs(h, max_iters=5000, time_limit=10.0):
    """
    Perform local search on the signs of h to minimize the max autoconvolution peak.
    We assume |h_i| is fixed at 1 for this search phase.
    """
    start_time = time.time()
    h = h.copy()
    # Ensure h is strictly +/- 1 for the discrete search
    h = np.sign(h)
    # In case any are 0, set to 1
    h[h == 0] = 1.0
    
    conv = _compute_conv(h)
    current_max = np.max(np.abs(conv))
    
    # Precompute the indices for efficient update?
    # conv[k] = sum_i h[i] h[i-(k-(N-1))]
    # If we flip h[j], conv[k] changes by -2*h[j]*h[j-(k-(N-1))]
    # Let's just recompute the affected parts or use the full update for simplicity given N=400.
    # Full recompute of conv is O(N^2) = 160,000 ops. 5000 iters = 800M ops. Might be too slow in pure Python loops if not vectorized.
    # np.convolve is C-optimized. 5000 calls to np.convolve of size 400 should be fast (< 1s).
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random index to flip
        j = np.random.randint(N)
        
        # Calculate change
        old_h_j = h[j]
        new_h_j = -old_h_j
        
        # Efficient update:
        # conv_old = h * h
        # conv_new = (h - 2*old_h_j*e_j) * (h - 2*old_h_j*e_j)
        # conv_new = h*h - 4*old_h_j*(h*e_j + e_j*h) + 4*old_h_j^2 * e_j*e_j
        # Since e_j is a unit vector at index j:
        # (h * e_j)[k] = h[k-j] (with shifts)
        # It's easier to just recompute conv if N is small enough.
        
        h[j] = new_h_j
        new_conv = _compute_conv(h)
        new_max = np.max(np.abs(new_conv))
        
        # Accept if better (or equal with some probability for exploration, but strictly better is safer for convergence)
        if new_max < current_max:
            current_max = new_max
            conv = new_conv
        else:
            # Revert
            h[j] = old_h_j
    
    return h, current_max

def construct():
    """
    Constructs a signed step function that minimizes the C3 upper bound.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    best_h = None
    best_val = float('inf')
    
    # Multiple random restarts
    n_restarts = 20
    
    rng = np.random.default_rng(42)
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 15.0: # Stop if not enough time for one more restart
            break
            
        # Initialize with random signs
        h_init = rng.choice([-1.0, 1.0], size=N)
        
        # Run local search
        # Limit time per restart to ensure we finish in time
        time_per_restart = (time_limit - (time.time() - start_time)) / (n_restarts - r)
        if time_per_restart < 1.0:
             time_per_restart = 1.0
             
        h_opt, val_max = _local_search_signs(h_init, max_iters=2000, time_limit=time_per_restart)
        
        # Calculate actual score (denominator is N^2 since |h_i|=1)
        # val_max is max |conv|. Score = val_max / N^2
        score = val_max / (N * N)
        
        if score < best_val:
            best_val = score
            best_h = h_opt
            
    if best_h is None:
        # Fallback to seed (all ones) if something went wrong
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
