import numpy as np
import time

N = 400
DEADLINE = 115.0  # Safety margin before 120s

def _compute_conv(h):
    """Compute linear autoconvolution of h using FFT."""
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    return C[:2*N-1]

def _compute_score(h, conv=None):
    """Compute max|conv| / sum(h^2)."""
    if conv is None:
        conv = _compute_conv(h)
    peak = np.max(np.abs(conv))
    energy = np.sum(h**2)
    if energy < 1e-8:
        return np.inf
    return peak / energy

def _flip_delta(h, conv, i):
    """
    Compute the change in convolution when flipping h[i].
    Returns delta array of length 2N-1.
    """
    # C[k] = sum_j h[j]*h[j+k]
    # Flipping h[i] -> -h[i]:
    # Terms affected: j=i gives h[i]*h[i+k], and j+k=i gives h[j]*h[i] = h[i-k]*h[i]
    # So delta[k] = -2*h[i]*(h[i+k] + h[i-k]) for valid indices
    delta = np.zeros(2*N-1)
    hi = h[i]
    # For lag k (0-indexed in conv array, conv[k] corresponds to lag k-(N-1) in centered sense)
    # Actually conv[k] = sum_j h[j]*h[j+k] where k ranges 0 to 2N-2
    # conv[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k] for k >= 0 (positive lag)
    # For the full linear conv, conv[k] for k in [0, 2N-2]:
    #   if k < N: conv[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
    #   if k >= N: conv[k] = sum_{j=k-N+1}^{N-1} h[j]*h[j-(k-N+1)] ... 
    # Let me use the standard: conv[k] = sum_{j} h[j]*h[j+k] with h out of range = 0
    # So for each k in [0, 2N-2], conv[k] = sum_{j=max(0,k-(N-1))}^{min(N-1,k)} h[j]*h[j+k-N+1]... 
    # This is getting confusing. Let me just use the direct formula.
    # 
    # The linear convolution (h*h)[k] for k=0..2N-2 is:
    #   (h*h)[k] = sum_{j=0}^{N-1} h[j] * h[k-j] where h[m]=0 if m not in [0,N-1]
    # So (h*h)[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j] * h[k-j]
    #
    # When we flip h[i], the terms where j=i: h[i]*h[k-i] (if k-i in [0,N-1])
    # and where k-j=i, i.e., j=k-i: h[k-i]*h[i] (if k-i in [0,N-1])
    # If i == k-i (i.e., k=2i), these are the same term, counted once.
    #
    # So delta[k] = -2*h[i]*h[k-i] if k-i in [0,N-1] and k != 2i
    #              = -2*h[i]*h[i] if k == 2i  (but h[i]*h[i] = 1 for binary, so -2)
    # Actually for binary, h[i]^2 = 1, so:
    #   if k-i in [0,N-1]:
    #     if k == 2i: delta[k] = -2*1 = -2
    #     else: delta[k] = -2*h[i]*h[k-i]
    #
    # But wait, h[i] is ±1, so h[i]*h[k-i] is ±1.
    
    # Let's compute efficiently
    # k ranges 0 to 2N-2
    # k-i must be in [0, N-1] for the term to exist
    k_min = i  # k-i >= 0 => k >= i
    k_max = i + N - 1  # k-i <= N-1 => k <= i+N-1
    k_min = max(0, k_min)
    k_max = min(2*N-2, k_max)
    
    if k_min <= k_max:
        ks = np.arange(k_min, k_max+1)
        vals = h[ks - i]
        delta[k_min:k_max+1] = -2.0 * hi * vals
        # Fix the double-count at k=2i
        if 2*i >= k_min and 2*i <= k_max:
            delta[2*i] = -2.0 * hi * hi  # = -2.0 since hi^2=1
    
    return delta

def _local_search_steepest(h, time_limit, max_iter=None):
    """
    Steepest-descent local search: at each step, find the best single flip.
    """
    h = h.copy()
    conv = _compute_conv(h)
    best_score = _compute_score(h, conv)
    
    start_time = time.time()
    no_improve = 0
    patience = 50
    
    while time.time() - start_time < time_limit:
        if time.time() > DEADLINE - 3.0:
            break
        
        # Evaluate all possible flips
        best_delta = None
        best_idx = -1
        best_new_score = best_score
        
        # For each possible flip, compute the new peak
        # We need to be efficient: compute delta for each i, then new_conv = conv + delta,
        # then new_peak = max|new_conv|
        # This is O(N) per flip, O(N^2) total per iteration
        
        # Optimization: precompute h once
        for i in range(N):
            delta = _flip_delta(h, conv, i)
            new_conv = conv + delta
            new_peak = np.max(np.abs(new_conv))
            new_score = new_peak / N  # energy = N for binary
            if new_score < best_new_score:
                best_new_score = new_score
                best_delta = delta
                best_idx = i
        
        if best_idx == -1 or best_new_score >= best_score:
            no_improve += 1
            if no_improve >= patience:
                break
        else:
            # Apply the best flip
            h[best_idx] = -h[best_idx]
            conv = conv + best_delta
            best_score = best_new_score
            no_improve = 0
    
    return h, best_score

def _local_search_random_walk(h, time_limit):
    """
    Random walk local search with occasional restarts.
    """
    h = h.copy()
    conv = _compute_conv(h)
    best_h = h.copy()
    best_score = _compute_score(h, conv)
    
    start_time = time.time()
    no_improve = 0
    patience = 2000
    
    while time.time() - start_time < time_limit:
        if time.time() > DEADLINE - 3.0:
            break
        
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -h[idx]
        
        delta = _flip_delta(h, conv, idx)  # This computes delta for the NEW h, but we need delta for the flip from old to new
        # Actually, _flip_delta computes the change when flipping h[idx] in the current h.
        # Since we already flipped, this gives the change to go back. We need the forward delta.
        # Let me recompute: the delta for flipping idx in the original h is what we need.
        # But we've already flipped. So let's undo, compute delta, then apply.
        
        # Undo
        h[idx] = old_val
        delta = _flip_delta(h, conv, idx)
        
        new_conv = conv + delta
        new_peak = np.max(np.abs(new_conv))
        new_score = new_peak / N
        
        if new_score < best_score:
            # Apply flip
            h[idx] = -h[idx]
            conv = new_conv
            best_score = new_score
            best_h = h.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            # Perturb best_h
            if time.time() > DEADLINE - 10.0:
                break
            h = best_h.copy()
            conv = _compute_conv(h)
            n_perturb = max(1, int(0.05 * N))
            perturb_idx = np.random.choice(N, size=n_perturb, replace=False)
            for pi in perturb_idx:
                h[pi] = -h[pi]
            conv = _compute_conv(h)
            no_improve = 0
            patience = 1000
    
    return best_h, best_score

def _generate_initializations():
    """Generate diverse initial binary sequences."""
    inits = []
    
    # All ones
    inits.append(np.ones(N, dtype=np.int8))
    
    # Alternating
    alt = np.ones(N, dtype=np.int8)
    alt[1::2] = -1
    inits.append(alt)
    
    # Block patterns
    for block_size in [10, 20, 40, 50, 80, 100]:
        h = np.ones(N, dtype=np.int8)
        for start in range(0, N, 2*block_size):
            h[start+block_size:start+2*block_size] = -1
        inits.append(h)
    
    # Random
    for _ in range(5):
        h = np.random.choice([-1, 1], size=N).astype(np.int8)
        inits.append(h)
    
    # Half positive, half negative (random split)
    for _ in range(3):
        h = np.random.choice([-1, 1], size=N).astype(np.int8)
        inits.append(h)
    
    return inits

def construct():
    start_time = time.time()
    
    best_h = None
    best_score = np.inf
    
    # Generate initializations
    inits = _generate_initializations()
    
    # For each initialization, run local search with time budget
    n_inits = len(inits)
    time_per_init = min(10.0, (DEADLINE - start_time - 5.0) / max(1, n_inits))
    
    for i, h_init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10.0:
            break
        
        remaining = DEADLINE - elapsed - 5.0
        if remaining < 2.0:
            break
        
        # Alternate between steepest and random walk
        if i % 2 == 0:
            h_result, score = _local_search_steepest(h_init, min(time_per_init, remaining))
        else:
            h_result, score = _local_search_random_walk(h_init, min(time_per_init, remaining))
        
        if score < best_score:
            best_score = score
            best_h = h_result.copy()
    
    # If no good result, fallback
    if best_h is None:
        best_h = np.ones(N, dtype=np.int8)
        best_score = _compute_score(best_h)
    
    # Final polish with remaining time
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed - 2.0
    if remaining > 5.0:
        h_polished, score_polished = _local_search_steepest(best_h, remaining)
        if score_polished < best_score:
            best_h = h_polished
            best_score = score_polished
    
    return best_h
