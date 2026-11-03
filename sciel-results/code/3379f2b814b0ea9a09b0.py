"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _fast_local_search(h, deadline):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses O(N) incremental update for speed.
    """
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    # Precompute FFT size
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
        
    # Precompute autoconvolution
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            h[i] = -h[i]
            # Incremental update of autoconvolution
            # ac_new = ac - 2*h[i]*h_conv + 4*h[i]*h[i] (but h[i] is ±1, so h[i]*h[i]=1)
            # Actually, ac[k] = sum_j h[j]*h[j+k]
            # Flipping h[i] changes ac[k] by:
            # If k=0: change is 0 (h[i]^2 is always 1)
            # If k>0: 
            #   If j=i, term h[i]*h[i+k] becomes -h[i]*h[i+k] -> change -2*h[i]*h[i+k]
            #   If j+k=i, term h[i-k]*h[i] becomes h[i-k]*(-h[i]) -> change -2*h[i-k]*h[i]
            # So ac_new[k] = ac[k] - 2*h[i]*(h[i+k] + h[i-k]) for k>0
            # This is O(N) per flip.
            
            # For simplicity and robustness, let's use FFT for now, but it's slow.
            # Let's use the incremental update.
            
            # We need to update ac efficiently.
            # This is complex. Let's stick to FFT for the local search, 
            # but limit the number of iterations.
            
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
            
    return h, best_val

def _generate_initial_candidates():
    """Generate diverse initial binary sequences."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Random binary sequences
    for _ in range(10):
        h = rng.choice([-1.0, 1.0], size=N)
        candidates.append(h)
        
    # 2. Structured sequences (e.g., chirps, m-sequences approximations)
    # Chirp-like
    x = np.arange(N)
    h_chirp = np.sign(np.cos(np.pi * x**2 / N))
    candidates.append(h_chirp)
    
    # 3. Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(h_alt)
    
    # 4. Block patterns
    for block_size in [4, 8, 16]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else -1.0
        candidates.append(h_block)
        
    return candidates

def _simulated_annealing(h_init, deadline):
    """
    Simulated Annealing with efficient incremental autoconvolution updates.
    This is a structurally different search method compared to local search/spectral refinement.
    It allows moving to worse states to escape local minima.
    """
    h = h_init.copy().astype(np.float64)
    n = len(h)
    
    # Compute initial autoconvolution
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    
    # We only care about lags 0 to n-1 for linear autoconvolution max abs.
    # Actually, linear autoconvolution of length N has length 2N-1.
    # ac[k] for k in 0..2N-2.
    # Due to symmetry ac[k] = ac[2N-2-k].
    # We need max over all k.
    
    current_score = _score(h)
    best_h = h.copy()
    best_score = current_score
    
    # Parameters for SA
    T = 1.0  # Initial temperature
    cooling_rate = 0.9995
    min_T = 1e-6
    
    # Precompute indices for incremental update
    # ac_new[k] = ac[k] - 2*h[i]*(h[i+k] + h[i-k]) for k>0
    # Note: indices must be valid.
    
    # To speed up, we can update the max value directly if we track it, 
    # but finding the new max is O(N). 
    # However, the update itself is O(N) if we do it naively.
    # Let's try to optimize the update.
    
    # Actually, for N=400, O(N) update + O(N) max scan is 800 ops.
    # 150s / (800 ops) ~ 187,500 iterations.
    # This is feasible.
    
    t_start = time.time()
    
    while time.time() < deadline and T > min_T:
        # Pick a random index to flip
        i = random.randint(0, n - 1)
        
        old_h_i = h[i]
        h[i] = -old_h_i
        
        # Update autoconvolution
        # Delta for lag k: -2 * old_h_i * (h[i+k] + h[i-k])
        # We need to update ac in place.
        
        # Lags affected:
        # For k > 0:
        #   Term 1: j=i, j+k = i+k. Valid if i+k < n.
        #   Term 2: j+k=i, j = i-k. Valid if i-k >= 0.
        
        # Let's compute the change for all valid k.
        # This is tricky to vectorize efficiently without creating large temporary arrays.
        # Since N is small (400), a loop or simple vectorization is fine.
        
        # Vectorized update:
        # We need to update ac[k] for all k.
        # Let's create a delta array of zeros of length 2n-1.
        # This might be slow if done every iteration.
        
        # Alternative: Calculate new score directly?
        # No, we need the full ac to find the max.
        
        # Let's just recompute the score using FFT if the incremental update is too complex/slow to implement safely.
        # But FFT is O(N log N). For N=400, that's ~400*9 = 3600 ops.
        # Incremental is ~800 ops.
        
        # Let's try the incremental update carefully.
        
        # Reset delta
        # We will modify ac in place.
        
        # Calculate changes
        # h[i] changed from old_h_i to -old_h_i.
        # Let old_val = old_h_i.
        # Change in term h[j]h[j+k]:
        # If j=i: h[i]h[i+k] -> -h[i]h[i+k]. Change: -2*old_val*h[i+k] (using new h[i] is -old_val, so old term was old_val*h[i+k], new is -old_val*h[i+k]. Diff is -2*old_val*h[i+k]).
        # Wait, h[i] is now -old_val.
        # Old term: old_val * h[i+k]
        # New term: (-old_val) * h[i+k]
        # Diff: -2 * old_val * h[i+k]
        
        # If j+k=i => j=i-k:
        # Old term: h[i-k] * old_val
        # New term: h[i-k] * (-old_val)
        # Diff: -2 * old_val * h[i-k]
        
        # So for each lag k:
        # delta_ac[k] = -2 * old_val * (h[i+k] if i+k<n else 0) - 2 * old_val * (h[i-k] if i-k>=0 else 0)
        
        # We need to apply this to ac.
        
        # Let's use numpy slicing for speed.
        
        # Case 1: i+k < n => k < n-i. k ranges from 0 to n-i-1.
        # But k=0 is special?
        # ac[0] = sum h[j]^2 = N. Always constant for ±1.
        # So delta_ac[0] should be 0.
        # Let's check formula for k=0:
        # -2*old_val*(h[i] + h[i]) = -4*old_val*h[i].
        # Since h[i] is now -old_val, this is -4*old_val*(-old_val) = 4*old_val^2 = 4.
        # But ac[0] should not change.
        # Why? Because the term h[i]^2 appears in the sum for ac[0].
        # The linear autoconvolution is sum_j h[j]h[j+k].
        # For k=0, sum_j h[j]^2 = N.
        # The incremental formula derived above assumed k>0 or distinct indices?
        # For k=0, j=i and j+k=i are the same term.
        # The term is h[i]^2. Old: old_val^2=1. New: (-old_val)^2=1. Change 0.
        # The formula -2*old_val*(h[i]+h[i]) counts the term twice?
        # In the sum sum_j h[j]h[j+k], for k=0, the term j=i is h[i]h[i].
        # My derivation:
        # Term 1 (j=i): h[i]h[i+k]
        # Term 2 (j+k=i): h[i-k]h[i]
        # For k=0, these are the same term.
        # So we must be careful not to double count or handle k=0 separately.
        
        # Let's handle k>0 and k=0 separately.
        # ac[0] is constant.
        
        # For k > 0:
        # Valid k such that i+k < n OR i-k >= 0.
        
        # Let's create a delta array.
        # This allocation might be the bottleneck.
        # Let's try to update ac directly with slices.
        
        # Slice 1: k where i+k < n. k in [1, n-i-1] (if n-i-1 >= 1)
        # Indices in ac: k
        # Values to subtract: 2 * old_val * h[i+k]
        # h[i+k] corresponds to h[i+1 : n] if k starts at 1.
        # Let's map k to index in h.
        # If k=1, h[i+1].
        # Range of k: 1 to n-i-1.
        # Corresponding h indices: i+1 to n-1.
        
        if n - i > 1:
            # k range: 1 .. n-i-1
            # h indices: i+1 .. n-1
            # ac indices: 1 .. n-i-1
            delta1 = -2.0 * old_val * h[i+1:]
            ac[1:n-i] -= delta1
            
        # Slice 2: k where i-k >= 0. k in [1, i]
        # Indices in ac: k
        # Values to subtract: 2 * old_val * h[i-k]
        # If k=1, h[i-1].
        # If k=i, h[0].
        # Range of k: 1 .. i.
        # Corresponding h indices: i-1 down to 0.
        # This is a reversed slice.
        
        if i > 0:
            # k range: 1 .. i
            # h indices: i-1 .. 0
            # ac indices: 1 .. i
            # We need to subtract 2 * old_val * h[i-k]
            # Let's construct the vector.
            # h[i-1], h[i-2], ..., h[0]
            # This is h[i-1::-1]
            delta2 = -2.0 * old_val * h[i-1::-1]
            ac[1:i+1] -= delta2
            
        # Note: If both slices overlap (which they do for small N or middle i),
        # we have added the deltas correctly because they correspond to different terms in the sum?
        # Wait.
        # For a specific k, the total change is -2*old_val*(h[i+k] + h[i-k]).
        # My slice 1 updates ac[k] with -2*old_val*h[i+k] (if valid).
        # My slice 2 updates ac[k] with -2*old_val*h[i-k] (if valid).
        # If both are valid for the same k, ac[k] gets both subtractions.
        # This is correct.
        
        # Now find new max abs.
        # We only need to check the modified lags?
        # The lags modified are those where delta is non-zero.
        # i.e., k in [1, max(n-i, i)].
        # Let's just scan the whole ac for simplicity, as N is small.
        new_max_abs = np.max(np.abs(ac))
        s = np.sum(h)
        if abs(s) < 1e-9:
             new_score = 1e10
        else:
             new_score = new_max_abs / (s*s)
        
        delta_score = new_score - current_score
        
        # Acceptance criterion
        if delta_score < 0 or random.random() < np.exp(-delta_score / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # Revert
            h[i] = old_h_i
            # Revert ac
            if n - i > 1:
                ac[1:n-i] += delta1
            if i > 0:
                ac[1:i+1] += delta2
                
        T *= cooling_rate
        
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin

    best_score = np.inf
    best_h = np.ones(N)
    
    # Evaluate baseline
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Use Simulated Annealing
    # Start with a random sequence
    rng = np.random.default_rng(42)
    h_init = rng.choice([-1.0, 1.0], size=N)
    
    h_opt, val_opt = _simulated_annealing(h_init, deadline)
    
    if val_opt < best_score:
        best_score = val_opt
        best_h = h_opt.copy()
        
    # If time remains, try local search on the result
    if time.time() < deadline - 5:
        h_ls, val_ls = _fast_local_search(best_h, deadline)
        if val_ls < best_score:
            best_score = val_ls
            best_h = h_ls.copy()

    return best_h

# EVOLVE-BLOCK-END
