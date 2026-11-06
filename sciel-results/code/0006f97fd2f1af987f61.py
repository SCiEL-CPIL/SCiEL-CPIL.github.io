import time
import random
import numpy as np

_N = 128
_LEN = 129  # indices 0..128
_T0 = 128
_T1 = 256
_WIN = _T1 - _T0 + 1  # 129 targets

_START = time.time()
_DEADLINE = 88.0  # safety margin before the 90s kill

def _elapsed():
    return time.time() - _START

def _compute_counts(x):
    """Compute unordered pair counts for targets [128, 256]."""
    # x is a numpy array of 0s and 1s of length _LEN
    # We want U[s] for s in [128, 256]
    # U[s] = number of pairs {a, b} with a, b in A, a <= b, a+b=s
    
    # Efficient way using convolution is good for initialization, 
    # but for updates we will use incremental logic.
    # For initial computation:
    y = np.convolve(x, x, mode="full")
    # y[k] is ordered count for sum k.
    # Unordered count for sum k:
    # if k is even, (y[k] + x[k//2]) // 2
    # if k is odd, y[k] // 2
    
    u_counts = np.zeros(_WIN, dtype=np.int64)
    for s in range(_T0, _T1 + 1):
        idx = s - _T0
        if s % 2 == 0:
            mid = s // 2
            if 0 <= mid < _LEN:
                u_counts[idx] = (y[s] + x[mid]) // 2
            else:
                u_counts[idx] = y[s] // 2
        else:
            u_counts[idx] = y[s] // 2
    return u_counts

def _valid(u_counts):
    return bool(np.all(u_counts > 0))

def _peak(u_counts):
    return int(u_counts.max())

def _variance(u_counts):
    return float(np.var(u_counts))

def _make_greedy_seed(rng, target_size=55):
    """Build a valid seed using greedy coverage."""
    x = np.zeros(_LEN, dtype=np.int64)
    
    # Start with a random set of size target_size
    candidates = np.arange(_LEN)
    rng.shuffle(candidates)
    x[candidates[:target_size]] = 1
    
    u = _compute_counts(x)
    
    # Greedy repair: add elements to cover uncovered targets
    while not _valid(u):
        uncovered = np.where(u == 0)[0] + _T0
        if len(uncovered) == 0:
            break
        
        best_c = None
        best_gain = -1
        
        # Try to find an element that covers many uncovered targets
        # We can just pick a random uncovered target and try to cover it
        # But to be efficient, let's iterate through candidates not in A
        for c in candidates:
            if x[c] == 1:
                continue
            
            # Check if adding c covers any uncovered target
            gain = 0
            for t in uncovered:
                p = t - c
                if 0 <= p < _LEN:
                    if x[p] == 1 or p == c: # p==c means 2c=t
                        gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
            if best_gain > 0:
                break # Found a good one? Or keep looking for best?
                # For speed, let's just take the first one with gain > 0
                # Or we can scan all. Scanning all 129*129 is fast enough for seed.
        
        if best_c is None:
            # Fallback: pick random
            best_c = rng.choice(candidates)
            if x[best_c] == 1:
                continue # Should not happen if we skip x==1
            
        x[best_c] = 1
        u = _compute_counts(x)
        
    return x

def _local_search(x, rng, max_time):
    """
    Fast local search using incremental count updates.
    """
    u = _compute_counts(x)
    if not _valid(u):
        return x
    
    best_x = x.copy()
    best_u = u.copy()
    best_peak = _peak(u)
    best_var = _variance(u)
    
    current_peak = best_peak
    current_var = best_var
    
    # Precompute indices for fast lookup
    # We need to know which targets are affected by a toggle at i.
    # For a toggle at i, affected sums are i + j for j in A.
    
    while _elapsed() < max_time:
        improved = False
        
        # Try all possible single-bit flips
        # We can randomize the order to avoid cycles
        order = np.arange(_LEN)
        rng.shuffle(order)
        
        for i in order:
            # Calculate delta for toggling i
            # We need to update u for sums s = i + j where j in A
            
            # List of j in A
            # We can maintain a list of active indices for speed?
            # Or just use np.where(x==1)
            # np.where is O(N). N=129. This is fine.
            active = np.where(x == 1)[0]
            
            # We will simulate the toggle
            new_u = u.copy()
            
            if x[i] == 0:
                # Adding i
                # For each j in active:
                #   s = i + j
                #   if s in [128, 256]: new_u[s-128] += 1
                # Also, if i == i, s = 2i. 
                # The loop above covers j=i if i is in active? No, i is not in active yet.
                # So we need to handle self-pair separately?
                # If we add i, pairs are (i, j) for j in A_old.
                # If j == i, it's not in A_old. So (i,i) is a new pair.
                # So we must add 1 to U[2i] if 2i in range.
                
                for j in active:
                    s = i + j
                    if _T0 <= s <= _T1:
                        new_u[s - _T0] += 1
                
                # Handle self pair (i,i)
                s = 2 * i
                if _T0 <= s <= _T1:
                    new_u[s - _T0] += 1
                    
            else:
                # Removing i
                # Pairs (i, j) for j in A_old (which includes i)
                # If j != i: remove 1 from U[i+j]
                # If j == i: remove 1 from U[2i]
                
                for j in active:
                    s = i + j
                    if _T0 <= s <= _T1:
                        new_u[s - _T0] -= 1
                        # Note: if j == i, s=2i. We subtract 1.
                        # Is this correct?
                        # U(s) counts pairs {a,b}.
                        # If we remove i, we remove pairs {i, j} for all j in A.
                        # If j == i, we remove {i,i}.
                        # The loop iterates over all j in A.
                        # So if j=i is in active, we subtract 1 from U[2i].
                        # If j!=i, we subtract 1 from U[i+j].
                        # This seems correct.
                        
            if not _valid(new_u):
                continue
                
            peak_new = _peak(new_u)
            var_new = _variance(new_u)
            
            # Acceptance criterion
            # Minimize peak, then variance
            if peak_new < current_peak or (peak_new == current_peak and var_new < current_var):
                # Accept
                x[i] = 1 - x[i]
                u = new_u
                current_peak = peak_new
                current_var = var_new
                improved = True
                
                if peak_new < best_peak or (peak_new == best_peak and var_new < best_var):
                    best_peak = peak_new
                    best_var = var_new
                    best_x = x.copy()
                    best_u = u.copy()
                    
        if not improved:
            # Local optimum reached. 
            # Do a random perturbation (kick) to escape
            # Flip 2-5 random bits
            kicks = rng.randint(2, 5)
            for _ in range(kicks):
                i = rng.randint(0, _LEN - 1)
                x[i] = 1 - x[i]
            
            # Recompute u and check validity
            u = _compute_counts(x)
            if not _valid(u):
                # Repair? Or revert?
                # For now, revert to best_x and try again with different kicks?
                # Better: just revert to best_x
                x = best_x.copy()
                u = best_u.copy()
                current_peak = best_peak
                current_var = best_var
            else:
                current_peak = _peak(u)
                current_var = _variance(u)

    return best_x

def construct():
    global _START
    _START = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Get a valid seed
    # Try different sizes to find a good starting point
    best_seed_x = None
    best_seed_peak = float('inf')
    
    for size in [50, 55, 60]:
        x = _make_greedy_seed(rng, target_size=size)
        u = _compute_counts(x)
        if _valid(u):
            p = _peak(u)
            if p < best_seed_peak:
                best_seed_peak = p
                best_seed_x = x
    
    if best_seed_x is None:
        # Fallback: full set
        best_seed_x = np.ones(_LEN, dtype=np.int64)
    
    # Phase 2: Local Search
    remaining = _DEADLINE - _elapsed() - 2
    if remaining > 1:
        x_opt = _local_search(best_seed_x, rng, max_time=remaining)
        u_opt = _compute_counts(x_opt)
        if _valid(u_opt):
            best_seed_x = x_opt
    
    # Convert to sorted list
    result = [int(i) for i in np.where(best_seed_x == 1)[0]]
    return result
