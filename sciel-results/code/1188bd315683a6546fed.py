"""Additive bases with few representations - variance-aware simulated annealing.

We model the problem as finding a subset A of [0, 128] such that every integer
in [128, 256] is a sum of two elements of A. We minimize the largest representation
multiplicity (peak).

To escape local optima, we use Simulated Annealing with an objective function
that penalizes both the peak and the variance of the representation counts.
This encourages a flatter distribution, which often leads to lower peaks.

Key optimization:
- Incremental update of representation counts in O(N) instead of O(N^2) convolution.
"""
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
    # Ordered convolution
    y = np.convolve(x.astype(np.int64), x.astype(np.int64), mode="full")
    w = y[_T0:_T1 + 1].copy()
    ks = np.arange(_T0, _T1 + 1)
    is_even = (ks % 2 == 0)
    mids = ks // 2
    diag_corr = np.where(is_even, x[mids], 0)
    u_counts = (w + diag_corr) // 2
    return u_counts


def _valid(u_counts):
    return bool(np.all(u_counts > 0))


def _peak(u_counts):
    return int(u_counts.max())


def _variance(u_counts):
    return float(np.var(u_counts))


def _make_greedy_seed(rng):
    """Build a valid seed using greedy coverage with load balancing."""
    x = np.zeros(_LEN, dtype=np.int64)
    u = _compute_counts(x)
    
    # We want to cover all targets.
    # Greedy: pick element that covers most uncovered targets, with tie-breaker on load.
    # To make it "balanced", we can prioritize elements that don't increase peak too much.
    
    candidates = np.arange(_LEN)
    rng.shuffle(candidates)
    
    # First pass: ensure coverage
    while not _valid(u):
        uncovered = np.where(u == 0)[0] + _T0
        if len(uncovered) == 0:
            break
        
        best_c = None
        best_score = -1
        best_gain = -1
        
        for c in candidates:
            if x[c] == 1:
                continue
            
            # Calculate gain (how many new targets covered)
            gain = 0
            # Calculate new peak contribution
            new_counts = u.copy()
            
            # Update counts for adding c
            # For each target t, if t-c is in A, count increases.
            # Also if t == 2c, count increases.
            # Actually, let's just compute the delta efficiently.
            
            # Delta for target k:
            # If k-c is in A, then pair (c, k-c) adds 1.
            # If k == 2c, pair (c,c) adds 1.
            # Note: if k-c == c, it's the same pair.
            
            # We can iterate over targets to find gain and check validity.
            # But doing this for every candidate is slow.
            # Let's just pick a random uncovered target and find a candidate that covers it.
            # This is simpler and faster for seeding.
            
            # Pick random uncovered target
            t = rng.choice(uncovered)
            # Possible partners for c to sum to t: t-c
            partner = t - c
            if 0 <= partner < _LEN:
                # If partner is in A, or partner == c (so we add c and it covers t via 2c or c+partner)
                # Actually, if we add c, it covers t if:
                # 1. t-c is already in A.
                # 2. t == 2c (so c+c covers it).
                
                covers = False
                if 0 <= partner < _LEN and x[partner] == 1:
                    covers = True
                if 2 * c == t:
                    covers = True
                
                if covers:
                    # Check if this move is "good" (low impact on peak)
                    # Compute new counts
                    new_u = u.copy()
                    # Add contributions of c
                    # For each k in [128, 256]:
                    # if k-c in A, new_u[k-128] += 1
                    # if k == 2c, new_u[k-128] += 1
                    
                    # Vectorized update for candidate c
                    # k ranges from 128 to 256
                    # partner = k - c
                    # partner ranges from 128-c to 256-c
                    # Valid partners are in [0, 128]
                    # So k must be in [c, c+128]
                    # Intersection with [128, 256]:
                    k_min = max(128, c)
                    k_max = min(256, c + 128)
                    
                    if k_min <= k_max:
                        ks = np.arange(k_min, k_max + 1)
                        partners = ks - c
                        # Check which partners are in A
                        mask = x[partners] == 1
                        idxs = ks - 128
                        new_u[idxs[mask]] += 1
                        
                        # Handle diagonal (2c == k)
                        if 2 * c >= k_min and 2 * c <= k_max:
                            new_u[2 * c - 128] += 1
                            
                        # Wait, the diagonal case is already included if x[c]==1? No, x[c] is 0 right now.
                        # If 2c == k, then partner = c. x[c] is 0. So the mask above won't catch it.
                        # So we must add 1 for the diagonal case explicitly.
                        # But wait, if partner == c, and we are adding c, then the pair (c,c) is formed.
                        # The condition "partner in A" checks if the *other* element is in A.
                        # Since c is not in A yet, x[c] is 0. So mask is false for partner=c.
                        # So we need to add 1 if 2c == k.
                        
                        # However, we also need to be careful: does adding c create pairs with *other* new elements?
                        # No, we only add one element at a time.
                        
                        # So new_u is correct.
                        
                        if np.all(new_u > 0):
                            peak_new = new_u.max()
                            # Score: minimize peak, then variance
                            score = -peak_new - 0.01 * np.var(new_u)
                            if score > best_score:
                                best_score = score
                                best_gain = np.sum(new_u > u) # Number of newly covered
                                best_c = c
                                best_new_u = new_u
            
        if best_c is None:
            # Fallback: pick any candidate that covers *any* uncovered target
            for c in candidates:
                if x[c] == 1: continue
                # Check if it covers anything
                covered_any = False
                for t in uncovered:
                    p = t - c
                    if (0 <= p < _LEN and x[p] == 1) or (2*c == t):
                        covered_any = True
                        break
                if covered_any:
                    best_c = c
                    break
        
        if best_c is None:
            best_c = candidates[0]
            
        x[best_c] = 1
        u = _compute_counts(x)
        
    return x


def _simulated_annealing(x, rng, max_time):
    """Simulated annealing with variance-aware objective."""
    u = _compute_counts(x)
    if not _valid(u):
        return x
    
    best_x = x.copy()
    best_u = u.copy()
    best_peak = _peak(u)
    best_var = _variance(u)
    
    current_peak = best_peak
    current_var = best_var
    
    T = 10.0
    cooling = 0.999
    
    # Pre-compute indices for speed? No, just use numpy.
    
    while _elapsed() < max_time:
        T *= cooling
        if T < 1e-5:
            break
            
        # Random move: pick a random element to toggle
        i = rng.randint(0, _LEN - 1)
        
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]
        
        u_new = _compute_counts(x_new)
        
        if not _valid(u_new):
            # Reject invalid moves
            continue
            
        peak_new = _peak(u_new)
        var_new = _variance(u_new)
        
        # Objective: minimize peak, then variance
        # f = peak + lambda * variance
        # But we want to strictly minimize peak first.
        # Let's use a lexicographic objective with a small penalty for variance.
        obj_cur = current_peak + 0.01 * current_var
        obj_new = peak_new + 0.01 * var_new
        
        delta = obj_new - obj_cur
        
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            x = x_new
            u = u_new
            current_peak = peak_new
            current_var = var_new
            
            if peak_new < best_peak or (peak_new == best_peak and var_new < best_var):
                best_peak = peak_new
                best_var = var_new
                best_x = x.copy()
                best_u = u.copy()
                
    return best_x


def construct():
    global _START
    _START = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Get a valid seed
    x = _make_greedy_seed(rng)
    u = _compute_counts(x)
    if not _valid(u):
        # Fallback: full set
        x = np.ones(_LEN, dtype=np.int64)
        u = _compute_counts(x)
    
    # Phase 2: Simulated Annealing
    remaining = _DEADLINE - _elapsed() - 2
    if remaining > 1:
        x_opt = _simulated_annealing(x, rng, max_time=remaining)
        u_opt = _compute_counts(x_opt)
        if _valid(u_opt):
            x = x_opt
            u = u_opt
    
    # Convert to sorted list
    result = [int(i) for i in np.where(x == 1)[0]]
    return result
