"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START

def _fft_cardinality(vec):
    """
    Given a binary vector vec (length N), compute the number of distinct
    values in the sumset (convolution) and difference set.
    We use FFT to compute the convolution for sums, and a similar trick for differences.
    
    For sums: sumset = {a+b : a,b in U}. The indicator of the sumset is 
    (1 if count > 0 else 0) where count = (vec * vec)[s].
    
    For differences: diffset = {a-b : a,b in U}. We can compute this by
    convolving vec with the reversed vec, or by noting that a-b = a + (-b).
    Actually, the difference set {a-b} can be computed as follows:
    Let w = vec. The number of pairs (a,b) with a-b = d is sum_a w[a] * w[a-d].
    This is the cross-correlation. We can compute it via FFT:
    corr[d] = sum_a w[a] * w[a-d]
    which is the same as convolution of w with reversed w.
    
    Let's be careful:
    sumset: s = a+b. count_s = sum_{a+b=s} w[a]*w[b] = (w*w)[s]
    diffset: d = a-b. count_d = sum_{a-b=d} w[a]*w[b] = sum_a w[a]*w[a-d]
    
    For the difference set, we can use the fact that:
    count_d = sum_a w[a] * w[a-d]
    This is the cross-correlation of w with itself.
    
    Using FFT:
    - For sums: FFT(w) * FFT(w) -> IFFT -> real part gives counts
    - For differences: FFT(w) * conj(FFT(w)) -> IFFT -> real part gives counts
      Wait, that gives the autocorrelation: sum_a w[a] * w[a+k] for lag k.
      Actually, the standard autocorrelation r[k] = sum_n w[n] * w[n+k] (or w[n-k]).
      
    Let me think again. The difference set is {a - b : a, b in U}.
    For a fixed difference d = a - b, the number of pairs is sum_b w[b+d] * w[b].
    This is the cross-correlation at lag d.
    
    Using FFT, the cross-correlation can be computed as:
    IFFT(FFT(w) * conj(FFT(w))) gives the autocorrelation.
    Specifically, if we compute c = IFFT(FFT(w) * conj(FFT(w))), then
    c[k] = sum_n w[n] * w[n-k] (with appropriate indexing).
    
    Actually, let's use a cleaner approach. We'll zero-pad to length 2*N-1 or more.
    For sums: pad to length >= 2*max_val, compute FFT, square, IFFT.
    For differences: the range of differences is from -max_val to max_val.
    We can shift the difference set to be non-negative by adding max_val.
    
    Let me implement this carefully.
    """
    n = len(vec)
    # For sums: max sum is 2*(n-1), so we need length at least 2*n-1
    # For differences: range is -(n-1) to (n-1), so we need length at least 2*n-1
    
    # Use next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    
    # Zero-pad
    vec_padded = np.zeros(fft_size, dtype=np.float64)
    vec_padded[:n] = vec
    
    # FFT
    f = np.fft.rfft(vec_padded)
    
    # Sumset: |U+U| = number of s where (w*w)[s] > 0
    f_sum = f * f
    conv_sum = np.fft.irfft(f_sum, n=fft_size)
    # The sumset values range from 0 to 2*(n-1)
    sumset_count = np.sum(conv_sum[:2*n-1] > 0.5)  # threshold for numerical stability
    
    # Difference set: |U-U|
    # The difference d = a-b ranges from -(n-1) to (n-1)
    # We compute the autocorrelation
    f_diff = f * np.conj(f)
    autocorr = np.fft.irfft(f_diff, n=fft_size)
    # autocorr[k] for k=0 to fft_size-1
    # The actual differences are at positions:
    # d=0 -> k=0
    # d=1 -> k=1 (and also k=fft_size-1 for d=-1 due to circularity, but we padded enough)
    # Actually, with zero-padding to fft_size >= 2*n-1, the linear autocorrelation is:
    # autocorr[k] = sum_i w[i] * w[i+k] for k >= 0
    # autocorr[fft_size-k] = sum_i w[i] * w[i-k] for k > 0
    # So differences d >= 0 are at indices d, and d < 0 are at indices fft_size + d
    diffset_count = 0
    for d in range(-(n-1), n):
        if d >= 0:
            idx = d
        else:
            idx = fft_size + d
        if autocorr[idx] > 0.5:
            diffset_count += 1
    
    return sumset_count, diffset_count


def _compute_score(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    U_sorted = sorted(U)
    max_val = max(U_sorted)
    n = len(U_sorted)
    
    # Create binary vector
    vec = np.zeros(max_val + 1, dtype=np.float64)
    for u in U_sorted:
        vec[u] = 1.0
    
    sumset_size, diffset_size = _fft_cardinality(vec)
    
    if sumset_size == 0 or max_val == 0:
        return 0.0
    
    score = 1.0 + math.log(diffset_size / sumset_size) / math.log(2 * max_val + 1)
    return score


def _greedy_improve(U, max_iter=500, time_limit=10.0):
    """
    Greedily add elements to U to maximize the score.
    """
    U_set = set(U)
    best_score = _compute_score(U_set)
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        best_gain = 0
        best_candidate = None
        max_val = max(U_set) if U_set else 0
        
        # Try adding elements in a range
        # We'll try a range of potential new elements
        candidates = set()
        # Try small additions near existing elements
        for u in U_set:
            for delta in range(1, max(5, int(math.sqrt(max_val + 1)) + 1)):
                candidates.add(u + delta)
                if u - delta >= 0:
                    candidates.add(u - delta)
        # Also try some larger jumps
        for _ in range(20):
            candidates.add(random.randint(1, max_val * 2 + 10))
        
        # Limit candidates for efficiency
        if len(candidates) > 200:
            candidates = list(candidates)
            random.shuffle(candidates)
            candidates = set(candidates[:200])
        
        for c in candidates:
            if c in U_set:
                continue
            new_U = U_set | {c}
            s = _compute_score(new_U)
            gain = s - best_score
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
        
        if best_candidate is not None and best_gain > 1e-10:
            U_set.add(best_candidate)
            best_score += best_gain
        else:
            break
    
    return list(U_set), best_score


def _simulated_annealing(U, time_limit=30.0):
    """
    Simulated annealing to optimize the score.
    """
    U_set = set(U)
    best_score = _compute_score(U_set)
    best_U = set(U_set)
    start_time = time.time()
    
    T = 0.1  # initial temperature
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        current_score = _compute_score(U_set)
        
        # Generate a random move
        move_type = random.random()
        max_val = max(U_set) if U_set else 0
        
        if move_type < 0.4:
            # Add a random element
            new_elem = random.randint(1, max_val * 3 + 10)
            if new_elem not in U_set:
                new_U = U_set | {new_elem}
            else:
                continue
        elif move_type < 0.7:
            # Remove a random element (not 0)
            removable = [u for u in U_set if u != 0]
            if not removable:
                continue
            to_remove = random.choice(removable)
            new_U = U_set - {to_remove}
        else:
            # Shift a random element
            movable = [u for u in U_set if u != 0]
            if not movable:
                continue
            old_elem = random.choice(movable)
            shift = random.randint(-max(1, max_val // 10), max(1, max_val // 10))
            new_elem = old_elem + shift
            if new_elem < 0 or new_elem in U_set:
                continue
            new_U = (U_set - {old_elem}) | {new_elem}
        
        new_score = _compute_score(new_U)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            U_set = new_U
            if new_score > best_score:
                best_score = new_score
                best_U = set(U_set)
        
        T *= cooling_rate
    
    return list(best_U), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with a small seed set
    U = [0, 1, 3]
    
    # Phase 1: Greedy improvement
    U, score = _greedy_improve(U, max_iter=1000, time_limit=20.0)
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 35.0:
        U, score = _simulated_annealing(U, time_limit=min(30.0, deadline - time.time() - 5.0))
    
    # Ensure 0 is in the set
    if 0 not in U:
        U.append(0)
    
    return sorted(U)

# EVOLVE-BLOCK-END
