import numpy as np
import time

def _compute_score_fast(U):
    """Compute the score using FFT for sumset/difference set sizes."""
    U = sorted(set(U))
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Sumset: U+U ranges from 0 to 2*max_u
    # Difference set: U-U ranges from -max_u to max_u
    # Use FFT convolution for sumset
    n = 2 * max_u + 1
    # Next power of 2 for FFT
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    vec = np.zeros(fft_size, dtype=np.float64)
    for u in U:
        vec[u] = 1.0
    conv = np.fft.irfft(np.fft.rfft(vec) * np.fft.rfft(vec), n=fft_size)
    # Sumset size: number of positions where conv > 0.5
    sumset_size = int(np.sum(conv[:n] > 0.5))
    # Difference set: U-U. This is the autocorrelation.
    # U-U = {a-b : a,b in U}. Use convolution of vec with reversed vec.
    # Shift: (U-U) + max_u ranges from 0 to 2*max_u
    vec_rev = vec.copy()
    vec_rev[:max_u+1] = vec[max_u::-1]
    vec_rev[max_u+1:] = 0
    conv_diff = np.fft.irfft(np.fft.rfft(vec) * np.fft.rfft(vec_rev), n=fft_size)
    diffset_size = int(np.sum(conv_diff[:n] > 0.5))
    # Score
    denom = np.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    return 1.0 + np.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    best_score = 0.0
    best_U = [0, 1, 3]
    
    # Strategy 1: Two-part construction {0} ∪ {a, a+1, ..., a+k}
    # where a > k to ensure sum-free property
    # U+U = {0} ∪ [a, a+k] ∪ [2a, 2a+2k] (disjoint if a > k)
    # |U+U| = 1 + (k+1) + (k+1) = 2k+3
    # |U| = k+2
    # U-U: differences between 0 and [a,a+k]: [-a-k, -a] and [a, a+k]
    # differences within [a,a+k]: [-(k), k]
    # So U-U = [-(a+k), a+k] but with gaps
    # Let's just compute it
    
    # Search over a and k
    # max_u = a+k
    # We want to balance: larger k gives larger |U-U|/|U+U|
    # but larger max_u increases denominator
    
    # Let's try various (a, k) pairs
    # The optimal ratio for this family:
    # |U+U| = 2k+3 (for a > k)
    # |U-U| = ? Let's compute: 
    #   0 - {a..a+k} = {-a-k .. -a} (k+1 values)
    #   {a..a+k} - 0 = {a .. a+k} (k+1 values)
    #   {a..a+k} - {a..a+k} = {-k .. k} (2k+1 values)
    #   0-0 = 0 (already counted)
    # Union: [-a-k, -a] ∪ [-k, k] ∪ [a, a+k]
    # If a > k: these are disjoint except possibly at boundaries
    # [-a-k, -a] and [-k, k]: disjoint if a > k (since -a < -k)
    # [-k, k] and [a, a+k]: disjoint if a > k
    # So |U-U| = (k+1) + (2k+1) + (k+1) = 4k+3 (if a > k)
    # Wait, need to be careful about 0 being in all three? No.
    # [-a-k, -a] doesn't contain 0 if a > 0
    # [-k, k] contains 0
    # [a, a+k] doesn't contain 0 if a > 0
    # So total = (k+1) + (2k+1) + (k+1) = 4k+3
    
    # Score = 1 + log((4k+3)/(2k+3)) / log(2*(a+k)+1)
    # To maximize: for fixed k, minimize a (but a > k, so a = k+1)
    # Then max_u = 2k+1
    # Score = 1 + log((4k+3)/(2k+3)) / log(4k+3)
    
    # Let's evaluate this for various k
    for k in range(1, 10000):
        if time.time() > deadline:
            break
        a = k + 1  # minimal a for sum-free
        U = [0] + list(range(a, a + k + 1))
        max_u = a + k
        sumset_size = 2 * k + 3
        diffset_size = 4 * k + 3
        denom = np.log(2 * max_u + 1)
        if denom <= 0:
            continue
        score = 1.0 + np.log(diffset_size / sumset_size) / denom
        if score > best_score:
            best_score = score
            best_U = U
    
    # Strategy 2: Geometric/exponential sets for better difference/sum ratio
    # Try {0, 1, 2^1, 2^2, ..., 2^m} type constructions
    # Or mixed: {0} ∪ {a^i for i=0..m}
    
    # Strategy 3: Random search with FFT evaluation
    rng = np.random.default_rng(42)
    for attempt in range(500):
        if time.time() > deadline:
            break
        # Random size
        n_elems = rng.integers(5, 200)
        # Random max
        max_u = rng.integers(10, 5000)
        # Random set containing 0
        U_set = {0}
        while len(U_set) < n_elems:
            U_set.add(int(rng.integers(1, max_u + 1)))
        U = sorted(U_set)
        score = _compute_score_fast(U)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Strategy 4: Local improvement on best_U
    if best_U and time.time() < deadline - 10:
        current = sorted(best_U)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            max_u = max(current) if current else 0
            # Try adding elements
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                new_elem = int(rng.integers(1, max_u + 50))
                new_U = sorted(current + [new_elem])
                if len(new_U) != len(set(new_U)):
                    continue
                new_score = _compute_score_fast(new_U)
                if new_score > current_score:
                    current = new_U
                    current_score = new_score
                    best_score = new_score
                    best_U = current
                    improved = True
                    break
            # Try removing elements (not 0)
            if len(current) > 2:
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    idx = int(rng.integers(1, len(current)))
                    new_U = current[:idx] + current[idx+1:]
                    new_score = _compute_score_fast(new_U)
                    if new_score > current_score:
                        current = new_U
                        current_score = new_score
                        best_score = new_score
                        best_U = current
                        improved = True
                        break
    
    return best_U
