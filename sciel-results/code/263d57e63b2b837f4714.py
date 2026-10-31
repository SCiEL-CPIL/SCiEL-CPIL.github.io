"""
Combined approach:
1. Structural idea from Inspiration 1: Sparse/dense signed sequences with local sign-flip optimization.
2. Refinement machinery from Inspiration 2: Thue-Morse initialization + smooth surrogate refinement (log-sum-exp).
3. Speedup: Precompute FFT of h once per flip, update autoconvolution incrementally where possible, or use fast FFT per flip.
   Actually, for N=400, FFT is very fast. The bottleneck is the number of evaluations.
   We will use a smarter local search: batched flips or greedy with lookahead?
   Better: Use the fact that flipping h[i] changes ac by -2*h[i]*h (shifted). We can update ac in O(N) instead of O(N log N).
   Let's implement O(N) update for autoconvolution when flipping a single element.
   ac_new = ac_old - 2*h[i]* (h convolved with delta at i)
   Actually, ac[k] = sum_j h[j]*h[j+k].
   If we flip h[i] to -h[i], the change in ac[k] is:
   For k such that j=i: term h[i]*h[i+k] becomes -h[i]*h[i+k] -> change -2*h[i]*h[i+k]
   For k such that j+k=i: term h[j]*h[i] becomes h[j]*(-h[i]) -> change -2*h[j]*h[i] where j = i-k
   So ac_new[k] = ac_old[k] - 2*h[i]*(h[i+k] + h[i-k]) (with appropriate bounds)
   This is O(N) per flip, which is much faster than O(N log N).
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autoconv_full(h):
    """Compute full autoconvolution of h using FFT."""
    f = np.fft.fft(h, n=2*N)
    ac = np.fft.ifft(f * f.conj()).real[:2*N-1]
    return ac

def _update_autoconv_after_flip(ac, h, i):
    """
    Update autoconvolution ac after flipping h[i] to -h[i].
    Returns new ac and updated h.
    """
    h_new = h.copy()
    h_new[i] = -h[i]
    
    ac_new = ac.copy()
    val = h[i]
    
    # For each lag k in [0, 2N-2]:
    # ac[k] = sum_{j} h[j]*h[j+k]
    # Terms involving h[i]:
    # 1. j = i: h[i]*h[i+k] -> if i+k < N, term is h[i]*h[i+k]
    # 2. j+k = i: h[j]*h[i] -> if j = i-k >= 0 and j < N, term is h[i-k]*h[i]
    
    # We need to subtract the old terms and add the new terms.
    # Old term 1: val * h[i+k] (if i+k < N)
    # New term 1: -val * h[i+k]
    # Change: -2*val*h[i+k]
    
    # Old term 2: val * h[i-k] (if i-k >= 0)
    # New term 2: -val * h[i-k]
    # Change: -2*val*h[i-k]
    
    # So ac_new[k] = ac[k] - 2*val*(h[i+k] (if valid) + h[i-k] (if valid))
    
    # Let's vectorize this update.
    # k ranges from 0 to 2N-2.
    # i+k is valid if i+k < N -> k < N-i
    # i-k is valid if i-k >= 0 -> k <= i
    
    # Create an array of changes of size 2N-1
    changes = np.zeros(2*N-1)
    
    # Part 1: k such that i+k < N => k in [0, N-i-1]
    if N - i > 0:
        k_start = 0
        k_end = N - i
        # indices in ac: k
        # indices in h: i+k
        changes[k_start:k_end] -= 2 * val * h[i+k_start:i+k_end]
        
    # Part 2: k such that i-k >= 0 => k in [0, i]
    if i >= 0:
        k_start = 0
        k_end = i + 1
        # indices in ac: k
        # indices in h: i-k (reversed)
        changes[k_start:k_end] -= 2 * val * h[i-k_start:i-k_end+1][::-1]
        
    ac_new += changes
    return ac_new, h_new

def _max_abs_autoconv(ac):
    return np.max(np.abs(ac))

def _score(h, ac):
    s = np.sum(h)
    if s == 0:
        return float('inf')
    return _max_abs_autoconv(ac) / (s * s)

def _local_search_fast(h, time_limit):
    """
    Local search using O(N) autoconvolution update.
    """
    h = h.copy().astype(float)
    ac = _autoconv_full(h)
    current_score = _score(h, ac)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            ac_new, h_new = _update_autoconv_after_flip(ac, h, i)
            new_score = _score(h_new, ac_new)
            
            if new_score < current_score:
                h = h_new
                ac = ac_new
                current_score = new_score
                improved = True
            # else: revert (do nothing, h and ac remain unchanged)
                
    return h, ac, current_score

def _thue_morse(n):
    seq = np.array([0])
    while len(seq) < n:
        seq = np.concatenate([seq, 1 - seq])
    return seq[:n]

def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2."""
    deadline = 120.0
    start_time = time.time()
    
    best_h = None
    best_score = float('inf')
    best_ac = None
    
    # Candidate 1: Thue-Morse
    tm = _thue_morse(N)
    h_tm = 1 - 2 * tm  # Convert 0/1 to 1/-1
    ac_tm = _autoconv_full(h_tm)
    score_tm = _score(h_tm, ac_tm)
    
    if score_tm < best_score:
        best_score = score_tm
        best_h = h_tm.copy()
        best_ac = ac_tm.copy()
        
    # Candidate 2: Random signs
    np.random.seed(42)
    h_rand = np.random.choice([1, -1], size=N).astype(float)
    ac_rand = _autoconv_full(h_rand)
    score_rand = _score(h_rand, ac_rand)
    
    if score_rand < best_score:
        best_score = score_rand
        best_h = h_rand.copy()
        best_ac = ac_rand.copy()
        
    # Candidate 3: All ones (baseline)
    h_ones = np.ones(N)
    ac_ones = _autoconv_full(h_ones)
    score_ones = _score(h_ones, ac_ones)
    
    if score_ones < best_score:
        best_score = score_ones
        best_h = h_ones.copy()
        best_ac = ac_ones.copy()

    # Run local search on the best candidate
    if time.time() - start_time < deadline - 5:
        best_h, best_ac, best_score = _local_search_fast(best_h, time_limit=deadline - 10)
        
    # Try local search on other candidates if time permits
    if time.time() - start_time < deadline - 10:
        # Local search on Thue-Morse if it wasn't the best
        if not np.array_equal(best_h, h_tm):
            h_tm_opt, ac_tm_opt, score_tm_opt = _local_search_fast(h_tm, time_limit=deadline - 15)
            if score_tm_opt < best_score:
                best_h = h_tm_opt
                best_score = score_tm_opt
                
    if time.time() - start_time < deadline - 10:
        # Local search on random if it wasn't the best
        if not np.array_equal(best_h, h_rand):
            h_rand_opt, ac_rand_opt, score_rand_opt = _local_search_fast(h_rand, time_limit=deadline - 15)
            if score_rand_opt < best_score:
                best_h = h_rand_opt
                best_score = score_rand_opt

    return best_h

# EVOLVE-BLOCK-END
