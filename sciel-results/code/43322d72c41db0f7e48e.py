"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Strategy: Use FFT to efficiently compute sumset and difference set sizes.
Perform a greedy hill-climbing search to construct the set, adding elements
that maximize the score improvement.
"""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _score_fft(u):
    """
    Compute the score for a set U using FFT for fast convolution.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    if not u or 0 not in u:
        return 0.0
    
    u = sorted(list(set(u)))
    n = len(u)
    if n < 2:
        return 0.0
        
    max_u = max(u)
    if max_u == 0:
        return 0.0
        
    # Range for sums: 0 to 2*max_u
    # Range for differences: -max_u to max_u
    # We can compute both using a single convolution if we shift indices appropriately
    # Or just compute two convolutions. 
    # Let's use the property: |U+U| is number of non-zero coeffs of U*U.
    # |U-U| is number of non-zero coeffs of U*U_reversed.
    
    # Size of array for FFT
    # Max sum is 2*max_u. Length 2*max_u + 1.
    # Next power of 2 for FFT efficiency
    size = 2 * max_u + 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
        
    # Create indicator vector
    vec = np.zeros(fft_size, dtype=np.float64)
    for x in u:
        vec[x] = 1.0
        
    # FFT
    f_vec = np.fft.fft(vec)
    
    # Sumset: |U+U|
    sum_conv = np.fft.ifft(f_vec * f_vec).real
    sum_size = np.sum(sum_conv[:size] > 1e-9)
    
    # Difference set: |U-U|
    # U - U is equivalent to U + (-U)
    # If we reverse the vector, we get -U (shifted)
    # Let's just compute the autocorrelation or similar.
    # Actually, simpler: 
    # U-U contains values d such that there exist a,b in U with a-b=d.
    # This is the support of the convolution of U and U_flipped.
    # Let v be indicator of U.
    # Let w be indicator of -U.
    # If U is in [0, max_u], -U is in [-max_u, 0].
    # Shift -U by max_u to be in [0, max_u].
    # Then conv(v, w_shifted) gives counts for U + (-U + max_u) = (U-U) + max_u.
    # The number of non-zero coefficients is |U-U|.
    
    # Create w: indicator of -U shifted by max_u
    # -x + max_u for x in U.
    w_vec = np.zeros(fft_size, dtype=np.float64)
    for x in u:
        w_vec[max_u - x] = 1.0
        
    f_w = np.fft.fft(w_vec)
    diff_conv = np.fft.ifft(f_vec * f_w).real
    
    # The differences range from -max_u to max_u.
    # Shifted by max_u, they range from 0 to 2*max_u.
    # So we look at indices 0 to 2*max_u.
    diff_size = np.sum(diff_conv[:size] > 1e-9)
    
    if sum_size == 0:
        return 0.0
        
    log2_max = math.log2(2 * max_u + 1)
    if log2_max == 0:
        return 0.0
        
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
        
    score = 1.0 + math.log(ratio) / log2_max
    return score

def _greedy_construct(start_set, max_limit, time_limit):
    """
    Greedily add elements to the set to improve score.
    """
    U = list(start_set)
    U.sort()
    current_score = _score_fft(U)
    
    # We will try adding elements from 1 to max_limit
    # To make it faster, we can randomize the order or use a heuristic
    # Let's iterate through candidates in increasing order, but allow skipping if time is tight
    # Actually, a better heuristic: try to add elements that are "far" from existing ones to maximize diff set,
    # but not too far to keep max(U) small.
    
    # Let's do a simple hill climbing:
    # 1. Generate a list of candidate integers to add.
    # 2. For each candidate, calculate new score.
    # 3. Pick the best one.
    # 4. Repeat until no improvement or time limit.
    
    # To avoid O(N * MaxLimit) which is too slow, we will restrict candidates.
    # We'll pick a subset of candidates to evaluate in each round.
    
    while _remaining() > 5.0:
        best_gain = 0
        best_candidate = None
        
        # Generate candidates: integers not in U, up to max_limit
        # To speed up, we can sample candidates or use a heuristic range
        # Let's try all integers from 1 to max_limit in a random order? No, that's expensive.
        # Let's try integers that are "sparse" or follow a pattern.
        
        # Heuristic: Try adding the next integer in a geometric progression or random integers
        # Let's try a mix:
        # 1. Small integers (1 to 10)
        # 2. Integers that are powers of 2
        # 3. Random integers in [1, max_limit]
        
        candidates = set()
        for i in range(1, 20):
            if i not in U:
                candidates.add(i)
        
        # Powers of 2
        p = 2
        while p <= max_limit:
            if p not in U:
                candidates.add(p)
            p *= 2
            
        # Random samples
        for _ in range(50):
            r = np.random.randint(1, max_limit + 1)
            if r not in U:
                candidates.add(r)
                
        # Evaluate candidates
        for c in candidates:
            if _remaining() < 5.0:
                break
            new_U = U + [c]
            new_score = _score_fft(new_U)
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
                
        if best_candidate is None or best_gain <= 0:
            # No improvement found in this round of candidates
            # Maybe try a wider search or break
            # To avoid infinite loop, we break if no gain
            break
            
        U.append(best_candidate)
        U.sort()
        current_score += best_gain
        
    return U, current_score

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the score: 1 + log(|U-U|/|U+U|) / log(2*max(U)+1).
    """
    best_score = 0.0
    best_set = [0]
    
    # Try multiple random starts / strategies
    # Strategy 1: Start with small primes
    # Strategy 2: Start with powers of 2
    # Strategy 3: Random start
    
    max_limit = 10000  # Upper bound for elements
    
    # Run 3 different greedy searches
    for i in range(3):
        if _remaining() < 10.0:
            break
            
        if i == 0:
            # Start with small integers
            start = [0, 1, 2, 3, 5]
        elif i == 1:
            # Start with powers of 2
            start = [0, 1, 2, 4, 8, 16, 32]
        else:
            # Random start
            start = [0]
            for _ in range(5):
                start.append(np.random.randint(1, 100))
            start = list(set(start))
            start.sort()
            
        U, score = _greedy_construct(start, max_limit, 30.0)
        
        if score > best_score:
            best_score = score
            best_set = U
            
    # Final check: ensure 0 is in the set
    if 0 not in best_set:
        best_set.append(0)
        best_set.sort()
        
    return best_set

if __name__ == "__main__":
    U = construct()
    print(f"Set size: {len(U)}")
    print(f"Max element: {max(U)}")
    print(f"Score: {_score_fft(U)}")
    U = construct()
    print(f"Set size: {len(U)}")
    print(f"Max element: {max(U)}")
    print(f"Score: {_score_set(U)}")
