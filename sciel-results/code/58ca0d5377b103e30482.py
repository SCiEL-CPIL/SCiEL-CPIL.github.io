import time
import random
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0  # Safety margin

def _score_from_set(U):
    """
    Calculate the score for a given set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    Uses FFT for efficient computation.
    """
    if not U or len(U) < 2:
        return 0.0
    
    max_val = max(U)
    if max_val < 1:
        return 0.0
        
    # Determine FFT size
    # We need to compute convolution of U with U (sums) and U with -U (diffs)
    # Sums range from 0 to 2*max_val
    # Diffs range from -max_val to max_val
    # Total range needed is roughly 2*max_val + 1 for sums, 2*max_val + 1 for diffs.
    # We can use a single FFT of size >= 2*max_val + 1 for sums.
    # For diffs, we can use autocorrelation.
    
    L = max_val + 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        vec[u] = 1.0
        
    f = np.fft.fft(vec)
    
    # Sum set: Convolution of vec with vec
    conv_sum = np.fft.ifft(f * f).real
    # Sums are in indices 0 to 2*max_val
    sums_count = np.sum(conv_sum[:2*max_val+1] > 0.5)
    
    # Difference set: Autocorrelation of vec with vec
    # Autocorrelation C[k] = sum vec[i] vec[i-k]
    # This is equivalent to conv(vec, conj(vec)) but shifted.
    # Actually, np.fft.ifft(np.conj(f) * f) gives circular autocorrelation.
    # For linear autocorrelation, we need to unwrap or just count non-zeros in the full range.
    # The support of the difference set is the set of k such that autocorrelation is non-zero.
    conv_diff = np.fft.ifft(np.conj(f) * f).real
    # The differences are in range -max_val to max_val.
    # In the FFT output, index 0 is diff 0.
    # Indices 1..max_val are positive diffs.
    # Indices fft_len-max_val..fft_len-1 are negative diffs.
    # We just need to count how many k have conv_diff[k] > 0.5.
    # Since the set is symmetric, we can just sum the whole thing?
    # No, conv_diff is periodic. But we zero-padded, so it's linear.
    # The non-zero values will be at indices corresponding to the differences.
    diffs_count = np.sum(conv_diff > 0.5)
    
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 1:
        # If ratio <= 1, log is negative, score < 1.
        # We want to maximize score.
        pass
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _generate_golomb(n):
    """Generate a greedy Golomb ruler of size n."""
    U = [0]
    diffs = {0}
    current_max = 0
    for _ in range(n - 1):
        candidate = current_max + 1
        while True:
            valid = True
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
            if valid:
                U.append(candidate)
                for u in U[:-1]:
                    diffs.add(candidate - u)
                current_max = candidate
                break
            else:
                candidate += 1
    return U

def _local_search(U, time_limit):
    """
    Perform a greedy local search to improve the score.
    Tries to move each element to a better position.
    """
    U = sorted(U)
    current_score = _score_from_set(U)
    best_score = current_score
    best_set = U[:]
    
    start_time = time.time()
    n = len(U)
    
    # We will iterate through elements and try to improve them.
    # To avoid getting stuck, we can do multiple passes.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(1, n):  # Keep 0 fixed
            if time.time() > _DEADLINE - 5.0:
                return best_set, best_score
            
            old_val = U[i]
            best_new_val = old_val
            best_local_score = current_score
            
            # Try moving to nearby values
            # Range: from 1 to max(U) + 10
            # We want to explore smaller values to reduce M.
            max_search_range = max(U) + 5
            if max_search_range > 500: # Cap search range for speed
                max_search_range = 500
                
            # Generate candidate values
            # We prioritize values close to old_val and smaller values.
            candidates = set()
            for delta in range(-10, 11):
                v = old_val + delta
                if 1 <= v <= max_search_range:
                    candidates.add(v)
            
            # Also add some random candidates in [1, max_search_range]
            for _ in range(20):
                v = random.randint(1, max_search_range)
                candidates.add(v)
                
            # And specifically try to reduce max if this is the max element
            if U[i] == max(U):
                for v in range(1, old_val):
                    if random.random() < 0.1: # Sample some lower values
                        candidates.add(v)

            for v in candidates:
                if v == old_val:
                    continue
                if v in U:
                    continue
                
                # Try the move
                new_U = U[:]
                new_U[i] = v
                new_U.sort()
                
                # Quick check: if max increases significantly, skip? 
                # No, let the score decide.
                
                new_score = _score_from_set(new_U)
                
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_new_val = v
                    best_set_candidate = new_U[:]
            
            if best_new_val != old_val:
                U = best_set_candidate
                current_score = best_local_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = U[:]
                improved = True
    
    return best_set, best_score

def construct():
    global _DEADLINE
    best_score = 0.0
    best_set = [0, 1]
    
    random.seed(42)
    
    # Try small n values
    # Based on analysis, n=3 to 15 is the sweet spot.
    n_values = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    
    # Allocate time roughly evenly, but favor smaller n as they are faster to search
    # and likely to have higher scores.
    
    for n in n_values:
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Initialize with Greedy Golomb
        U_init = _generate_golomb(n)
        
        # Run local search
        # Give more time to smaller n? Or just a fixed chunk.
        # Let's say 5 seconds per n.
        time_limit = 5.0
        if n > 10:
            time_limit = 3.0
            
        U_opt, score_opt = _local_search(U_init, time_limit)
        
        if score_opt > best_score:
            best_score = score_opt
            best_set = U_opt[:]
            
    # Final check
    if best_score <= 0:
        best_set = [0, 1]
        
    return best_set

if __name__ == "__main__":
    # For local testing
    pass
