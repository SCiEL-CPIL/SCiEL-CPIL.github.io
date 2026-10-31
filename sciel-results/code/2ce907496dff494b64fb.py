import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # Compute all pairwise sums and differences
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    if len(unique_sums) == 0:
        return 0.0
        
    ratio = len(unique_diffs) / float(len(unique_sums))
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _fft_score(u):
    """
    Fast scoring using FFT to count unique sums and differences.
    This is O(M log M) where M is the span, allowing evaluation of larger sets.
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    
    # For sums: U+U is in [0, 2M].
    # For diffs: U-U is in [-M, M].
    
    # We can compute the autocorrelation of the indicator vector to get sum counts?
    # No, autocorrelation gives difference counts.
    # Let f be indicator of U.
    # FFT(f) * FFT(f) -> inverse FFT gives correlation (differences).
    # To get sums, we can shift U by M so it's in [M, 2M], then sum is in [2M, 4M].
    # Or simply compute sumset via convolution of f with f.
    
    # Size of FFT needs to be at least 2*max(U) + 1 for sums if we don't shift?
    # Actually, sum of two numbers in [0, M] is in [0, 2M].
    # Diff of two numbers in [0, M] is in [-M, M].
    
    # Let's compute both using FFT.
    # N_fft should be power of 2 >= 2*M + 1.
    size = 2 * m + 1
    n_fft = 1
    while n_fft < size:
        n_fft *= 2
        
    # Indicator vector
    f = np.zeros(n_fft, dtype=np.float64)
    f[u_arr] = 1.0
    
    # FFT
    F = np.fft.fft(f)
    
    # Sums: Convolution of f with f.
    # Inverse FFT of F * F gives circular convolution.
    # Since we padded to n_fft >= 2*size, linear convolution is embedded in the first 'size' elements.
    conv = np.fft.ifft(F * F).real
    # The sumset U+U corresponds to indices 0..2M.
    # conv[i] is the number of pairs (a,b) in U such that a+b=i.
    # We want the number of unique sums, i.e., count of i where conv[i] > 0.
    # Note: conv might have small negative values due to floating point errors.
    unique_sums = np.sum(conv[:size] > 0.5)
    
    # Diffs: Autocorrelation.
    # Inverse FFT of F * conj(F) gives autocorrelation.
    # autocorr[k] = sum f[i] f[i+k].
    # This corresponds to differences.
    # The difference d = a - b is counted in autocorr at index (d + M) or similar shift?
    # Standard autocorrelation c[k] = sum_x f[x] f[x+k].
    # If a - b = d, then a = b + d.
    # c[d] = sum_b f[b+d] f[b].
    # So c[d] counts pairs with difference d.
    # The range of d is [-M, M].
    # In the FFT output, negative indices wrap around.
    # c[d] for d>=0 is at index d.
    # c[d] for d<0 is at index d + n_fft.
    
    autocorr = np.fft.ifft(F * np.conj(F)).real
    
    # Extract unique diffs
    # d ranges from -M to M.
    # Indices in autocorr array:
    # d = 0 -> index 0
    # d > 0 -> index d
    # d < 0 -> index d + n_fft
    
    # We need to check which differences d in [-M, M] have count > 0.
    # We can construct a boolean mask for the valid indices.
    
    # Positive diffs: 1..M -> indices 1..M
    # Zero diff: 0 -> index 0
    # Negative diffs: -1..-M -> indices n_fft-1 .. n_fft-M
    
    mask = np.zeros(n_fft, dtype=bool)
    mask[0] = True
    mask[1:m+1] = True
    mask[n_fft-m:n_fft] = True
    
    unique_diffs = np.sum(autocorr[mask] > 0.5)
    
    ratio = unique_diffs / float(unique_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _random_initial_set(n, max_val):
    """
    Generate a random set of size n containing 0.
    Uses a mix of random and structured initialization.
    """
    u = {0}
    # Add random points
    for _ in range(n - 1):
        val = random.randint(1, max_val)
        u.add(val)
    return sorted(u)

def _local_search(u, timeout=5.0, use_fft=True):
    """
    Improve a set using local search.
    Moves: Replace an element with a nearby one that improves score.
    Uses FFT for faster evaluation if enabled.
    """
    u = sorted(set(u))
    if len(u) < 2:
        return [0, 1], 0.0
    
    score_func = _fft_score if use_fft else _score_from_set
    current_score = score_func(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    
    n = len(u)
    max_val = max(u)
    
    # We will try to improve by swapping elements.
    # Focus on elements that might be causing "wasted" space (large span) 
    # or not contributing enough to unique sums/diffs.
    
    # Adaptive search range
    # If span is large, we might want to compress.
    # If ratio is low, we might want to break collisions.
    
    for step in range(10000):
        if time.time() > start_time + timeout or time.time() > _DEADLINE - 5.0:
            break
            
        if n < 2:
            break
        
        # Pick a random element to perturb (excluding 0)
        idx = random.randint(1, n - 1)
        old_val = u[idx]
        
        # Determine search range
        # Heuristic: larger span -> larger search range to allow compression
        # Smaller n -> smaller search range for fine tuning
        search_range = max(1, int(max_val * 0.05) + 10)
        
        # Generate candidates
        # We try a mix of small steps (fine tuning) and large steps (restructuring)
        candidates = []
        
        # Small steps
        for delta in range(-search_range, search_range + 1, max(1, search_range // 10)):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 0:
                continue
            if new_val in u:
                continue
            candidates.append(new_val)
            
        # Large jumps (random restarts within local neighborhood)
        for _ in range(3):
            new_val = random.randint(1, max_val + 100)
            if new_val not in u:
                candidates.append(new_val)
                
        if not candidates:
            continue
            
        random.shuffle(candidates)
        
        improved = False
        # Limit tries per step to keep pace
        for new_val in candidates[:10]: 
            new_u = u[:]
            new_u[idx] = new_val
            new_u = sorted(set(new_u))
            
            if len(new_u) != n:
                continue
                
            s = score_func(new_u)
            if s > current_score:
                u = new_u
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_u = u[:]
                improved = True
                # Update max_val if it changed significantly
                new_max = max(u)
                if new_max > max_val:
                    max_val = new_max
                break
                
        if not improved:
            # Try adding an element if size is small and span allows
            if n < 30 and random.random() < 0.05:
                # Add a random element in a gap
                if n > 1:
                    idx_add = random.randint(0, n-2)
                    low = u[idx_add]
                    high = u[idx_add+1]
                    if high - low > 1:
                        new_val = random.randint(low+1, high-1)
                        if new_val not in u:
                            test_u = sorted(u + [new_val])
                            s = score_func(test_u)
                            if s > current_score:
                                u = test_u
                                current_score = s
                                if current_score > best_score:
                                    best_score = current_score
                                    best_u = u[:]
                                n += 1
            # Try removing an element if size is large
            elif n > 15 and random.random() < 0.05:
                idx_rm = random.randint(1, n - 1)
                test_u = u[:]
                test_u.pop(idx_rm)
                s = score_func(test_u)
                if s > current_score:
                    u = test_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    n -= 1
                    max_val = max(u) if u else 1
                    
    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = 0.0
    
    # Strategy:
    # 1. Use FFT-based scoring to allow exploring larger sets (n up to ~50-100) 
    #    and larger spans without timing out.
    # 2. Initialize with random sets of various sizes and spans.
    # 3. Perform local search to optimize the ratio.
    # 4. The key insight is that "Golomb" (max diff) is not always optimal.
    #    Sometimes allowing some diff collisions while reducing span or sum collisions yields a better ratio.
    
    # Try a range of sizes
    sizes_to_try = [10, 20, 30, 40, 50, 60, 80, 100]
    
    # Sort by time budget allocation
    # Smaller sets are faster, larger sets are slower but might have better potential.
    # We'll iterate and allocate remaining time.
    
    for n in sizes_to_try:
        if time.time() > _DEADLINE - 15.0:
            break
            
        # Number of restarts for this size
        # Larger n -> fewer restarts
        restarts = max(1, 20 - n // 5)
        
        for r in range(restarts):
            if time.time() > _DEADLINE - 10.0:
                break
                
            # Initialize
            # Span heuristic: for Sidon-like sets, span ~ n^2. 
            # But we want to explore beyond that.
            # Let's try spans from n^1.5 to n^2.5
            min_span = int(n ** 1.5)
            max_span = int(n ** 2.5)
            
            # Ensure max_span is not too huge for FFT (memory/time)
            # n_fft ~ 2 * max_span.
            # If max_span = 100^2.5 = 100000, n_fft = 200000. FFT is fast.
            # If max_span = 1000000, n_fft = 2000000. Still okay.
            
            init_max_val = random.randint(min_span, max_span)
            
            u_init = _random_initial_set(n, init_max_val)
            
            # Optimize
            u_opt, s_opt = _local_search(u_init, timeout=5.0, use_fft=True)
            
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
                
    # Final polish with the best found
    if time.time() < _DEADLINE - 5.0:
        # Use exact scoring for final verification and potentially fine-tuning
        # If FFT gave a good candidate, exact score might be slightly different due to FP errors in FFT?
        # Actually, FFT counts are integers theoretically, but FP noise can cause 0.9999 or 1.0001.
        # Our threshold 0.5 handles this.
        # However, for the final return, we should ensure the set is valid.
        
        # One more local search pass with exact scoring if time permits
        # and if the set is small enough for exact scoring to be fast.
        if len(best_u) < 50:
            u_final, s_final = _local_search(best_u, timeout=10.0, use_fft=False)
            if s_final > best_s:
                best_s = s_final
                best_u = u_final[:]
        else:
            # For larger sets, stick with FFT optimized version
            # But verify score with exact method to be safe?
            # Exact method is O(n^2). For n=100, 10000 ops. Very fast.
            s_exact = _score_from_set(best_u)
            if s_exact > best_s:
                best_s = s_exact
            # We don't re-optimize with exact if it's too slow, but n=100 is fine.
            if time.time() < _DEADLINE - 2.0:
                 u_final, s_final = _local_search(best_u, timeout=5.0, use_fft=False)
                 if s_final > best_s:
                     best_s = s_final
                     best_u = u_final[:]

    # Sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END
