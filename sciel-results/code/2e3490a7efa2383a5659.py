# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct(q, n, radius):
    """
    Optimized greedy set-cover algorithm with high-volume random restarts.
    Designed for small spaces (q^n < 10000) where bitmask operations are efficient.
    Includes a post-processing reduction phase to remove redundant codewords.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words > 5000:
        # Fallback for larger spaces, though this task is small
        return _construct_greedy_simple(q, n, radius)
        
    return _construct_greedy_optimized(q, n, radius)

def _construct_greedy_optimized(q, n, radius):
    """
    Optimized greedy set-cover using precomputed bitmask balls.
    Uses vectorized-like list comprehensions for speed.
    Includes a reduction phase and a local search phase.
    """
    total_words = q ** n
    
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
        
    # Precompute ball masks
    ball_masks = [0] * total_words
    
    if radius == 1:
        for i in range(total_words):
            mask = 1 << i
            for p in range(n):
                step = q_powers[p]
                digit = (i // step) % q
                for v in range(q):
                    if v == digit:
                        continue
                    j = i + (v - digit) * step
                    mask |= 1 << j
            ball_masks[i] = mask
    else:
        # Generic fallback for radius > 1
        def idx_to_word(idx):
            word = [0] * n
            for k in range(n-1, -1, -1):
                word[k] = idx % q
                idx //= q
            return word
        
        for i in range(total_words):
            mask = 1 << i
            center = idx_to_word(i)
            if radius == 2:
                # Distance 1
                for p in range(n):
                    step = q_powers[p]
                    digit = (i // step) % q
                    for v in range(q):
                        if v == digit: continue
                        j = i + (v - digit) * step
                        mask |= 1 << j
                # Distance 2
                for p1 in range(n):
                    for p2 in range(p1+1, n):
                        step1 = q_powers[p1]
                        step2 = q_powers[p2]
                        d1 = (i // step1) % q
                        d2 = (i // step2) % q
                        for v1 in range(q):
                            if v1 == d1: continue
                            for v2 in range(q):
                                if v2 == d2: continue
                                j = i + (v1 - d1) * step1 + (v2 - d2) * step2
                                mask |= 1 << j
            ball_masks[i] = mask

    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = 110.0
    restart_count = 0
    rng = np.random.RandomState(42)
    
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for k in range(n-1, -1, -1):
            word[k] = temp % q
            temp //= q
        return word

    # Precompute list of all indices for shuffling
    all_indices = list(range(total_words))

    while True:
        # Shuffle indices for random tie-breaking / order
        perm = rng.permutation(all_indices).tolist()
        
        uncovered = all_mask
        codeword_indices = []
        
        while uncovered != 0:
            # Calculate scores for all candidates
            # Using list comprehension is faster than a manual loop in Python
            scores = [ (ball_masks[idx] & uncovered).bit_count() for idx in perm ]
            
            max_score = max(scores)
            
            if max_score == 0:
                break
                
            # Find all candidates with max score
            # This list might be small, so linear scan is fine
            best_candidates = [idx for idx, score in zip(perm, scores) if score == max_score]
            best_idx = rng.choice(best_candidates)
            
            codeword_indices.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
        # Reduction Phase
        if len(codeword_indices) > 0:
            # Count coverage
            coverage_count = [0] * total_words
            for idx in codeword_indices:
                mask = ball_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    w_idx = lsb.bit_length() - 1
                    coverage_count[w_idx] += 1
                    m ^= lsb
            
            active_indices = list(codeword_indices)
            
            # Remove redundant codewords
            # Iterate through active indices
            # To be efficient, we can just check and remove.
            for i in range(len(active_indices)):
                idx = active_indices[i]
                mask = ball_masks[idx]
                m = mask
                is_redundant = True
                while m:
                    lsb = m & -m
                    w_idx = lsb.bit_length() - 1
                    if coverage_count[w_idx] == 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    m = mask
                    while m:
                        lsb = m & -m
                        w_idx = lsb.bit_length() - 1
                        coverage_count[w_idx] -= 1
                        m ^= lsb
                    active_indices[i] = -1
            
            final_indices = [idx for idx in active_indices if idx != -1]
            
            # Local Search: Try to swap 2 for 1? Or just simple 1-swap to enable removal?
            # Given time constraints, stick to greedy+reduction for now, but maybe add a simple 1-swap?
            # Let's try a simple improvement: 
            # If we have space in time, try to find a c' not in set that allows removal of c.
            # This is expensive. Let's rely on the speed of the greedy to do more restarts.
            
            codewords = [idx_to_word(idx) for idx in final_indices]
            
            if len(codewords) < best_len:
                best_len = len(codewords)
                best_codewords = codewords
                
                if best_len == 73:
                    break
                    
        restart_count += 1
        
        if time.time() - start_time > max_time:
            break
            
        if restart_count > 20000:
            break
            
    return best_codewords

def _construct_greedy_simple(q, n, radius):
    """Fallback simple greedy for larger spaces."""
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    ball_masks = [0] * total_words
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 1 << idx
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        ball_masks[idx] = mask
        
    uncovered = (1 << total_words) - 1
    codewords = []
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    return codewords
# EVOLVE-BLOCK-END
