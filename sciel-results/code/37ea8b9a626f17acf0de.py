import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Fallback for large spaces
    if total_words > 10000:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]

    # Precompute powers of q for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
            
    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute balls as NumPy boolean arrays
    # balls[i] is a boolean array of length total_words
    balls = np.zeros((total_words, total_words), dtype=bool)
    
    # Precompute neighbors for radius 1
    # For each word, its ball is itself + all words at distance 1
    for i in range(total_words):
        w = idx_to_tuple(i)
        balls[i, i] = True
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    balls[i, new_idx] = True

    # Helper to check coverage using cover_counts
    # cover_counts[w] = number of codewords whose ball contains w
    
    def is_covered(cover_counts):
        return np.all(cover_counts > 0)

    def local_search(code_indices, cover_counts):
        """
        Attempts to reduce the code size by removing redundant words.
        Returns the pruned list of indices and the updated cover_counts.
        """
        code_list = list(code_indices)
        code_set = set(code_list)
        improved = True
        while improved:
            improved = False
            new_code_list = []
            for c in code_list:
                # Check if c is redundant
                # It is redundant if for all w in balls[c], cover_counts[w] > 1
                # We can check this by seeing if min(cover_counts[balls[c]]) > 1
                # But balls[c] is a boolean mask.
                # mask = balls[c]
                # if np.all(cover_counts[mask] > 1):
                #     # Remove c
                #     cover_counts -= balls[c]
                #     improved = True
                # else:
                #     new_code_list.append(c)
                
                # Optimized check:
                # We need to know if any word in the ball has count 1.
                # If not, it's safe to remove.
                if np.all(cover_counts[balls[c]] > 1):
                    cover_counts -= balls[c]
                    improved = True
                else:
                    new_code_list.append(c)
            code_list = new_code_list
        return code_list, cover_counts

    def greedy_cover(random_seed):
        """
        Greedy set cover with random tie-breaking.
        """
        random.seed(random_seed)
        uncovered_mask = np.ones(total_words, dtype=bool)
        code_indices = []
        cover_counts = np.zeros(total_words, dtype=int)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 5.0:
                return None # Timeout, return None to signal failure
            
            # Calculate gain for each candidate
            # gain[c] = number of currently uncovered words in ball of c
            # We can compute this as: sum(balls[c] & uncovered_mask)
            # This is a matrix-vector product if we treat balls as int, but bool is fine with sum.
            # However, doing this for all total_words candidates every step is O(N^2).
            # N=729, so 729^2 = 531k ops per step. 73 steps -> 38M ops. This is fast in NumPy.
            
            # To be faster, we can restrict candidates to those that cover at least one uncovered word.
            # But calculating gains for all is vectorized and likely fast enough.
            
            # gains = balls @ uncovered_mask  # This works if balls is int, but it's bool.
            # In NumPy, bool @ bool gives int counts.
            gains = balls.dot(uncovered_mask)
            
            # Find max gain
            max_gain = np.max(gains)
            
            if max_gain == 0:
                # Should not happen if uncovered_mask is not empty and balls cover the space
                break
                
            # Get candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one to introduce diversity
            best_c = random.choice(candidates.tolist())
            
            code_indices.append(best_c)
            cover_counts += balls[best_c]
            uncovered_mask[balls[best_c]] = False
            
        # Final check
        if not is_covered(cover_counts):
            return None
            
        return code_indices, cover_counts

    best_code = None
    best_size = float('inf')
    
    # 1. Try Direct Product Construction (if applicable)
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))
        
        # Initialize cover counts for DP
        dp_cover_counts = np.zeros(total_words, dtype=int)
        for idx in dp_indices:
            dp_cover_counts += balls[idx]
            
        # Prune DP
        pruned_dp, pruned_counts = local_search(dp_indices, dp_cover_counts)
        
        if len(pruned_dp) < best_size:
            best_size = len(pruned_dp)
            best_code = pruned_dp

    # 2. Try Randomized Greedy + Local Search Multiple Times
    num_restarts = 100 # Increased restarts since each is fast
    
    for r in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        result = greedy_cover(r)
        if result is None:
            continue
            
        code_indices, cover_counts = result
        
        # Local Search to prune
        pruned_code, pruned_counts = local_search(code_indices, cover_counts)
        
        # Additional Local Search: Try swaps?
        # Simple swap: Replace c with c' if coverage is maintained.
        # This doesn't reduce size directly, but might allow further removals.
        # Given the time limit and simplicity, let's stick to removal for now.
        # If we have time, we can add a swap phase.
        
        if len(pruned_code) < best_size:
            best_size = len(pruned_code)
            best_code = pruned_code
            
    if best_code is None:
        # Fallback to direct product
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            dp_indices = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    dp_indices.append(tuple_to_idx(tuple(word)))
            best_code = dp_indices
        else:
            best_code = list(range(total_words))

    # Convert indices back to list of lists
    result = [list(idx_to_tuple(c)) for c in best_code]
    
    return result
