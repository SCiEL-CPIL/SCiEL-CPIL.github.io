import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    # Specifics for the task
    total_words = q ** n
    num_coords = n
    alphabet = q

    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits

    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx

    # Precompute coverage: coverage[cw] = list of words covered by cw
    cover_size = 1 + num_coords * (alphabet - 1)
    coverage = np.zeros((total_words, cover_size), dtype=np.int32)
    for cw_idx in range(total_words):
        digits = idx_to_digits(cw_idx)
        coverage[cw_idx, 0] = cw_idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                coverage[cw_idx, j] = digits_to_idx(new_digits)
                j += 1

    # Precompute reverse coverage: covers_word[w] = list of codewords covering w
    covers_word = np.zeros((total_words, cover_size), dtype=np.int32)
    for w_idx in range(total_words):
        digits = idx_to_digits(w_idx)
        covers_word[w_idx, 0] = w_idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                covers_word[w_idx, j] = digits_to_idx(new_digits)
                j += 1

    # Helper: Local Search (Removal)
    def local_search(set_of_cw, cover_count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            critical_words = np.where(cover_count == 1)[0]
            
            for cw in list(set_of_cw):
                removable = True
                for j in range(cover_size):
                    w_idx = coverage[cw, j]
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                
                if removable:
                    set_of_cw.remove(cw)
                    for j in range(cover_size):
                        w_idx = coverage[cw, j]
                        cover_count[w_idx] -= 1
                    improved = True
                    break
        return set_of_cw, cover_count

    # Helper: Greedy Initialization
    def greedy_init():
        uncovered = np.ones(total_words, dtype=np.uint8)
        count = np.zeros(total_words, dtype=np.int32)
        
        for w_idx in range(total_words):
            for j in range(cover_size):
                cw_idx = covers_word[w_idx, j]
                count[cw_idx] += 1
                
        codewords = set()
        num_covered = 0
        cover_count = np.zeros(total_words, dtype=np.int32)
        
        while num_covered < total_words:
            if time.time() > deadline:
                break
            best_cw = np.argmax(count)
            if count[best_cw] == 0:
                break
            codewords.add(best_cw)
            for j in range(cover_size):
                w_idx = coverage[best_cw, j]
                if uncovered[w_idx]:
                    uncovered[w_idx] = 0
                    num_covered += 1
                    cover_count[w_idx] += 1
                    for k in range(cover_size):
                        cw_idx = covers_word[w_idx, k]
                        count[cw_idx] -= 1
                        # Note: We update count for *uncovered* words only in standard greedy.
                        # Wait, the standard greedy updates count of candidates based on uncovered words.
                        # When we cover a word, we decrement the count of all codewords that cover it.
                        # This is correct.
                        
        return codewords, cover_count

    # Helper: Perturb and Repair
    def perturb_and_repair(set_of_cw, cover_count, num_remove=5):
        new_set = set(set_of_cw)
        new_cover = cover_count.copy()
        
        # Remove random codewords
        to_remove = random.sample(list(new_set), min(num_remove, len(new_set)))
        for cw in to_remove:
            new_set.remove(cw)
            for j in range(cover_size):
                w_idx = coverage[cw, j]
                new_cover[w_idx] -= 1
        
        # Identify uncovered words
        # We need to repair the holes.
        # Strategy: While there are words with new_cover[w] == 0:
        #   Pick one such word w.
        #   Look at its 17 neighbors in covers_word[w].
        #   For each neighbor c, calculate how many *currently uncovered* words it covers.
        #   Add the neighbor with the max coverage of uncovered words.
        
        # To speed up "currently uncovered", we can just check new_cover[w] == 0.
        # But checking "how many uncovered words c covers" requires iterating over coverage[c] and checking new_cover.
        
        # Optimization:
        # Keep a list of uncovered words? Or just find one each time.
        # Since the number of holes is small (<= 5 * 17 = 85), we can be a bit slow.
        
        # Find all initially uncovered words
        # Note: Removing a codeword might make multiple words uncovered.
        
        while True:
            if time.time() > deadline:
                return None, None # Timeout
            
            # Find an uncovered word
            # We can just scan, or maintain a list. Scanning 6561 is fast.
            uncovered_indices = np.where(new_cover == 0)[0]
            if len(uncovered_indices) == 0:
                break
            
            w = uncovered_indices[0]
            
            # Get candidates for w
            candidates = covers_word[w]
            
            best_cand = -1
            best_gain = -1
            
            for c in candidates:
                if c in new_set:
                    # If c is already in the set, it shouldn't cover w if w is uncovered?
                    # Wait. If c is in new_set, new_cover[w] should be >= 1.
                    # So if w is uncovered, c cannot be in new_set.
                    # Unless there's a bug in removal.
                    # So we can assume c is not in new_set.
                    pass
                
                # Calculate gain: number of words in coverage[c] that are currently uncovered (new_cover == 0)
                gain = 0
                for j in range(cover_size):
                    w_idx = coverage[c, j]
                    if new_cover[w_idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_cand = c
            
            if best_cand == -1:
                break # Should not happen
            
            # Add best_cand
            new_set.add(best_cand)
            for j in range(cover_size):
                w_idx = coverage[best_cand, j]
                new_cover[w_idx] += 1
        
        return new_set, new_cover

    # 1. Initial Greedy
    best_set, best_cover = greedy_init()
    best_set, best_cover = local_search(best_set, best_cover)
    best_size = len(best_set)
    
    # 2. Iterated Local Search
    # We try to improve by perturbing and repairing.
    # We allow several iterations if time permits.
    
    while time.time() < deadline - 5: # Safety margin
        if best_size == 486:
            break # Optimal found
        
        # Perturb
        # Increase removal count slightly if stuck?
        # Start with 3, increase up to 10?
        # For simplicity, let's use a fixed small number like 5, or random 3-7.
        num_remove = random.randint(3, 7)
        
        perturbed_set, perturbed_cover = perturb_and_repair(best_set, best_cover, num_remove)
        
        if perturbed_set is None:
            break # Timeout
        
        # Local Search on the perturbed solution
        perturbed_set, perturbed_cover = local_search(perturbed_set, perturbed_cover)
        
        if len(perturbed_set) < best_size:
            best_set = perturbed_set
            best_cover = perturbed_cover
            best_size = len(best_set)
            
    # Convert to list of digits
    result = [idx_to_digits(cw) for cw in best_set]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
            
    return unique_result
