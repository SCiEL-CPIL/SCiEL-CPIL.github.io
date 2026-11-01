import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Fallback to seed solution if something goes wrong or parameters don't match
    def seed_solution():
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    if n != 8 or q != 3 or radius != 1:
        return seed_solution()

    try:
        total_words = q ** n
        num_coords = n
        alphabet = q
        cover_size = 1 + num_coords * (alphabet - 1) # 1 + 8*2 = 17

        # Helper functions
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
        # coverage is a 2D array (total_words, cover_size)
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

        # 1. Greedy Initialization
        def greedy_init():
            uncovered = np.ones(total_words, dtype=np.uint8)
            count = np.zeros(total_words, dtype=np.int32)
            
            # Initialize counts: count[cw] = number of words covered by cw
            for w_idx in range(total_words):
                for j in range(cover_size):
                    cw_idx = covers_word[w_idx, j]
                    count[cw_idx] += 1
                    
            codewords = []
            num_covered = 0
            
            while num_covered < total_words:
                if time.time() > deadline:
                    break
                best_cw = np.argmax(count)
                if count[best_cw] == 0:
                    break
                codewords.append(best_cw)
                for j in range(cover_size):
                    w_idx = coverage[best_cw, j]
                    if uncovered[w_idx]:
                        uncovered[w_idx] = 0
                        num_covered += 1
                        for k in range(cover_size):
                            cw_idx = covers_word[w_idx, k]
                            count[cw_idx] -= 1
                            
            return codewords

        # 2. Local Search / Minimization with Repair
        def local_search(initial_cw):
            set_of_cw = set(initial_cw)
            cover_count = np.zeros(total_words, dtype=np.int32)
            
            # Initialize cover_count
            for cw in set_of_cw:
                for j in range(cover_size):
                    w_idx = coverage[cw, j]
                    cover_count[w_idx] += 1
            
            # Check validity
            if np.any(cover_count == 0):
                return None # Invalid start

            best_size = len(set_of_cw)
            
            # Iterative improvement
            # We use a loop that tries to delete. If stuck, it tries a kick.
            # To avoid infinite loops, we limit kicks.
            
            kick_count = 0
            max_kicks = 50
            
            while time.time() < deadline - 5:
                improved = False
                # Try to delete any codeword
                # A codeword is removable if min(cover_count[coverage[cw]]) > 1
                # We iterate through the set.
                
                # Optimization: Only check codewords that don't cover critical words (cover_count==1)
                # But finding which codewords cover critical words is also a scan.
                # Given size ~500-700, scanning all is fine.
                
                for cw in list(set_of_cw):
                    removable = True
                    for j in range(cover_size):
                        w_idx = coverage[cw, j]
                        if cover_count[w_idx] <= 1:
                            removable = False
                            break
                    
                    if removable:
                        # Remove cw
                        set_of_cw.remove(cw)
                        for j in range(cover_size):
                            w_idx = coverage[cw, j]
                            cover_count[w_idx] -= 1
                        improved = True
                        best_size = len(set_of_cw)
                        break # Restart scan after deletion
                
                if improved:
                    continue
                
                # If no deletion possible, try a kick
                if kick_count < max_kicks:
                    kick_count += 1
                    # Save state
                    orig_set = set(set_of_cw)
                    orig_cover = cover_count.copy()
                    
                    # Remove 1 random codeword
                    rand_cw = random.choice(list(orig_set))
                    new_set = set(orig_set)
                    new_set.remove(rand_cw)
                    new_cover = orig_cover.copy()
                    for j in range(cover_size):
                        w_idx = coverage[rand_cw, j]
                        new_cover[w_idx] -= 1
                        
                    # Repair: While there are uncovered words, add best candidate
                    # Uncovered words are where new_cover == 0
                    # We need to find the codeword that covers the most uncovered words.
                    
                    # Calculate gain for all codewords not in new_set?
                    # This is expensive if done from scratch.
                    # Instead, we can track the "uncovered" set and update counts.
                    
                    uncovered_mask = (new_cover == 0)
                    if not np.any(uncovered_mask):
                        # Still valid after removal? That means it was redundant but we missed it?
                        # Or maybe the deletion check was too strict?
                        # If valid, accept immediately.
                        set_of_cw = new_set
                        cover_count = new_cover
                        best_size = len(set_of_cw)
                        kick_count = 0 # Reset kicks on success
                        continue
                    
                    # Greedy repair
                    # We need count of how many *currently uncovered* words each candidate covers.
                    # Let's compute this once for all candidates.
                    # This is O(N * CoverSize) = 6561 * 17 ~ 111k ops. Fast.
                    
                    candidate_gains = np.zeros(total_words, dtype=np.int32)
                    # Only consider candidates not in new_set
                    # And only if they cover at least one uncovered word
                    
                    # Optimization: Only compute for candidates that are "neighbors" of uncovered words?
                    # Or just compute for all.
                    
                    for cw_idx in range(total_words):
                        if cw_idx in new_set:
                            continue
                        gain = 0
                        for j in range(cover_size):
                            w_idx = coverage[cw_idx, j]
                            if uncovered_mask[w_idx]:
                                gain += 1
                        if gain > 0:
                            candidate_gains[cw_idx] = gain
                    
                    # Add best candidates until covered
                    while np.any(new_cover == 0):
                        if time.time() > deadline - 2:
                            break
                        best_add = np.argmax(candidate_gains)
                        if candidate_gains[best_add] == 0:
                            break # Stuck
                        new_set.add(best_add)
                        for j in range(cover_size):
                            w_idx = coverage[best_add, j]
                            if new_cover[w_idx] == 0:
                                new_cover[w_idx] += 1
                                # Update gains for other candidates
                                # This is the expensive part.
                                # When a word becomes covered, all candidates covering it lose 1 gain.
                                for k in range(cover_size):
                                    cw_idx = covers_word[w_idx, k]
                                    if cw_idx in new_set:
                                        continue
                                    candidate_gains[cw_idx] -= 1
                            else:
                                new_cover[w_idx] += 1
                                # No change to gains for uncovered set
                                
                        # If we broke out of time, revert
                        if time.time() > deadline - 2:
                            break

                    # Check if repair was successful and resulted in smaller set
                    if len(new_set) < best_size:
                        set_of_cw = new_set
                        cover_count = new_cover
                        best_size = len(set_of_cw)
                        kick_count = 0 # Reset
                    else:
                        # Revert
                        set_of_cw = orig_set
                        cover_count = orig_cover

                else:
                    # Too many kicks without improvement, break
                    break
                    
            return list(set_of_cw)

        # Run multiple restarts
        best_solution = None
        best_size = float('inf')
        
        # First run: Standard Greedy
        cw1 = greedy_init()
        if cw1:
            res1 = local_search(cw1)
            if res1:
                best_solution = res1
                best_size = len(res1)
        
        # Second run: Randomized Greedy?
        # Or just run local_search again on the result of the first?
        # local_search already does kicks.
        
        # If time permits, try another restart with different random seed
        if time.time() < deadline - 20 and best_size > 486:
            random.seed(time.time())
            cw2 = greedy_init()
            if cw2:
                res2 = local_search(cw2)
                if res2 and len(res2) < best_size:
                    best_solution = res2
                    best_size = len(res2)

        if best_solution and best_size < float('inf'):
            # Verify validity
            cover_count = np.zeros(total_words, dtype=np.int32)
            for cw in best_solution:
                for j in range(cover_size):
                    w_idx = coverage[cw, j]
                    cover_count[w_idx] += 1
            if np.any(cover_count == 0):
                return seed_solution()
            return best_solution
        else:
            return seed_solution()

    except Exception:
        return seed_solution()
