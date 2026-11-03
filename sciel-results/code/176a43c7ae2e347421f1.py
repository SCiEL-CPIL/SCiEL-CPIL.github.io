import itertools
import random
import time

def construct(q, n, radius):
    # Parameters for the specific task
    # q=3, n=6, radius=1
    # Total words: q**n = 729
    
    total_words = q ** n
    # Map word tuple to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute spheres (bitmasks) for each word
    # A sphere of radius 1 contains the word itself and all words at distance 1.
    spheres = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 1 << i
        # Add self
        # Add neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                j = word_to_idx(new_w)
                mask |= (1 << j)
        spheres[i] = mask

    # Initial best solution: the direct product construction from the seed
    # This gives 81 codewords.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    best_codewords = initial_codewords
    best_size = len(best_codewords)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Randomized Greedy Set Cover
    # We run multiple restarts.
    # In each restart, we greedily pick codewords.
    # To diversify, we pick randomly from the top-k candidates with max gain.
    
    full_mask = (1 << total_words) - 1
    
    def popcount(x):
        return bin(x).count('1')

    for restart in range(500): # Run up to 500 restarts
        if time.time() > deadline:
            break
            
        uncovered = full_mask
        current_codewords = []
        
        while uncovered != 0:
            # Find best candidate
            max_gain = -1
            best_candidates = []
            
            for i in range(total_words):
                gain = popcount(uncovered & spheres[i])
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain:
                    best_candidates.append(i)
            
            if max_gain == 0:
                # Should not happen if uncovered is not 0 and spheres cover everything
                break
                
            # Pick randomly from best candidates to explore different solutions
            # Bias towards picking the first one sometimes to keep it deterministic-ish?
            # No, pure random is better for diversity.
            chosen_idx = random.choice(best_candidates)
            current_codewords.append(chosen_idx)
            uncovered &= ~spheres[chosen_idx]
            
        # Check if we improved
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            # Convert indices back to words
            best_codewords = [all_words[i] for i in current_codewords]
            
            # Try to remove redundant codewords
            # Check if removing any codeword still leaves full coverage
            improved = True
            while improved:
                improved = False
                for i in range(len(best_codewords)):
                    # Temporarily remove best_codewords[i]
                    temp_codewords = best_codewords[:i] + best_codewords[i+1:]
                    # Check coverage
                    covered = 0
                    for cw in temp_codewords:
                        idx = word_to_idx(cw)
                        covered |= spheres[idx]
                    if covered == full_mask:
                        best_codewords = temp_codewords
                        best_size = len(best_codewords)
                        improved = True
                        break # Restart the loop as indices changed
        
        # If we reached the theoretical lower bound or a very good number, we can stop early?
        # Lower bound is 71. If we find 71 or 72, it's excellent.
        if best_size <= 72:
             # We can break early if we find a very good solution, 
             # but let's give it a bit more time to see if we can hit 71.
             # Actually, 73 is the known best. 71 is lower bound.
             # If we find 73, it's a huge improvement over 81.
             # Let's continue searching for a bit longer to try for 72/71.
             pass

    return best_codewords
