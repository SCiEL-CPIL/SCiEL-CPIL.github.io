import numpy as np
import itertools

def construct(q, n, radius):
    # Parameters for this task are fixed: q=3, n=8, radius=1
    # But we write it generally for the interface.
    
    # 1. Generate initial solution using Tetracode product
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all 9x9 = 81 base combinations? No, the previous solution was 729.
    # The previous solution was Tetracode (9) x AllWords(3^4=81) = 729.
    # Let's stick to the 729 solution as the seed for local search.
    
    all_tails = list(itertools.product(range(3), repeat=n - 4))
    initial_codewords = []
    for h in tetracode:
        for tail in all_tails:
            initial_codewords.append(h + list(tail))
            
    # Convert to integers for efficient processing
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
            
    total_words = q ** n
    codewords_int = [to_int(w) for w in initial_codewords]
    codewords_set = set(codewords_int)
    
    # Precompute neighbors (ball of radius 1) for all words in the space
    # neighbors[i] is a list of integers
    neighbors = []
    for i in range(total_words):
        word = to_word(i)
        ball = [i]
        for pos in range(n):
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (word[pos] + delta) % q
                ball.append(to_int(new_word))
        neighbors.append(ball)
        
    # Convert neighbors to numpy array for faster slicing? 
    # neighbors_arr = np.array(neighbors) # Shape (6561, 17)
    # Using lists is okay, but numpy is faster for cov updates.
    
    cov = np.zeros(total_words, dtype=np.int32)
    
    # Initialize coverage
    for c in codewords_int:
        for w in neighbors[c]:
            cov[w] += 1
            
    # Local Search
    improved = True
    while improved:
        improved = False
        
        # Try removals
        removal_found = False
        for c in list(codewords_set):
            ball = neighbors[c]
            # Check if all words in ball have cov > 1
            if np.all(cov[ball] > 1):
                # Remove c
                codewords_set.remove(c)
                for w in ball:
                    cov[w] -= 1
                improved = True
                removal_found = True
                break # Restart loop to re-evaluate other removals
        
        if removal_found:
            continue
            
        # Try swaps
        swap_found = False
        for c in list(codewords_set):
            ball_c = neighbors[c]
            # Find words that would become uncovered
            # These are w in ball_c where cov[w] == 1
            uncovered_if_removed = [w for w in ball_c if cov[w] == 1]
            
            if not uncovered_if_removed:
                # This case should have been caught by removal, but just in case
                codewords_set.remove(c)
                for w in ball_c:
                    cov[w] -= 1
                improved = True
                swap_found = True
                break

            # Find a replacement c'
            # c' must cover all words in uncovered_if_removed
            # i.e., c' must be in intersection of neighbors[w] for all w in uncovered_if_removed
            
            if len(uncovered_if_removed) == 1:
                w0 = uncovered_if_removed[0]
                candidates = neighbors[w0]
            else:
                # Intersect neighbor sets
                candidates = set(neighbors[uncovered_if_removed[0]])
                for w in uncovered_if_removed[1:]:
                    candidates.intersection_update(neighbors[w])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                candidates = list(candidates)
            
            # Pick a candidate not in codewords_set
            replacement = None
            for c_prime in candidates:
                if c_prime not in codewords_set:
                    replacement = c_prime
                    break
            
            if replacement is not None:
                # Perform swap
                # Remove c
                for w in ball_c:
                    cov[w] -= 1
                codewords_set.remove(c)
                
                # Add replacement
                ball_rep = neighbors[replacement]
                for w in ball_rep:
                    cov[w] += 1
                codewords_set.add(replacement)
                
                improved = True
                swap_found = True
                break
        
        if swap_found:
            continue

    # Convert back to list of lists
    result = [to_word(c) for c in codewords_set]
    return result
