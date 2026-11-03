import numpy as np
import itertools
import time

def construct(q, n, radius):
    # Parameters for the specific task
    # q=3, n=7, radius=1
    
    total_words = q ** n
    # Map word (tuple of n ints) to index (int)
    def to_index(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors
    # neighbors[i] is a list of indices of words within Hamming distance <= radius of word i
    # For radius 1, neighbors[i] includes i itself and words differing in 1 position.
    
    all_words = [to_word(i) for i in range(total_words)]
    neighbors = []
    for i in range(total_words):
        w = all_words[i]
        nb = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                nb.append(to_index(new_w))
        neighbors.append(nb)
    
    # Greedy algorithm
    def greedy_cover():
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        # Optimization: maintain a count of how many uncovered neighbors each word has
        # Initially, every word has 'len(neighbors[i])' uncovered neighbors
        # When we cover a word j, we decrement the count for all i such that j is in neighbors[i]
        
        # Let's use a simpler approach first: recompute gain each time.
        # With 2187 words, it's fast enough.
        
        while not covered.all():
            # Find the word with max uncovered neighbors
            best_idx = -1
            max_gain = -1
            
            # Vectorized gain calculation
            # uncovered is a boolean array
            # We want sum(uncovered[neighbors[i]]) for each i
            # neighbors is a list of lists. We can create a sparse matrix or just loop.
            
            # Loop is fine for 2187
            for i in range(total_words):
                # Count uncovered in neighbors[i]
                # neighbors[i] is a list of indices
                gain = np.sum(covered[neighbors[i]] == False)
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            # Add best_idx to code
            code.append(all_words[best_idx])
            
            # Mark neighbors of best_idx as covered
            covered[neighbors[best_idx]] = True
            
        return code

    # Run greedy a few times with slight variations if possible, or just once.
    # The greedy choice is deterministic if we pick the first max.
    # To get different results, we can randomize the tie-breaking.
    
    best_code = None
    start_time = time.time()
    
    # Try a few restarts with random tie-breaking
    for restart in range(10):
        if time.time() - start_time > 100: # Safety margin
            break
            
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        while not covered.all():
            gains = np.zeros(total_words)
            for i in range(total_words):
                gains[i] = np.sum(covered[neighbors[i]] == False)
                
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one of the best candidates
            best_idx = candidates[np.random.randint(len(candidates))]
            
            code.append(all_words[best_idx])
            covered[neighbors[best_idx]] = True
            
        if best_code is None or len(code) < len(best_code):
            best_code = code

    return best_code
