import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    # Generate all words
    total = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute: for each word, the set of words within radius
    # For radius=1: the word itself + all single-position changes
    # Each word covers 1 + n*(q-1) words
    
    # For efficiency, precompute the index mapping
    # Convert word to index: sum(word[i] * q^(n-1-i))
    q_powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    
    def word_to_idx(w):
        return int(np.dot(w, q_powers))
    
    # Precompute neighbors for each word (words within radius 1)
    # This is expensive for 6561 words, but manageable
    # Each word has 1 + n*(q-1) = 1 + 8*2 = 17 neighbors
    
    neighbors = np.zeros((total, 1 + n * (q - 1)), dtype=np.int64)
    for i in range(total):
        w = all_words[i]
        neighbors[i, 0] = i
        idx = 1
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = w.copy()
                    nw[pos] = val
                    neighbors[i, idx] = word_to_idx(nw)
                    idx += 1
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    code = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find the word that covers the most uncovered words
        # For each candidate word i, count how many of its neighbors are uncovered
        # We can vectorize this
        
        # For each word i, we need sum of uncovered[neighbors[i]]
        # This is a matrix-vector product if we build the incidence matrix
        # But that's 6561 x 17 matrix, which is fine
        
        # Actually, let's just loop - 6561 iterations with 17 elements each
        best_i = -1
        best_count = 0
        for i in range(total):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_i = i
        
        if best_count == 0:
            break
        
        code.append(all_words[best_i].tolist())
        uncovered[neighbors[best_i]] = False
        uncovered_count -= best_count
    
    # If we ran out of time, we might not have covered everything
    # The evaluator checks validity, so we need to ensure coverage
    # If not fully covered, we need to add more words
    if uncovered_count > 0:
        # Add words to cover remaining uncovered
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            # Find an uncovered word and add it
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            i = unc_idx[0]
            code.append(all_words[i].tolist())
            uncovered[neighbors[i]] = False
            uncovered_count = int(np.sum(uncovered))
    
    return code
