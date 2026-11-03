import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Fallback for non-standard parameters
    if q != 3 or n != 8 or radius != 1:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n

    def idx_to_word(idx):
        w = [0] * n
        for j in range(n - 1, -1, -1):
            w[j] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    # Precompute neighbors for all words
    # Shape: (total_words, 1 + n * (q - 1))
    neighbors = np.zeros((total_words, 1 + n * (q - 1)), dtype=np.int32)
    
    for i in range(total_words):
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                new_idx = word_to_idx(w)
                neigh.append(new_idx)
            w[pos] = orig
        neighbors[i] = neigh

    uncovered = np.ones(total_words, dtype=bool)
    code_indices = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Vectorized greedy step
        # counts[i] is the number of uncovered words within distance 1 of word i
        counts = uncovered[neighbors].sum(axis=1)
        best = np.argmax(counts)
        
        code_indices.append(best)
        uncovered[neighbors[best]] = False
        
    return [idx_to_word(i) for i in code_indices]
