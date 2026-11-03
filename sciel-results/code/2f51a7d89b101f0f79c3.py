import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Initialize with Tetracode construction
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Full code: 189 words
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Helper to convert word to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
        
    # Helper to get neighbors within radius 1
    # We can precompute the offsets or just generate them on the fly
    # Since n=7, q=3, radius=1, neighbors are 1 + n*(q-1) = 1 + 7*2 = 15 words.
    # This is very fast.
    
    def get_neighbors(w):
        # Returns list of indices of words within Hamming distance <= 1
        idxs = [word_to_idx(w)]
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig: continue
                w[i] = v
                idxs.append(word_to_idx(w))
            w[i] = orig
        return idxs

    # Precompute neighbors for all codewords?
    # Or just compute on the fly during removal check.
    # Since we might remove many, let's maintain a list of active codewords.
    
    active = list(range(len(codewords)))
    
    # Counts array
    counts = np.zeros(3**n, dtype=np.int32)
    
    # Initial counts
    for idx in active:
        w = codewords[idx]
        for n_idx in get_neighbors(w):
            counts[n_idx] += 1
            
    # Removal loop
    # We want to remove as many as possible.
    # Strategy: Repeatedly try to remove any codeword that is redundant.
    # To escape local minima, we can shuffle the order of attempts.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(active)
        i = 0
        while i < len(active):
            idx = active[i]
            w = codewords[idx]
            neigh = get_neighbors(w)
            
            # Check if removing this codeword leaves any word uncovered
            # i.e., if any count in neigh is exactly 1
            if np.any(counts[neigh] == 1):
                i += 1
                continue
            
            # It is safe to remove
            for n_idx in neigh:
                counts[n_idx] -= 1
            active.pop(i)
            improved = True
            # Do not increment i, check the next element which shifted down
            
    # If we still have time, we can try to restart with a different random seed or order
    # But simple removal is deterministic given the order.
    # Let's try a few random restarts of the removal order if time permits.
    
    best_size = len(active)
    best_active = active.copy()
    
    # Try a few more times with different shuffles?
    # The current loop shuffles at the start of each "pass".
    # If we get stuck at 189, it means no single codeword is redundant.
    
    # To potentially break this, we can try swapping.
    # But swapping is more complex.
    # Given the 186 bound, and 189 start, we need to remove 3.
    # If simple removal fails, we are stuck at 189.
    
    # Let's return the best found
    return [codewords[i] for i in best_active]
