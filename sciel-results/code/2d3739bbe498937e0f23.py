import time
import numpy as np
import itertools


def construct(q, n, radius):
    t0 = time.time()
    deadline = 118.0          # safety margin before the 150s kill
    total = q ** n
    
    # Precompute all words
    all_words = np.array(
        [list(t) for t in itertools.product(range(q), repeat=n)], dtype=np.int8
    )
    
    # Precompute ball masks as bitsets
    # mask[i] is an integer where bit j is 1 if word j is within radius of word i
    masks = [0] * total
    for i in range(total):
        w = all_words[i]
        mask = 0
        # Use numpy for efficient distance calculation
        dists = np.sum(all_words != w, axis=1)
        indices = np.where(dists <= radius)[0]
        for j in indices:
            mask |= (1 << j)
        masks[i] = mask
        
    def popcount(x):
        return bin(x).count('1')
        
    # Initial seed: Direct sum of tetracode if possible
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        seed_indices = []
        word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                idx = word_to_idx[tuple(word)]
                seed_indices.append(idx)
    else:
        seed_indices = [0]
        
    covered_bits = 0
    chosen_indices = []
    
    # Add seeds
    for idx in seed_indices:
        covered_bits |= masks[idx]
        chosen_indices.append(idx)
        
    if popcount(covered_bits) >= total:
        return [list(all_words[i]) for i in chosen_indices]
        
    # Greedy Set Cover
    while popcount(covered_bits) < total:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_idx = -1
        
        for j in range(total):
            gain_bits = masks[j] & ~covered_bits
            if gain_bits:
                gain = popcount(gain_bits)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
        
        if best_idx == -1 or best_gain == 0:
            break
            
        covered_bits |= masks[best_idx]
        chosen_indices.append(best_idx)
        
    # Safety: if not covered, fill gaps
    if popcount(covered_bits) < total:
        for j in range(total):
            if (masks[j] & ~covered_bits) != 0:
                covered_bits |= masks[j]
                chosen_indices.append(j)
                if popcount(covered_bits) >= total:
                    break
                    
    return [list(all_words[i]) for i in chosen_indices]
