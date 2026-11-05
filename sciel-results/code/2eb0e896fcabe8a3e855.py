"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return [list(w) for w in all_words]

    # Specific optimization for q=3, n=8, radius=1
    # Start with the direct sum of the ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    m = len(initial_code)
    
    # Map from word tuple to index in 0..6560
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx

    # Precompute neighborhoods for each codeword in initial_code
    # neighborhoods[i] is a list of 17 indices
    neighborhoods = []
    for c in initial_code:
        neigh = [word_to_idx(c)]
        for pos in range(n):
            orig = c[pos]
            for val in range(3):
                if val != orig:
                    new_c = c[:]
                    new_c[pos] = val
                    neigh.append(word_to_idx(new_c))
        neighborhoods.append(neigh)
    
    # Coverage count: how many codewords cover each word
    coverage = np.zeros(6561, dtype=np.int32)
    for neigh in neighborhoods:
        for idx in neigh:
            coverage[idx] += 1
            
    # Set of active codeword indices
    active = set(range(m))
    
    # Peeling loop
    while True:
        best_idx = -1
        best_score = -1
        
        for i in active:
            neigh = neighborhoods[i]
            # Check if removable: all neighbors covered at least twice
            # Using numpy min for speed
            if np.min(coverage[neigh]) >= 2:
                # Calculate redundancy score
                score = np.sum(coverage[neigh])
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
        if best_idx == -1:
            break
            
        # Remove best_idx
        neigh = neighborhoods[best_idx]
        coverage[neigh] -= 1
        active.remove(best_idx)
        
    # Reconstruct the list of codewords
    result = [initial_code[i] for i in sorted(active)]
    return result
# EVOLVE-BLOCK-END
