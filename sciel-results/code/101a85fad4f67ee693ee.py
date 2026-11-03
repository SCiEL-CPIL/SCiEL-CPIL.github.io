# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct(q, n, radius):
    # Generate all words in {0,...,q-1}^n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total = all_words.shape[0]  # q**n
    
    # Precompute coverage matrix
    cov_size = 1 + n * (q - 1)
    coverage_matrix = np.zeros((total, cov_size), dtype=np.int32)
    
    # Precompute powers of q for index calculation
    powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    for i in range(total):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                # Calculate index of neighbor
                # idx = w @ powers + (val - orig) * powers[pos]
                idx = int(np.dot(w, powers)) + (val - orig) * powers[pos]
                neighbors.append(idx)
        coverage_matrix[i] = neighbors
        
    # Efficient Randomized Greedy Set Cover
    best_selected = None
    best_len = total + 1
    start_time = time.time()
    timeout = 110
    
    seed = 42
    while time.time() - start_time < timeout:
        rng = np.random.default_rng(seed)
        uncovered = np.ones(total, dtype=bool)
        cover_count = np.full(total, cov_size, dtype=np.int32)
        selected = []
        
        while uncovered.any():
            max_count = cover_count.max()
            if max_count == 0:
                break
            
            candidates = np.where(cover_count == max_count)[0]
            best_idx = candidates[rng.integers(len(candidates))]
            
            selected.append(int(best_idx))
            
            # Update cover_count and uncovered
            # For each neighbor j of best_idx
            for j in coverage_matrix[best_idx]:
                if uncovered[j]:
                    uncovered[j] = False
                    # Decrement cover_count for all candidates covering j
                    # which are the neighbors of j
                    cover_count[coverage_matrix[j]] -= 1
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_selected = selected[:]
            
        seed += 1

    if best_selected is None:
        # Fallback
        uncovered = np.ones(total, dtype=bool)
        cover_count = np.full(total, cov_size, dtype=np.int32)
        best_selected = []
        while uncovered.any():
            max_count = cover_count.max()
            if max_count == 0: break
            best_idx = np.argmax(cover_count)
            best_selected.append(int(best_idx))
            for j in coverage_matrix[best_idx]:
                if uncovered[j]:
                    uncovered[j] = False
                    cover_count[coverage_matrix[j]] -= 1

    codewords = [all_words[i].tolist() for i in best_selected]
    return codewords
# EVOLVE-BLOCK-END
