"""Combined program: greedy covering code construction with NumPy acceleration."""
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    total_words = q ** n
    
    # For very large spaces, fall back to direct sum construction
    if total_words > 50000:
        # Use tetracode-based direct sum for q=3
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            base_n = 4
            base_code = tetracode
        else:
            # Generic fallback: use seed logic
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            base_n = 4
            base_code = tetracode
        
        if n < base_n:
            return [list(range(n))]
        
        tails = list(itertools.product(range(q), repeat=n - base_n))
        return [h + list(tail) for h in base_code for tail in tails]
    
    # For small spaces, use NumPy-accelerated greedy
    # Precompute neighbors for each word (distance <= radius)
    if radius == 1:
        sphere_size = 1 + n * (q - 1)
        neighbors = np.full((total_words, sphere_size), -1, dtype=np.int32)
        
        for i in range(total_words):
            nbrs = [i]
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                current_val = (i // pow_q) % q
                for new_val in range(q):
                    if new_val != current_val:
                        nbrs.append(i + (new_val - current_val) * pow_q)
            neighbors[i] = nbrs
    else:
        # Generic radius: compute neighbors by checking Hamming distance
        # This is slower but correct for any radius
        sphere_size = 1
        # Upper bound on sphere size
        from math import comb
        for k in range(min(radius, n) + 1):
            sphere_size += comb(n, k) * (q - 1) ** k
        
        neighbors = np.full((total_words, sphere_size), -1, dtype=np.int32)
        neighbor_counts = np.zeros(total_words, dtype=np.int32)
        
        # Precompute powers
        q_powers = [q ** (n - 1 - pos) for pos in range(n)]
        
        for i in range(total_words):
            # Get word from index
            w = np.array([(i // q_powers[pos]) % q for pos in range(n)], dtype=np.int32)
            
            nbrs = [i]
            # Generate all words within Hamming distance radius
            # Use recursive generation
            def gen_neighbors(pos, remaining_dist, current_word, current_idx):
                if pos == n:
                    if remaining_dist >= 0:
                        nbrs.append(current_idx)
                    return
                # Keep same value
                gen_neighbors(pos + 1, remaining_dist, current_word, current_idx)
                # Change value
                if remaining_dist > 0:
                    for new_val in range(q):
                        if new_val != current_word[pos]:
                            new_idx = current_idx + (new_val - current_word[pos]) * q_powers[pos]
                            gen_neighbors(pos + 1, remaining_dist - 1, current_word, new_idx)
            
            gen_neighbors(0, radius, w, i)
            
            # Deduplicate and trim
            nbrs = list(set(nbrs))
            if len(nbrs) <= sphere_size:
                neighbors[i, :len(nbrs)] = nbrs
                neighbor_counts[i] = len(nbrs)
            else:
                # Should not happen, but handle gracefully
                neighbors[i] = nbrs[:sphere_size]
                neighbor_counts[i] = sphere_size
    
    # Greedy algorithm
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    while time.time() < deadline:
        # Calculate gains: number of uncovered words each candidate covers
        if radius == 1:
            gains = (~covered)[neighbors].sum(axis=1)
        else:
            # For variable-length neighbors, need to handle carefully
            # Create a masked version
            gains = np.zeros(total_words, dtype=np.int32)
            for i in range(total_words):
                cnt = neighbor_counts[i]
                if cnt > 0:
                    gains[i] = (~covered)[neighbors[i, :cnt]].sum()
        
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
        
        code_indices.append(best_idx)
        
        # Mark covered
        if radius == 1:
            covered[neighbors[best_idx]] = True
        else:
            cnt = neighbor_counts[best_idx]
            if cnt > 0:
                covered[neighbors[best_idx, :cnt]] = True
    
    # Convert indices back to words
    q_powers = [q ** (n - 1 - pos) for pos in range(n)]
    result_words = []
    for idx in code_indices:
        w = [(idx // q_powers[pos]) % q for pos in range(n)]
        result_words.append(w)
    
    return result_words
