import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    def word_to_idx(w):
        idx = 0
        for k in range(n):
            idx = idx * q + w[k]
        return idx

    # Precompute spheres and neighbor lists
    neighbor_indices = [[] for _ in range(total_words)]
    
    for i, word in enumerate(all_words):
        sphere = set([i])
        if radius == 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word_list = list(word)
                    new_word_list[pos] = val
                    new_word = tuple(new_word_list)
                    j = word_to_idx(new_word)
                    sphere.add(j)
        else:
            # Generic radius
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for offsets in itertools.product(range(q), repeat=dist):
                        if all(word[p] == o for p, o in zip(positions, offsets)):
                            continue
                        new_word_list = list(word)
                        for pos, off in zip(positions, offsets):
                            new_word_list[pos] = off
                        new_word = tuple(new_word_list)
                        j = word_to_idx(new_word)
                        sphere.add(j)
        
        neighbor_indices[i] = list(sphere)

    best_size = total_words + 1
    best_solution = None
    
    start_time = time.time()
    max_time = 115
    
    # Pruning approach
    # Data structures for pruning
    # coverers[x] = set of codewords c such that x is in Sphere(c)
    coverers = [set() for _ in range(total_words)]
    
    # Initialize coverers
    for c in range(total_words):
        for x in neighbor_indices[c]:
            coverers[x].add(c)
    
    # cov_count[x] = len(coverers[x])
    cov_count = [len(coverers[x]) for x in range(total_words)]
    
    num_restarts = 1000
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Reset data structures
        # We can reuse coverers if we reset them, but it's faster to just re-init or track changes?
        # Re-init is O(N*S). 1000 * 10^5 = 10^8. Might be tight.
        # Let's try to optimize by only doing a few restarts or optimizing the inner loop.
        # Actually, 10^8 simple ops in Python is ~10-20s.
        # Let's reduce restarts to 200 and see. Or optimize the check.
        
        # For speed, let's just do 100 restarts.
        if restart >= 100:
            break
            
        # Re-init coverers and cov_count for this restart
        # To avoid O(N*S) init cost every time, we can store the initial state and restore?
        # Or just accept the cost.
        coverers_restart = [set() for _ in range(total_words)]
        for c in range(total_words):
            for x in neighbor_indices[c]:
                coverers_restart[x].add(c)
        cov_count_restart = [len(coverers_restart[x]) for x in range(total_words)]
        
        safe_candidates = set(range(total_words))
        selected = set(range(total_words))
        
        # Random order
        order = list(range(total_words))
        random.shuffle(order)
        
        for c in order:
            if c not in safe_candidates:
                continue
                
            # Remove c
            selected.remove(c)
            
            for x in neighbor_indices[c]:
                coverers_restart[x].remove(c)
                cov_count_restart[x] -= 1
                
                if cov_count_restart[x] == 1:
                    # The unique coverer is now unsafe
                    # Get the unique coverer
                    # coverers_restart[x] has size 1
                    unique_coverer = next(iter(coverers_restart[x]))
                    safe_candidates.discard(unique_coverer)
                    
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = list(selected)
            
    return [all_words[i] for i in best_solution]
