# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_left():
        return deadline - time.time()
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
    
    # Precompute all words as arrays
    all_words = np.zeros((total_words, n), dtype=np.int32)
    for i in range(total_words):
        val = i
        for pos in range(n - 1, -1, -1):
            all_words[i, pos] = val % q
            val //= q
    
    # Precompute neighbors
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            orig = all_words[w, pos]
            for v in range(q):
                if v == orig:
                    continue
                new_val = w + (v - orig) * (q ** (n - 1 - pos))
                neighbors[w, idx] = new_val
                idx += 1
    
    # Build neighbor sets for fast lookup
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]
    
    # Greedy initialization
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    remaining = total_words
    
    while remaining > 0 and time_left() > 10.0:
        gains = np.zeros(total_words, dtype=np.int32)
        for w in range(total_words):
            if not covered[w]:
                gains[w] = np.sum(~covered[neighbors[w]])
        
        best_word = np.argmax(gains)
        if gains[best_word] == 0:
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            best_word = uncovered[0]
        
        codewords.append(best_word)
        for u in neighbors[best_word]:
            if not covered[u]:
                covered[u] = True
                remaining -= 1
    
    # Local search
    cw = codewords[:]
    best_cw = cw[:]
    best_size = len(cw)
    
    count = np.zeros(total_words, dtype=np.int32)
    for ci in cw:
        for u in neighbors[ci]:
            count[u] += 1
    
    # Main local search loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time_left() > 3.0:
        improved = False
        
        # Phase 1: Remove redundant codewords
        indices = list(range(len(cw)))
        random.shuffle(indices)
        
        removed = 0
        for i in indices:
            if i >= len(cw):
                continue
            ci = cw[i]
            nb = neighbors[ci]
            if np.all(count[nb] > 1):
                for u in nb:
                    count[u] -= 1
                cw.pop(i)
                removed += 1
                improved = True
                if len(cw) < best_size:
                    best_size = len(cw)
                    best_cw = cw[:]
        
        if removed > 0:
            no_improve_count = 0
            continue
        
        # Phase 2: Try swaps
        swaps_tried = 0
        max_swaps = 30
        
        while swaps_tried < max_swaps and time_left() > 3.0:
            swaps_tried += 1
            i = random.randint(0, len(cw) - 1)
            ci = cw[i]
            nb_ci = neighbors[ci]
            critical = [u for u in nb_ci if count[u] == 1]
            
            if not critical:
                for u in nb_ci:
                    count[u] -= 1
                cw.pop(i)
                improved = True
                if len(cw) < best_size:
                    best_size = len(cw)
                    best_cw = cw[:]
                break
            
            # Find candidates that cover all critical words
            candidates = set()
            for u in critical:
                for v in neighbors[u]:
                    if v != ci:
                        candidates.add(v)
            
            if not candidates:
                continue
            
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            found = False
            for cj in cand_list[:15]:
                if all(u in neighbor_sets[cj] for u in critical):
                    for u in nb_ci:
                        count[u] -= 1
                    for u in neighbors[cj]:
                        count[u] += 1
                    cw[i] = cj
                    improved = True
                    found = True
                    break
            
            if found:
                break
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Phase 3: If stuck, try a more aggressive reorganization
        if no_improve_count >= max_no_improve and time_left() > 10.0:
            # Try to find a better starting point by removing some codewords and re-adding
            if len(cw) > 20:
                # Remove 10% of random codewords
                num_remove = max(1, len(cw) // 10)
                remove_indices = random.sample(range(len(cw)), num_remove)
                remove_set = set(remove_indices)
                
                # Recompute coverage
                count2 = np.zeros(total_words, dtype=np.int32)
                for ci in cw:
                    for u in neighbors[ci]:
                        count2[u] += 1
                
                # Remove the selected codewords
                for i in sorted(remove_set, reverse=True):
                    ci = cw[i]
                    for u in neighbors[ci]:
                        count2[u] -= 1
                    cw.pop(i)
                
                # Re-add greedily until covered
                covered2 = count2 > 0
                remaining2 = np.sum(~covered2)
                
                while remaining2 > 0 and time_left() > 3.0:
                    gains2 = np.zeros(total_words, dtype=np.int32)
                    for w in range(total_words):
                        if not covered2[w]:
                            gains2[w] = np.sum(~covered2[neighbors[w]])
                    
                    best_w = np.argmax(gains2)
                    if gains2[best_w] == 0:
                        uncovered2 = np.where(~covered2)[0]
                        if len(uncovered2) == 0:
                            break
                        best_w = uncovered2[0]
                    
                    cw.append(best_w)
                    for u in neighbors[best_w]:
                        if not covered2[u]:
                            covered2[u] = True
                            remaining2 -= 1
                    count2[neighbors[best_w]] += 1
                
                # Update count
                count = count2
                no_improve_count = 0
        
        if no_improve_count >= 100:
            break
    
    # Convert back to list of lists
    result = []
    for ci in best_cw:
        word = []
        val = ci
        for pos in range(n - 1, -1, -1):
            word.append(val % q)
            val //= q
        word.reverse()
        result.append(word)
    
    return result

# EVOLVE-BLOCK-END
