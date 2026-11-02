import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    timeout = 115
    
    # 1. Precomputation
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute spheres
    spheres = [None] * total_words
    for i in range(total_words):
        word = idx_to_word(i)
        sphere = [i]
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    word[pos] = new_val
                    j = word_to_idx(word)
                    sphere.append(j)
            word[pos] = orig_val
        spheres[i] = sphere

    # 2. Initialization
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        centers = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                centers.append(word_to_idx(h + list(tail)))
    else:
        centers = list(range(total_words))
        random.shuffle(centers)
        centers = centers[:total_words // 2]

    # 3. Optimization
    def init_coverage(centers_list):
        cov = np.zeros(total_words, dtype=np.int32)
        for c in centers_list:
            for w in spheres[c]:
                cov[w] += 1
        return cov

    def remove_redundant(centers_list, cov):
        changed = True
        while changed:
            changed = False
            new_centers = []
            for c in centers_list:
                is_redundant = True
                for w in spheres[c]:
                    if cov[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in spheres[c]:
                        cov[w] -= 1
                    changed = True
                else:
                    new_centers.append(c)
            centers_list = new_centers
        return centers_list

    coverage = init_coverage(centers)
    centers = remove_redundant(centers, coverage)
    
    best_centers = centers.copy()
    best_coverage = coverage.copy()
    best_len = len(centers)
    
    # Local Search
    while time.time() - start_time < timeout:
        curr_centers = best_centers.copy()
        curr_coverage = best_coverage.copy()
        
        if not curr_centers:
            break
            
        c_to_remove = random.choice(curr_centers)
        w_to_add = random.randint(0, total_words - 1)
        
        if c_to_remove == w_to_add:
            continue
            
        # Avoid adding duplicate
        if w_to_add in curr_centers:
            continue
            
        # Apply change
        for w in spheres[c_to_remove]:
            curr_coverage[w] -= 1
        curr_centers.remove(c_to_remove)
        
        for w in spheres[w_to_add]:
            curr_coverage[w] += 1
        curr_centers.append(w_to_add)
        
        # Optimize
        curr_centers = remove_redundant(curr_centers, curr_coverage)
        
        if len(curr_centers) < best_len:
            best_len = len(curr_centers)
            best_centers = curr_centers
            best_coverage = curr_coverage

    result = [idx_to_word(c) for c in best_centers]
    return result
