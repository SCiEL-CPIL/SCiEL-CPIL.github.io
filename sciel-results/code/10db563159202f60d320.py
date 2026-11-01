import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    total_words = q ** n
    num_coords = n
    alphabet = q
    cover_size = 1 + num_coords * (alphabet - 1)
    
    # Precompute digits
    all_digits = np.zeros((total_words, num_coords), dtype=np.int8)
    for i in range(total_words):
        val = i
        for j in range(num_coords - 1, -1, -1):
            all_digits[i, j] = val % alphabet
            val //= alphabet
            
    # Precompute coverage
    coverage = np.zeros((total_words, cover_size), dtype=np.int32)
    for cw_idx in range(total_words):
        digits = all_digits[cw_idx]
        coverage[cw_idx, 0] = cw_idx
        j = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digits[pos]:
                    continue
                new_digits = digits.copy()
                new_digits[pos] = val
                idx = 0
                for d in new_digits:
                    idx = idx * alphabet + d
                coverage[cw_idx, j] = idx
                j += 1

    covers_word = coverage

    # Greedy Init
    uncovered = np.ones(total_words, dtype=np.uint8)
    count = np.zeros(total_words, dtype=np.int32)
    # Initialize count to cover_size
    count[:] = cover_size
    
    codewords = []
    num_covered = 0
    
    while num_covered < total_words:
        if time.time() > deadline:
            break
        best_cw = np.argmax(count)
        if count[best_cw] == 0:
            break
        codewords.append(best_cw)
        for w_idx in coverage[best_cw]:
            if uncovered[w_idx]:
                uncovered[w_idx] = 0
                num_covered += 1
                for k in range(cover_size):
                    cw_idx = covers_word[w_idx, k]
                    count[cw_idx] -= 1

    # Fallback
    if num_covered < total_words:
        while num_covered < total_words and time.time() < deadline:
            best_cw = np.argmax(count)
            if count[best_cw] == 0: break
            if best_cw not in codewords:
                codewords.append(best_cw)
                for w_idx in coverage[best_cw]:
                    if uncovered[w_idx]:
                        uncovered[w_idx] = 0
                        num_covered += 1
                        for k in range(cover_size):
                            cw_idx = covers_word[w_idx, k]
                            count[cw_idx] -= 1

    # Local Search
    set_of_cw = set(codewords)
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in set_of_cw:
        for j in range(cover_size):
            w_idx = coverage[cw, j]
            cover_count[w_idx] += 1
            
    def try_remove(cw):
        for j in range(cover_size):
            if cover_count[coverage[cw, j]] <= 1:
                return False
        return True

    # Minimize
    def minimize(set_of_cw, cover_count):
        cw_list = list(set_of_cw)
        for cw in cw_list:
            if cw not in set_of_cw:
                continue
            if try_remove(cw):
                set_of_cw.remove(cw)
                for j in range(cover_size):
                    w_idx = coverage[cw, j]
                    cover_count[w_idx] -= 1
        return set_of_cw, cover_count

    # ILS
    while time.time() < deadline - 5:
        set_of_cw, cover_count = minimize(set_of_cw, cover_count)
        
        if time.time() > deadline - 5:
            break
            
        # Perturb
        if len(set_of_cw) > 0:
            orig_set = set(set_of_cw)
            orig_cover = cover_count.copy()
            
            rand_cw = random.choice(list(orig_set))
            new_set = set(orig_set)
            new_set.remove(rand_cw)
            new_cover = orig_cover.copy()
            for j in range(cover_size):
                w_idx = coverage[rand_cw, j]
                new_cover[w_idx] -= 1
                
            # Repair
            while np.any(new_cover == 0) and time.time() < deadline - 5:
                uncovered_mask = (new_cover == 0)
                gain = np.zeros(total_words, dtype=np.int32)
                for cw_idx in range(total_words):
                    if cw_idx in new_set:
                        continue
                    g = 0
                    for j in range(cover_size):
                        w_idx = coverage[cw_idx, j]
                        if uncovered_mask[w_idx]:
                            g += 1
                    gain[cw_idx] = g
                best_cw = np.argmax(gain)
                if gain[best_cw] == 0:
                    break
                new_set.add(best_cw)
                for j in range(cover_size):
                    w_idx = coverage[best_cw, j]
                    new_cover[w_idx] += 1
                    
            # Minimize new state
            new_set, new_cover = minimize(new_set, new_cover)
            
            # Accept if better
            if len(new_set) < len(orig_set):
                set_of_cw = new_set
                cover_count = new_cover
            # If equal, we might accept to move state?
            # Let's accept if equal to explore.
            elif len(new_set) == len(orig_set):
                # Random accept to escape local minima
                if random.random() < 0.5:
                    set_of_cw = new_set
                    cover_count = new_cover

    result = []
    seen = set()
    for cw in set_of_cw:
        digits = all_digits[cw].tolist()
        t = tuple(digits)
        if t not in seen:
            seen.add(t)
            result.append(digits)
    return result
