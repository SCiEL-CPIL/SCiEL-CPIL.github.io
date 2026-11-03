import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using bitmask-based greedy set cover
    and local search (removal, 1-swaps, and 2-swaps) with candidate-pool
    optimization for fast swap searches.
    
    Returns: list of codewords (each a list of n integers in 0..q-1)
    """
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1
    
    # Helper functions for base conversion
    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute neighborhoods as bitmasks AND as explicit word lists
    masks = [0] * total_words
    word_lists = [[] for _ in range(total_words)]
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        mask = 1 << i
        word_lists[i].append(i)
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                nb_idx = digits_to_int(d, q)
                mask |= (1 << nb_idx)
                word_lists[i].append(nb_idx)
            d[pos] = orig
        masks[i] = mask

    # 1. Greedy Set Cover
    uncovered = (1 << total_words) - 1
    code = []
    
    while uncovered != 0:
        if time.time() > deadline:
            break
            
        best_center = -1
        best_gain = 0
        
        for c in range(total_words):
            gain = (masks[c] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_center = c
        
        if best_center == -1 or best_gain == 0:
            break
            
        code.append(best_center)
        uncovered &= ~masks[best_center]

    # Coverage count array: cov[w] = number of codewords whose ball covers word w
    cov = np.zeros(total_words, dtype=np.int32)
    for c in code:
        cov[word_lists[c]] += 1

    def trim_code():
        nonlocal code, cov
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                # c is redundant if every word in its ball has coverage >= 2
                if np.min(cov[word_lists[c]]) >= 2:
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    cov[word_lists[c]] -= 1
                    code.pop(idx)
                changed = True

    trim_code()

    # 2. Local Search with candidate-pool optimization
    def local_search():
        nonlocal code, cov
        code_set = set(code)
        
        max_rounds = 500
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
                
            improved = False
            code_list = list(code)
            random.shuffle(code_list)
            
            # Phase 1: Try removals
            for c in code_list:
                if np.min(cov[word_lists[c]]) >= 2:
                    cov[word_lists[c]] -= 1
                    code.remove(c)
                    code_set.discard(c)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try 1-swaps (remove c, add w) using candidate pool
            for c in code_list:
                if time.time() > deadline:
                    break
                unique_words = [w for w in word_lists[c] if cov[w] == 1]
                if not unique_words:
                    continue
                    
                # Build candidate pool: intersection of neighbors of all unique words
                # A valid replacement w must be within distance 1 of every unique word
                # i.e., w is in the neighborhood of each unique word
                # Start with neighbors of the first unique word, intersect with rest
                candidate_pool = set(word_lists[unique_words[0]])
                for uw in unique_words[1:]:
                    candidate_pool &= set(word_lists[uw])
                    if not candidate_pool:
                        break
                
                if not candidate_pool:
                    continue
                
                # Filter out existing codewords
                candidate_pool -= code_set
                
                if not candidate_pool:
                    continue
                
                # Pick the best candidate (one that covers the most currently covered words
                # to maximize future flexibility, or just pick the first valid one)
                # For speed, pick the first one; for quality, pick one with max overlap
                best_w = None
                best_overlap = -1
                for w in candidate_pool:
                    # Count how many of w's neighbors are already covered (overlap)
                    # Higher overlap means the new codeword integrates better
                    overlap = 0
                    for nb in word_lists[w]:
                        if cov[nb] > 0:
                            overlap += 1
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_w = w
                
                if best_w is not None:
                    cov[word_lists[c]] -= 1
                    cov[word_lists[best_w]] += 1
                    code[code.index(c)] = best_w
                    code_set.discard(c)
                    code_set.add(best_w)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 3: Try 2-remove 1-add (remove c1, c2, add w)
            if time.time() > deadline - 20:
                break
                
            n_code = len(code)
            if n_code < 3:
                continue
                
            # Sample pairs
            for _ in range(min(30, n_code * (n_code - 1) // 2)):
                if time.time() > deadline:
                    break
                i1, i2 = random.sample(range(n_code), 2)
                c1, c2 = code[i1], code[i2]
                
                # Compute risk mask: words that would become uncovered
                risk_mask = 0
                for w in word_lists[c1]:
                    if cov[w] == 1:
                        risk_mask |= (1 << w)
                for w in word_lists[c2]:
                    if cov[w] == 1:
                        risk_mask |= (1 << w)
                    elif cov[w] == 2 and (masks[c1] & (1 << w)):
                        risk_mask |= (1 << w)
                
                if risk_mask == 0:
                    # Both c1 and c2 are redundant together - remove both
                    cov[word_lists[c1]] -= 1
                    cov[word_lists[c2]] -= 1
                    code.remove(c1)
                    code.remove(c2)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    improved = True
                    break
                
                # Try to find w that covers all risk_mask words using candidate pool
                # Extract unique words from risk_mask
                risk_words = []
                rm = risk_mask
                while rm:
                    lsb = rm & (-rm)
                    idx = lsb.bit_length() - 1
                    risk_words.append(idx)
                    rm ^= lsb
                
                if not risk_words:
                    continue
                
                # Build candidate pool: intersection of neighbors of all risk words
                candidate_pool = set(word_lists[risk_words[0]])
                for rw in risk_words[1:]:
                    candidate_pool &= set(word_lists[rw])
                    if not candidate_pool:
                        break
                
                if not candidate_pool:
                    continue
                
                candidate_pool -= code_set
                
                if not candidate_pool:
                    continue
                
                # Pick best candidate
                best_w = None
                best_overlap = -1
                for w in candidate_pool:
                    overlap = 0
                    for nb in word_lists[w]:
                        if cov[nb] > 0:
                            overlap += 1
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_w = w
                
                if best_w is not None:
                    cov[word_lists[c1]] -= 1
                    cov[word_lists[c2]] -= 1
                    cov[word_lists[best_w]] += 1
                    code.remove(c1)
                    code.remove(c2)
                    code.append(best_w)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    code_set.add(best_w)
                    improved = True
                    break
            
            if not improved:
                # Try a few random perturbations to escape local minima
                if rnd % 15 == 0 and len(code) > 2:
                    if time.time() > deadline:
                        break
                    # Randomly try a 1-swap with a random codeword
                    c = random.choice(code)
                    unique_words = [w for w in word_lists[c] if cov[w] == 1]
                    if unique_words:
                        candidate_pool = set(word_lists[unique_words[0]])
                        for uw in unique_words[1:]:
                            candidate_pool &= set(word_lists[uw])
                            if not candidate_pool:
                                break
                        candidate_pool -= code_set
                        if candidate_pool:
                            best_w = random.choice(list(candidate_pool))
                            cov[word_lists[c]] -= 1
                            cov[word_lists[best_w]] += 1
                            code[code.index(c)] = best_w
                            code_set.discard(c)
                            code_set.add(best_w)
                            improved = True

        # Final trim
        trim_code()

    # Run local search with a few different random seeds
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044]:
        if time.time() > deadline - 5:
            break
        random.seed(seed)
        local_search()
        trim_code()

    # Final validation check (should always pass)
    # Convert back to list of lists
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))
        
    return result
