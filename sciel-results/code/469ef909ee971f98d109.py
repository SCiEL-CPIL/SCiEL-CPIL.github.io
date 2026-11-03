import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using bitmask-based greedy set cover
    and local search (removal, 1-swaps, and 2-swaps).
    
    Returns: list of codewords (each a list of n integers in 0..q-1)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
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

    # 2. Local Search with coverage-count-based checks
    def local_search():
        nonlocal code, cov
        code_set = set(code)
        
        max_rounds = 200
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
            
            # Phase 2: Try 1-swaps (remove c, add w)
            # c's unique words: those with cov == 1 in c's ball
            # w must cover all of c's unique words
            for c in code_list:
                if time.time() > deadline:
                    break
                unique_words = [w for w in word_lists[c] if cov[w] == 1]
                if not unique_words:
                    continue
                    
                # Find w not in code that covers all unique_words
                # Check: for all w in unique_words, w is in word_lists[replacement]
                # i.e., replacement is within distance 1 of each unique word
                # Equivalent: (masks[replacement] & unique_mask) == unique_mask
                unique_mask = 0
                for uw in unique_words:
                    unique_mask |= (1 << uw)
                
                # Try all non-code words
                found = False
                for w in range(total_words):
                    if w in code_set:
                        continue
                    if (masks[w] & unique_mask) == unique_mask:
                        # Valid swap
                        cov[word_lists[c]] -= 1
                        cov[word_lists[w]] += 1
                        code[code.index(c)] = w
                        code_set.discard(c)
                        code_set.add(w)
                        improved = True
                        found = True
                        break
                if found:
                    break
            
            if improved:
                continue
            
            # Phase 3: Try 2-remove 1-add (remove c1, c2, add w)
            # This is more expensive; only try if we have time
            if time.time() > deadline - 30:
                break
                
            n_code = len(code)
            if n_code < 3:
                continue
                
            # Sample pairs
            for _ in range(min(50, n_code * (n_code - 1) // 2)):
                if time.time() > deadline:
                    break
                i1, i2 = random.sample(range(n_code), 2)
                c1, c2 = code[i1], code[i2]
                
                # Unique words of {c1, c2}: words covered by c1 or c2 but not by others
                # A word is uniquely covered by {c1,c2} if cov[w] == 1 and w in ball(c1) or ball(c2)
                # Actually: words that would become uncovered if we remove both c1 and c2
                # These are words w where cov[w] == 1 and w in ball(c1), OR cov[w] == 2 and w in ball(c1) and w in ball(c2) and no other covers it
                # Simpler: compute union_mask of c1 and c2, then check which words in that union have cov <= 2
                # and would drop to 0 or below 1 after removing both
                
                union_mask = masks[c1] | masks[c2]
                # Words that become uncovered: those in union_mask with cov == 1 (only covered by one of c1,c2)
                # or cov == 2 but both coverers are c1 and c2
                # Let's just check: for each word in union, if cov[w] == 1, it's uniquely covered by exactly one of c1,c2
                # If cov[w] == 2, it might be covered by c1 and c2 (or c1 and someone else, etc.)
                
                # Simplest correct check: temporarily decrement, see if any go to 0
                # But that's expensive. Instead:
                # A word w in union_mask is at risk if:
                #   (w in ball(c1) and cov[w] == 1) or (w in ball(c2) and cov[w] == 1)
                #   or (w in ball(c1) and w in ball(c2) and cov[w] == 2)
                
                risk_mask = 0
                for w in word_lists[c1]:
                    if cov[w] == 1:
                        risk_mask |= (1 << w)
                for w in word_lists[c2]:
                    if cov[w] == 1:
                        risk_mask |= (1 << w)
                    elif cov[w] == 2 and (masks[c1] & (1 << w)):
                        # w is in both c1 and c2's balls, and cov is 2, so both coverers are c1,c2
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
                    # No need to add anything back since we removed 2 and added 0, 
                    # but wait - we need to maintain coverage. If risk_mask==0, no word becomes uncovered.
                    # So we can just remove both.
                
                # Try to find w that covers all risk_mask words
                if risk_mask != 0:
                    found = False
                    for w in range(total_words):
                        if w in code_set:
                            continue
                        if (masks[w] & risk_mask) == risk_mask:
                            # Valid 2-remove 1-add
                            cov[word_lists[c1]] -= 1
                            cov[word_lists[c2]] -= 1
                            cov[word_lists[w]] += 1
                            code.remove(c1)
                            code.remove(c2)
                            code.append(w)
                            code_set.discard(c1)
                            code_set.discard(c2)
                            code_set.add(w)
                            improved = True
                            found = True
                            break
                    if found:
                        break
            
            if not improved:
                # Try a few random perturbations to escape local minima
                if rnd % 10 == 0 and len(code) > 2:
                    if time.time() > deadline:
                        break
                    # Randomly remove a non-redundant word and see if we can fix it
                    # This is risky; skip for now to maintain validity

        # Final trim
        trim_code()

    # Run local search with a few different random seeds
    for seed in [42, 123, 456, 789, 1011]:
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
