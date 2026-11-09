import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Seed solution: Direct sum of tetracode
    def seed_solution():
        if n < 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_code = seed_solution()
    
    if q != 3 or n != 6 or radius != 1:
        return best_code

    # Greedy algorithm for q=3, n=6, radius=1
    # Total words: 3^6 = 729
    all_words = [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # Precompute neighbors for each word
    # neighbors[i] is a set of indices of words covered by word i
    neighbors = []
    for i, w in enumerate(all_words):
        covered = {i}
        for pos in range(n):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                w[pos] = val
                # Find index of this new word
                # Since 3^6 is small, we can use a dictionary mapping tuple to index
                # But creating a dict for 729 items is fast.
                pass
            w[pos] = orig
        neighbors.append(covered)
        
    # Better: map tuple to index for O(1) lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i, w in enumerate(all_words):
        covered = {i}
        for pos in range(n):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                w[pos] = val
                idx = word_to_idx[tuple(w)]
                covered.add(idx)
            w[pos] = orig
        neighbors.append(covered)
        
    uncovered = set(range(len(all_words)))
    code_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = 0
        # To make it deterministic and potentially better, we can try to break ties
        # or just pick the first max.
        for i in range(len(all_words)):
            if i in code_indices:
                continue
            # Count how many uncovered words are covered by i
            # Optimization: only check if i itself is uncovered? No, a covered word might still cover other uncovered words.
            # However, if we maintain a list of candidate centers, we can be smarter.
            # For now, simple loop. 729 * 73 * (set intersection size)
            # Set intersection size is fast.
            count = len(neighbors[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
                if best_count == len(uncovered):
                    break # Perfect cover found for this step? No, just max possible.
                    
        if best_idx == -1 or best_count == 0:
            break
            
        code_indices.append(best_idx)
        uncovered -= neighbors[best_idx]
        
    if uncovered:
        # If greedy didn't finish or failed, try a random restart or just return seed
        # But greedy usually works well.
        # Let's verify validity of the greedy result
        pass
        
    # Verify the greedy result
    if code_indices:
        code_words = [all_words[i] for i in code_indices]
        # Check if valid
        is_valid = True
        covered_set = set()
        for w in code_words:
            covered_set.add(tuple(w))
            for pos in range(n):
                orig = w[pos]
                for val in range(3):
                    if val == orig:
                        continue
                    w[pos] = val
                    covered_set.add(tuple(w))
                w[pos] = orig
        if len(covered_set) == len(all_words):
            if len(code_words) < len(best_code):
                best_code = code_words
        else:
            # Invalid, stick with seed
            pass
    else:
        # No code found? Return seed
        pass

    return best_code
