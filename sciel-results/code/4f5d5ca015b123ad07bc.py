import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    
    # Precompute the covered set for each word
    covered_sets = [set() for _ in range(total_words)]
    for idx, w in enumerate(all_words):
        covered_sets[idx].add(idx)
        if radius >= 1:
            w_list = list(w)
            for pos in range(n):
                orig = w_list[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w_list[pos] = new_val
                    new_w = tuple(w_list)
                    covered_sets[idx].add(word_to_idx[new_w])
                    w_list[pos] = orig
        # Convert to list for faster numpy indexing later if needed, 
        # but set is good for intersection in greedy.
        # For removal, we need fast access. List is better for numpy.
        # Let's store both.
        
    covered_lists = [list(s) for s in covered_sets]

    # Greedy Construction
    code = []
    remaining_uncovered = set(range(total_words))
    
    while remaining_uncovered and time.time() < deadline:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            if not remaining_uncovered:
                break
            count = len(covered_sets[idx] & remaining_uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            break
            
        code.append(best_idx)
        remaining_uncovered -= covered_sets[best_idx]
        
    def try_remove(code_list):
        code_set = set(code_list)
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_set:
            counts[covered_lists[idx]] += 1
            
        new_code_list = list(code_set)
        random.shuffle(new_code_list)
        
        removed = 0
        for idx in new_code_list:
            if idx not in code_set:
                continue
            min_count = min(counts[covered_lists[idx]])
            if min_count >= 2:
                code_set.remove(idx)
                counts[covered_lists[idx]] -= 1
                removed += 1
        return list(code_set), removed

    current_code = code
    best_code = current_code
    best_size = len(current_code)
    
    # Initial removal passes
    while time.time() < deadline - 5:
        new_code, removed = try_remove(current_code)
        if removed == 0:
            break
        current_code = new_code
        if len(current_code) < best_size:
            best_code = current_code
            best_size = len(current_code)
            
    # Local Search
    stall_count = 0
    while time.time() < deadline - 2:
        if not current_code:
            break
            
        idx_to_replace = random.choice(current_code)
        found_improvement = False
        
        # Try a few random replacements
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            new_idx = random.randint(0, total_words - 1)
            if new_idx == idx_to_replace or new_idx in current_code:
                continue
                
            temp_code = [c for c in current_code if c != idx_to_replace]
            temp_code.append(new_idx)
            
            new_code, _ = try_remove(temp_code)
            
            if len(new_code) < best_size:
                current_code = new_code
                best_code = new_code
                best_size = len(new_code)
                found_improvement = True
                break
            elif len(new_code) < len(current_code):
                current_code = new_code
                found_improvement = True
                break

        if found_improvement:
            stall_count = 0
        else:
            stall_count += 1
            # If stalled for a long time, break or reset?
            # We can't reset easily without losing progress.
            # Just keep trying.
            if stall_count > 500:
                break

    return [list(idx_to_word[idx]) for idx in best_code]
