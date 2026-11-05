import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # This task is specifically for q=3, n=6, radius=1
    # We use a Greedy Set Cover approach followed by Local Search.
    
    total_words = q ** n
    
    # Precompute all words and their coverage sets
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
        
    # Precompute coverage for each word
    # covers[i] is a list of indices of words covered by word i
    covers = []
    for w in all_words:
        w_list = list(w)
        covered = set()
        covered.add(word_to_idx[tuple(w)])
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_list[pos] = new_val
                covered.add(word_to_idx[tuple(w_list)])
            w_list[pos] = orig_val
        covers.append(list(covered))
        
    # Greedy Set Cover
    selected_bool = [False] * total_words
    selected_list = []
    coverage_count = [0] * total_words
    uncovered_count = total_words
    
    while uncovered_count > 0:
        if time.time() > deadline - 10.0:
            break
            
        best_gain = 0
        best_word = -1
        
        for i in range(total_words):
            if selected_bool[i]:
                continue
                
            gain = 0
            for w_idx in covers[i]:
                if coverage_count[w_idx] == 0:
                    gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_word = i
            # Early termination if we found a word that covers all its potential? No, just max gain.
            # If gain is 0, we can skip further checks for this word? No, we need to check all to find max.
            # But if gain == 0, it's not a candidate.
            
        if best_word == -1:
            break # No progress possible
            
        # Select best_word
        selected_bool[best_word] = True
        selected_list.append(best_word)
        
        # Update coverage
        for w_idx in covers[best_word]:
            if coverage_count[w_idx] == 0:
                uncovered_count -= 1
            coverage_count[w_idx] += 1
            
    best_size = len(selected_list)
    
    # Helper function to try greedy removal
    def try_greedy_removal():
        nonlocal selected_list, coverage_count, best_size
        changed = True
        while changed:
            if time.time() > deadline - 5.0:
                break
            changed = False
            # Try removing each codeword in random order
            random.shuffle(selected_list)
            for idx in range(len(selected_list)):
                if time.time() > deadline - 5.0:
                    break
                c_idx = selected_list[idx]
                # Check if c_idx is redundant
                is_redundant = True
                for w_idx in covers[c_idx]:
                    if coverage_count[w_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove it
                    selected_list.pop(idx)
                    selected_bool[c_idx] = False
                    for w_idx in covers[c_idx]:
                        coverage_count[w_idx] -= 1
                    best_size -= 1
                    changed = True
                    break # Restart shuffle to avoid index issues and ensure full pass
            
    # Phase 1: Initial Greedy Removal
    try_greedy_removal()
    
    # Phase 2: Local Search Swaps
    # We try to find a swap that enables a removal.
    # Swap: Remove i, Add j.
    # If after swap, a removal is possible, we do it.
    # We iterate a fixed number of times or until time limit.
    
    max_swaps = 10000
    for _ in range(max_swaps):
        if time.time() > deadline - 5.0:
            break
            
        # Pick random i to remove
        i = random.randint(0, len(selected_list) - 1)
        old_c = selected_list[i]
        
        # Temporarily remove old_c
        temp_coverage = coverage_count[:]
        for w_idx in covers[old_c]:
            temp_coverage[w_idx] -= 1
            
        # Find uncovered words
        uncovered = [w_idx for w_idx in range(total_words) if temp_coverage[w_idx] == 0]
        
        if len(uncovered) == 0:
            # We can remove old_c directly!
            # This means our previous state was not fully minimized? 
            # Or maybe random shuffle in try_greedy_removal missed it?
            # Unlikely, but handle it.
            selected_list.pop(i)
            selected_bool[old_c] = False
            coverage_count = temp_coverage
            best_size -= 1
            # Run greedy removal again to clean up
            try_greedy_removal()
            continue
            
        # If uncovered > 0, try to find a replacement j
        if len(uncovered) <= 10:
            uncovered_set = set(uncovered)
            found_j = -1
            for j in range(total_words):
                if selected_bool[j]:
                    continue
                # Check if covers[j] covers all uncovered
                valid = True
                for w in uncovered_set:
                    if w not in covers[j]:
                        valid = False
                        break
                if valid:
                    found_j = j
                    break
                    
            if found_j != -1:
                # Perform swap
                # Remove old_c
                selected_list[i] = found_j
                selected_bool[old_c] = False
                selected_bool[found_j] = True
                
                # Update coverage
                # We have temp_coverage (without old_c)
                # Add found_j
                for w_idx in covers[found_j]:
                    temp_coverage[w_idx] += 1
                coverage_count = temp_coverage
                
                # After swap, try to remove something (maybe found_j is redundant? or old_c's slot allows removal of another?)
                # Actually, just run a limited greedy removal check
                # To save time, just check if found_j is redundant? No, it was needed to cover uncovered.
                # Check if any OTHER codeword is now redundant?
                # Let's just call try_greedy_removal with a strict time limit?
                # No, try_greedy_removal might take too long if it loops.
                # Let's do a single pass of removal check.
                
                removal_possible = False
                # Check if found_j is redundant? No.
                # Check if any existing codeword is redundant.
                # We can just pick a random few to check?
                # Or just skip and let the next swap iteration handle it?
                # Better: Just check if we can remove *any* codeword now.
                # We'll do a quick check on a few random candidates.
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    k = random.randint(0, len(selected_list) - 1)
                    c_k = selected_list[k]
                    is_redundant = True
                    for w_idx in covers[c_k]:
                        if coverage_count[w_idx] <= 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        # Remove it
                        selected_list.pop(k)
                        selected_bool[c_k] = False
                        for w_idx in covers[c_k]:
                            coverage_count[w_idx] -= 1
                        best_size -= 1
                        removal_possible = True
                        break
                        
                if removal_possible:
                    # Full cleanup
                    try_greedy_removal()

    # Final Cleanup
    try_greedy_removal()

    # Convert indices back to words
    result = [list(all_words[i]) for i in selected_list]
    
    return result
