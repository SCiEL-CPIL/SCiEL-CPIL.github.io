import time
import random
import itertools
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for w in all_words:
        nbrs = [w]
        for i in range(n):
            orig = w[i]
            for val in range(q):
                if val != orig:
                    new_w = list(w)
                    new_w[i] = val
                    nbrs.append(tuple(new_w))
        neighbors.append([word_to_idx[t] for t in nbrs])
    
    # Build neighbor list as list of lists for fast access
    # Also build a set for each neighbor for O(1) membership if needed
    neighbor_sets = [set(nbrs) for nbrs in neighbors]
    
    # Greedy Construction
    gains = [len(nbrs) for nbrs in neighbors]
    uncovered = [True] * total_words
    code_indices = []
    
    # Use a max-heap approach or just scan (N=6561 is small)
    while time.time() < deadline:
        # Find the word with max gain
        best_word = -1
        best_gain = -1
        for i in range(total_words):
            if gains[i] > best_gain:
                best_gain = gains[i]
                best_word = i
        
        if best_word == -1 or best_gain <= 0:
            break
            
        code_indices.append(best_word)
        
        # Update gains
        for u in neighbors[best_word]:
            if uncovered[u]:
                uncovered[u] = False
                for c in neighbors[u]:
                    gains[c] -= 1
    
    # Local Search: Removal
    cover_count = [0] * total_words
    
    def update_cover_count(idx, delta):
        for nbr_idx in neighbors[idx]:
            cover_count[nbr_idx] += delta
            
    for c_idx in code_indices:
        update_cover_count(c_idx, 1)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_code_indices = []
        for c_idx in code_indices:
            redundant = True
            for nbr_idx in neighbors[c_idx]:
                if cover_count[nbr_idx] <= 1:
                    redundant = False
                    break
                    
            if redundant:
                update_cover_count(c_idx, -1)
                improved = True
            else:
                new_code_indices.append(c_idx)
                
        code_indices = new_code_indices
    
    # Local Search: 1-swap moves (replace one codeword with another)
    # Try to find a pair (a, b) where removing a and adding b still covers everything
    # This can sometimes reduce the code size if b covers more "critical" words
    remaining_time = deadline - time.time()
    if remaining_time > 5 and len(code_indices) > 1:
        code_set = set(code_indices)
        swap_improved = True
        while swap_improved and time.time() < deadline:
            swap_improved = False
            new_code_indices = list(code_indices)
            
            # For each codeword in the code, try replacing it with a non-codeword
            # that maintains coverage
            for idx, a in enumerate(new_code_indices):
                if time.time() > deadline - 2:
                    break
                
                # Remove a temporarily
                update_cover_count(a, -1)
                
                # Check if all words are still covered
                all_covered = True
                for u in range(total_words):
                    if cover_count[u] <= 0:
                        all_covered = False
                        break
                
                if not all_covered:
                    # Restore a
                    update_cover_count(a, 1)
                    continue
                
                # All covered without a. Now try to add a different word b
                # that might allow further removals later
                # For now, just remove a (this is the removal case, already handled)
                # Actually, we want to swap: remove a, add b, where b != a
                # But if all is covered without a, we can just remove a (done by removal pass)
                # The swap is useful when we need to keep the same size but improve structure
                # Let's try: find b not in code such that adding b and removing some c works
                # This is complex. Let's do a simpler improvement:
                # Try to find a replacement b for a such that the new code is still valid
                # and has the same size (no net reduction, but may enable future reductions)
                
                # For efficiency, let's just check if removing a works (already done)
                # and skip the swap for now since removal is the main gain
                update_cover_count(a, 1)
                
                # Try a limited number of swap candidates
                # Pick random non-codewords to try
                num_swaps_to_try = min(50, total_words - len(code_indices))
                tried = 0
                while tried < num_swaps_to_try and time.time() < deadline - 1:
                    b = random.randint(0, total_words - 1)
                    if b in code_set:
                        continue
                    tried += 1
                    
                    # Try swapping a with b
                    # Remove a
                    update_cover_count(a, -1)
                    # Add b
                    update_cover_count(b, 1)
                    
                    # Check coverage
                    valid = True
                    for u in range(total_words):
                        if cover_count[u] <= 0:
                            valid = False
                            break
                    
                    if valid:
                        # Accept the swap
                        new_code_indices[idx] = b
                        code_set.discard(a)
                        code_set.add(b)
                        swap_improved = True
                        break
                    else:
                        # Undo
                        update_cover_count(b, -1)
                        update_cover_count(a, 1)
                
                # If we didn't find a swap, a is still in the code
                # (coverage was restored in the loop above)
            
            if swap_improved:
                code_indices = new_code_indices
    
    return [list(all_words[i]) for i in code_indices]
