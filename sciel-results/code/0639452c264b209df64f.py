import time
import random
import itertools
import sys
import numpy as np
import heapq

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
    
    # Greedy Construction with max-heap
    gains = [len(nbrs) for nbrs in neighbors]
    uncovered = [True] * total_words
    code_indices = []
    
    # Use a max-heap: store (-gain, index) since heapq is min-heap
    heap = [(-gains[i], i) for i in range(total_words)]
    heapq.heapify(heap)
    in_heap = [True] * total_words  # Track if index is still in heap
    
    while time.time() < deadline - 10:  # Leave time for local search
        # Find the word with max gain from heap
        best_word = -1
        best_gain = -1
        while heap:
            neg_gain, idx = heapq.heappop(heap)
            if in_heap[idx]:
                best_word = idx
                best_gain = -neg_gain
                in_heap[idx] = False
                break
        
        if best_word == -1 or best_gain <= 0:
            break
            
        code_indices.append(best_word)
        
        # Update gains
        for u in neighbors[best_word]:
            if uncovered[u]:
                uncovered[u] = False
                for c in neighbors[u]:
                    gains[c] -= 1
                    if in_heap[c]:
                        # Push updated gain
                        heapq.heappush(heap, (-gains[c], c))
    
    # Local Search: Removal with multiple passes
    cover_count = [0] * total_words
    
    def update_cover_count(idx, delta):
        for nbr_idx in neighbors[idx]:
            cover_count[nbr_idx] += delta
            
    for c_idx in code_indices:
        update_cover_count(c_idx, 1)
        
    # Multi-pass removal with randomization to escape local minima
    max_removal_passes = 20
    for pass_num in range(max_removal_passes):
        if time.time() > deadline - 15:
            break
            
        improved = False
        new_code_indices = []
        
        # Randomize order for diversity
        indices_list = list(range(len(code_indices)))
        if pass_num > 0:
            random.shuffle(indices_list)
            
        for pos in indices_list:
            c_idx = code_indices[pos]
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
        
        if not improved:
            break
    
    # Local Search: 1-swap moves (replace one codeword with another)
    # Try to find a pair (a, b) where removing a and adding b still covers everything
    remaining_time = deadline - time.time()
    if remaining_time > 5 and len(code_indices) > 1:
        code_set = set(code_indices)
        swap_improved = True
        swap_passes = 0
        max_swap_passes = 50
        
        while swap_improved and time.time() < deadline - 2 and swap_passes < max_swap_passes:
            swap_improved = False
            swap_passes += 1
            new_code_indices = list(code_indices)
            
            # For each codeword in the code, try replacing it with a non-codeword
            # that maintains coverage
            for idx, a in enumerate(new_code_indices):
                if time.time() > deadline - 2:
                    break
                
                # Try a limited number of swap candidates
                # Pick random non-codewords to try, prioritizing words that cover uncovered areas
                num_swaps_to_try = min(100, total_words - len(code_indices))
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
