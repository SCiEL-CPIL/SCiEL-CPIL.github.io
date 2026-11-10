"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Generate all possible words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index for fast lookup
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks for all words
    # mask[i] is a bitmask where bit j is set if word j is within radius of word i
    # For radius 1, neighbors are the word itself and words differing in 1 position.
    def get_word_mask(word):
        mask = 0
        idx = word_to_idx[word]
        mask |= (1 << idx)
        
        word_list = list(word)
        for i in range(n):
            orig_val = word_list[i]
            for val in range(q):
                if val == orig_val:
                    continue
                word_list[i] = val
                new_word = tuple(word_list)
                new_idx = word_to_idx[new_word]
                mask |= (1 << new_idx)
            word_list[i] = orig_val
        return mask

    all_masks = [get_word_mask(w) for w in all_words]
    
    # Initial code: Direct sum of tetracode
    if n >= 4:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        initial_codewords = list(all_words)
        
    # Ensure distinct codewords and convert to tuples
    initial_codewords = list(set(tuple(c) for c in initial_codewords))
    current_indices = [word_to_idx[c] for c in initial_codewords]
    current_masks = [all_masks[i] for i in current_indices]
    
    # Helper to check validity
    def is_valid(masks_list):
        union_mask = 0
        for m in masks_list:
            union_mask |= m
        return union_mask == (1 << N) - 1
        
    if not is_valid(current_masks):
        # Fallback: greedy construction
        pass

    # Helper to compute cover counts
    def compute_cover_counts(masks_list):
        counts = [0] * N
        for m in masks_list:
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                temp ^= lsb
        return counts

    # Helper to perform greedy removal
    def greedy_remove(indices, masks):
        indices = list(indices)
        masks = list(masks)
        counts = compute_cover_counts(masks)
        
        removed = True
        while removed:
            removed = False
            for i in range(len(indices)):
                can_remove = True
                m = masks[i]
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        can_remove = False
                        break
                    temp ^= lsb
                
                if can_remove:
                    # Update counts
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
                    
                    indices.pop(i)
                    masks.pop(i)
                    removed = True
                    break # Restart scan
            
        return indices, masks

    # 1. Initial Greedy Removal
    current_indices, current_masks = greedy_remove(current_indices, current_masks)
    
    best_indices = list(current_indices)
    best_masks = list(current_masks)
    best_size = len(best_indices)
    
    # 2. Local Search: Add-Swap-Remove
    # Strategy: Add a random word, then greedily remove as many as possible.
    # If the resulting size is smaller, accept.
    
    # Precompute a list of available words for random selection? 
    # Or just pick random indices from 0..N-1 and check if they are in the set.
    
    in_set = [False] * N
    for i in best_indices:
        in_set[i] = True
        
    # To speed up random selection of a word NOT in the set, 
    # we can maintain a list of available indices or just retry.
    # Given N=729 and set size ~73, the probability of picking an available word is high.
    
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        # Pick a random word not in the current set
        while True:
            add_idx = random.randint(0, N - 1)
            if not in_set[add_idx]:
                break
        
        # Add it
        new_indices = best_indices + [add_idx]
        new_masks = best_masks + [all_masks[add_idx]]
        
        # Greedy removal
        new_indices, new_masks = greedy_remove(new_indices, new_masks)
        
        if len(new_indices) < best_size:
            best_indices = new_indices
            best_masks = new_masks
            best_size = len(best_indices)
            # Update in_set
            in_set = [False] * N
            for i in best_indices:
                in_set[i] = True
        elif len(new_indices) == best_size:
            # Tie-breaking: sometimes accept a different solution of the same size 
            # to escape local optima? 
            # For now, let's just keep the best size found.
            pass

    return [list(all_words[i]) for i in best_indices]
# EVOLVE-BLOCK-END
