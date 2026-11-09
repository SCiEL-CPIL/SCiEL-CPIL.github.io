import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Specific to q=3, n=6, radius=1
    # General logic below, but optimized for this case if possible.
    
    total_words = q ** n
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (indices)
    # neighbor_idx[i] is a list of indices of words within distance 'radius' of word i
    # For radius=1, this is 1 + n*(q-1)
    neighbor_indices = []
    for i, w in enumerate(all_words):
        nbrs = [i]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    nbrs.append(word_to_idx[tuple(nw)])
        neighbor_indices.append(nbrs)
    
    # Initialize with Direct Sum Construction (81 words for q=3, n=6)
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Direct sum with all words of length 2
    tails = list(itertools.product(range(3), repeat=2))
    
    initial_code_indices = []
    for h in tetracode:
        for t in tails:
            w = h + list(t)
            initial_code_indices.append(word_to_idx[tuple(w)])
            
    current_code = set(initial_code_indices)
    
    # Initialize coverage counts
    # cover_count[i] = number of codewords in current_code that are within distance radius of word i
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in current_code:
        for nbr in neighbor_indices[cw]:
            cover_count[nbr] += 1
            
    best_code_size = len(current_code)
    best_code_set = set(current_code)
    
    # Local Search Loop
    # 1. Try to remove redundant words
    # 2. If stuck, try random swaps
    
    # Helper to check if current state is valid (all cover_count > 0)
    # We maintain cover_count, so we just check min(cover_count) > 0
    
    # Helper to try removing a word
    def try_remove(cw_idx):
        nonlocal cover_count, current_code
        # Decrement
        for nbr in neighbor_indices[cw_idx]:
            cover_count[nbr] -= 1
        # Check validity
        if np.min(cover_count) > 0:
            current_code.remove(cw_idx)
            return True
        # Revert
        for nbr in neighbor_indices[cw_idx]:
            cover_count[nbr] += 1
        return False

    # Helper to try swapping out and in
    def try_swap(cw_out, cw_in):
        nonlocal cover_count, current_code
        if cw_out == cw_in:
            return False
        if cw_in in current_code:
            return False
        # Decrement out
        for nbr in neighbor_indices[cw_out]:
            cover_count[nbr] -= 1
        # Increment in
        for nbr in neighbor_indices[cw_in]:
            cover_count[nbr] += 1
        # Check validity
        if np.min(cover_count) > 0:
            current_code.remove(cw_out)
            current_code.add(cw_in)
            return True
        # Revert
        for nbr in neighbor_indices[cw_out]:
            cover_count[nbr] += 1
        for nbr in neighbor_indices[cw_in]:
            cover_count[nbr] -= 1
        return False

    # Main Loop
    while time.time() < deadline:
        # Phase 1: Greedy Deletion
        # Iterate through current code, try to remove each
        # Repeat until no change
        changed = True
        while changed:
            changed = False
            # Randomize order to avoid bias
            for cw in list(current_code):
                if try_remove(cw):
                    changed = True
                    # Update best if improved
                    if len(current_code) < best_code_size:
                        best_code_size = len(current_code)
                        best_code_set = set(current_code)
                    break # Restart deletion pass after a removal
        
        # Phase 2: Random Swaps (if stuck)
        # We only do swaps if we haven't improved in a while or just to explore
        # To save time, we do a fixed number of swap attempts per loop iteration
        # If a swap succeeds, we go back to deletion phase
        
        # Heuristic: Only swap if we are not making progress with deletions?
        # Actually, swaps allow us to escape local minima where no single deletion is possible.
        
        # Let's do a batch of swap attempts
        swap_attempts = 100
        for _ in range(swap_attempts):
            if time.time() > deadline - 1.0:
                break
                
            # Pick random word to remove
            if not current_code:
                break
            cw_out = random.choice(list(current_code))
            
            # Pick random word to add
            # Bias towards words that might cover "sparse" areas?
            # For simplicity, uniform random
            cw_in = random.randint(0, total_words - 1)
            
            if try_swap(cw_out, cw_in):
                # Swap succeeded, code is still valid
                # Update best if size is same? No, size is same.
                # But structure changed.
                # We should immediately try to delete again
                # The 'changed' flag in Phase 1 will handle this in next iteration
                # But we can break out of swap loop to do deletions
                break
        
        # If no swap succeeded, we are in a local minimum for single swaps.
        # We could try 2-swaps, but that's expensive.
        # Given the time, single swaps + deletions are usually sufficient for K_3(6,1)
        # to reach 73 from 81.

    # Convert best_code_set to list of words
    result = [list(all_words[idx]) for idx in best_code_set]
    return result
