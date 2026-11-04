import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute powers for index conversion
    powers = [q ** p for p in range(n)]

    # Generate neighbors matrix using vectorization
    # neighbors[i] contains indices of all words within Hamming distance 1 of word i
    neighbors = np.empty((N, 1 + n * (q - 1)), dtype=np.int32)
    neighbors[:, 0] = np.arange(N)
    
    # Vectorized neighbor generation
    for pos in range(n):
        for alt in range(q):
            # Calculate indices for words differing at 'pos' with value 'alt'
            # We need to be careful not to include the word itself if alt == original value
            # But since we loop all alts, we can just calculate all and filter out self later or handle in logic
            # Actually, it's easier to just compute the index change.
            
            # Get the digit at 'pos' for all words
            # i // powers[pos] % q
            digits = (np.arange(N) // powers[pos]) % q
            
            # Mask for words that do NOT have value 'alt' at 'pos'
            # We want to add neighbors where the value IS 'alt'
            # Wait, the neighbor set for word i includes words j such that dist(i,j)=1.
            # So for a fixed i, j differs at pos.
            # It is more efficient to build the adjacency by iterating over positions and values.
            
            # Let's stick to a simpler loop for neighbor generation to ensure correctness, 
            # but use numpy for the heavy lifting in the search.
            pass

    # Re-do neighbor generation efficiently
    neighbors = np.empty((N, 13), dtype=np.int32)
    neighbors[:, 0] = np.arange(N)
    k = 1
    for pos in range(n):
        for alt in range(q):
            # For each word i, if its value at pos is not alt, then the word formed by changing pos to alt is a neighbor.
            # We can compute these indices in bulk.
            digits = (np.arange(N) // powers[pos]) % q
            mask = digits != alt
            # For indices i where mask is True, the neighbor index is i + (alt - digits[i]) * powers[pos]
            # We need to fill neighbors[i, k] for these i.
            # This is tricky to vectorize directly into a column because we need to know which column k for which i.
            # Given N=729 is small, a python loop for neighbor construction is fast enough (<0.1s).
            pass

    # Fast python loop for neighbor construction (N=729 is small)
    neighbors = np.zeros((N, 13), dtype=np.int32)
    for i in range(N):
        neighbors[i, 0] = i
        col = 1
        for pos in range(n):
            orig_val = (i // powers[pos]) % q
            for alt in range(q):
                if alt == orig_val:
                    continue
                new_val = i + (alt - orig_val) * powers[pos]
                neighbors[i, col] = new_val
                col += 1

    # Greedy Set Cover with Vectorized Updates
    uncovered = np.ones(N, dtype=np.int8)
    selected_indices = []
    
    # Precompute neighbor lists as a list of arrays for faster access in python loops if needed, 
    # but we will use numpy indexing for the main loop.
    
    while uncovered.sum() > 0:
        if time.time() > deadline - 20.0: # Reserve time for local search
            break
        
        # Count how many uncovered neighbors each candidate covers
        # uncovered[neighbors] gives a (N, 13) boolean matrix
        # Sum along axis 1 gives the count for each candidate
        counts = np.sum(uncovered[neighbors], axis=1)
        
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = 0

    # Initial solution
    best_selected = selected_indices.copy()
    global_best_size = len(best_selected)
    
    # Local Search
    # We maintain a list of selected indices and a coverage count array.
    # To speed up, we use numpy for coverage updates.
    
    def get_coverage(sel_list):
        cov = np.zeros(N, dtype=np.int32)
        for idx in sel_list:
            cov[neighbors[idx]] += 1
        return cov

    best_cov = get_coverage(best_selected)
    
    # Helper: Check if a codeword can be removed
    def can_remove(idx, cov):
        # All neighbors of idx must have coverage > 1
        return np.min(cov[neighbors[idx]]) >= 2

    # Local Search Loop
    # Strategy: 
    # 1. Try to remove redundant codewords.
    # 2. If stuck, try a swap: remove one, add one different, then try to remove more.
    
    current_selected = best_selected.copy()
    current_cov = best_cov.copy()
    current_size = len(current_selected)
    
    # Random restarts to escape local minima
    rng = np.random.default_rng(42)
    
    for restart in range(500):
        if time.time() > deadline - 5.0:
            break
            
        # Reset to current best for each restart to ensure we always improve or stay
        current_selected = list(global_best_selected) if 'global_best_selected' in locals() else best_selected.copy()
        # Actually, let's maintain a single global best and try to improve it.
        # We will run a local search on a copy of the global best.
        
        if 'global_best_selected' not in locals():
            global_best_selected = best_selected.copy()
            global_best_cov = get_coverage(global_best_selected)
            global_best_size = len(global_best_selected)
            
        cur_sel = list(global_best_selected)
        cur_cov = global_best_cov.copy()
        cur_size = len(cur_sel)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to remove redundant codewords
            # Iterate through all selected, try to remove any that are redundant
            # To make it random, shuffle the order
            order = list(range(len(cur_sel)))
            rng.shuffle(order)
            
            for i in order:
                if i >= len(cur_sel): continue
                idx = cur_sel[i]
                if can_remove(idx, cur_cov):
                    # Remove it
                    cur_cov[neighbors[idx]] -= 1
                    cur_sel.pop(i)
                    cur_size -= 1
                    improved = True
                    # Continue checking from the beginning (or continue loop, but list changed)
                    # Break to restart the removal loop to check new redundancies
                    break
            
            if cur_size < global_best_size:
                global_best_size = cur_size
                global_best_selected = cur_sel.copy()
                global_best_cov = cur_cov.copy()
                
            if cur_size <= 71: # Lower bound
                break
            
            # 2. If no removal possible, try a swap
            # Pick a random codeword to remove
            if len(cur_sel) > 0:
                rem_idx_pos = rng.integers(len(cur_sel))
                rem_idx = cur_sel[rem_idx_pos]
                
                # Tentatively remove
                cur_cov[neighbors[rem_idx]] -= 1
                cur_sel.pop(rem_idx_pos)
                cur_size -= 1
                
                # Now we have a hole. We need to add a codeword that covers the uncovered parts.
                # Ideally, we want to add a codeword that allows further removals.
                # Let's find the most "urgent" uncovered word.
                # Actually, just find any codeword that covers the most currently uncovered (or under-covered) words.
                
                # Words with coverage 0 are critical.
                # Words with coverage 1 are fragile.
                
                # Let's try to add a random codeword first, or a greedy one.
                # Greedy: add the codeword that maximizes coverage of words with count 0.
                
                # If there are words with count 0, we MUST cover them.
                zero_cov_mask = (cur_cov == 0)
                
                if np.any(zero_cov_mask):
                    # Find candidate that covers most zero-cov words
                    # counts_zero[c] = number of neighbors of c that have cov 0
                    counts_zero = np.sum(zero_cov_mask[neighbors], axis=1)
                    
                    # Only consider candidates not already in cur_sel
                    in_sel_mask = np.zeros(N, dtype=bool)
                    in_sel_mask[cur_sel] = True
                    counts_zero[in_sel_mask] = -1
                    
                    add_idx = np.argmax(counts_zero)
                    
                    if counts_zero[add_idx] <= 0:
                        # Cannot fix? Revert
                        cur_sel.insert(rem_idx_pos, rem_idx)
                        cur_cov[neighbors[rem_idx]] += 1
                        cur_size += 1
                        continue
                else:
                    # No zero cov words. We can try to add a codeword that helps with future removals.
                    # Or just pick a random one?
                    # Let's pick one that is not in the set.
                    in_sel_mask = np.zeros(N, dtype=bool)
                    in_sel_mask[cur_sel] = True
                    candidates = np.where(~in_sel_mask)[0]
                    if len(candidates) == 0:
                        cur_sel.insert(rem_idx_pos, rem_idx)
                        cur_cov[neighbors[rem_idx]] += 1
                        cur_size += 1
                        continue
                    add_idx = candidates[rng.integers(len(candidates))]
                
                # Add the chosen codeword
                cur_sel.append(add_idx)
                cur_cov[neighbors[add_idx]] += 1
                cur_size += 1
                
                # Now try to remove redundant codewords again
                # This might result in a net reduction if the new codeword covers more than the one removed.
                # We don't need to immediately check for improvement, the next loop iteration will handle removals.
                # But we should check if we improved the global best? No, size is same or +1.
                # The value is in the subsequent removals.
                
                # To avoid infinite loops of same swaps, we could track visited states, but that's expensive.
                # The randomization helps.

    # Final verification and return
    # Ensure the best solution is valid
    final_sel = global_best_selected
    # Verify coverage
    final_cov = get_coverage(final_sel)
    if np.any(final_cov == 0):
        # Fallback to greedy result if local search broke it (shouldn't happen with careful reverts)
        # Or just use the greedy result
        final_sel = best_selected
        final_cov = get_coverage(final_sel)
        # If even greedy is bad (shouldn't be), return all words
        if np.any(final_cov == 0):
             return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Convert indices to codewords
    codewords = []
    for idx in final_sel:
        word = [(idx // powers[p]) % q for p in range(n)]
        codewords.append(word)
        
    return codewords
