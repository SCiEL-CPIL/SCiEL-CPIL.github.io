"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute covers (balls of radius 1)
    covers = []
    for idx in range(N):
        w = all_words[idx]
        cover_set = [idx]
        for i in range(n):
            for alt in range(q):
                if alt == w[i]:
                    continue
                nw = list(w)
                nw[i] = alt
                cover_set.append(word_to_idx[tuple(nw)])
        covers.append(cover_set)
        
    covers_np = [np.array(c, dtype=np.int32) for c in covers]
    
    # Build Incidence Matrix A
    # A[i, j] = 1 if word j is in covers[i]
    A = np.zeros((N, N), dtype=np.int8)
    for i, cover_set in enumerate(covers):
        A[i, cover_set] = 1
        
    # Greedy Set Cover Phase
    covered = np.zeros(N, dtype=bool)
    selected_indices = []
    
    while not covered.all():
        if time.time() > deadline - 15.0: # Reserve 15s for local search
            break
            
        uncovered = ~covered
        # scores[i] = number of uncovered words covered by codeword i
        scores = A @ uncovered
        # Mask out already selected
        if selected_indices:
            scores[selected_indices] = 0
            
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        covered[covers_np[best_idx]] = True
        
    # Ensure validity if greedy broke early
    if not covered.all():
        while not covered.all() and time.time() < deadline - 5.0:
            uncovered = ~covered
            scores = A @ uncovered
            if selected_indices:
                scores[selected_indices] = 0
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            covered[covers_np[best_idx]] = True
            
    if not covered.all():
        # Fallback: Add random codewords
        remaining = np.where(~covered)[0]
        while remaining.size > 0 and time.time() < deadline - 2.0:
            w_idx = random.choice(remaining)
            c_idx = random.choice(covers[w_idx])
            if c_idx not in selected_indices:
                selected_indices.append(c_idx)
                covered[covers_np[c_idx]] = True
                remaining = np.where(~covered)[0]
                
    # Local Search Phase
    if time.time() < deadline - 5.0:
        # Use boolean mask for selection
        is_selected = np.zeros(N, dtype=bool)
        for idx in selected_indices:
            is_selected[idx] = True
            
        # Calculate initial counts
        # counts[w] = number of selected codewords covering w
        # We want column sums of A for selected rows
        # A is (N, N), is_selected is (N,)
        # counts = A.T @ is_selected
        counts = A.T @ is_selected
        counts = counts.astype(np.int32)
        
        # Helper to check if a codeword can be removed
        # A codeword c can be removed if all words it covers are covered by >= 2 codewords
        # i.e., min(counts[covers[c]]) >= 2
        
        # Greedy Removal Pass (Optimized)
        def greedy_remove_pass():
            changed = True
            while changed:
                changed = False
                # Iterate over a copy of the list
                # We can optimize by checking candidates likely to be removable
                # But simple iteration is okay for now
                for idx in list(selected_indices):
                    # Check if removable
                    # If min count in its cover is >= 2
                    if np.min(counts[covers_np[idx]]) >= 2:
                        # Remove it
                        is_selected[idx] = False
                        selected_indices.remove(idx)
                        # Update counts
                        counts[covers_np[idx]] -= 1
                        changed = True

        greedy_remove_pass()
        
        # Local Search Swaps (1-exchange)
        # Try to swap a codeword c with c' to reduce size or enable removal
        # We look for codewords c that are "critical" (cover some words with count 1)
        # and try to find a replacement c' that covers those words and is not selected.
        
        # To speed up, we can focus on codewords that cover many critical words.
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 2.0 and no_improve_count < max_no_improve:
            # Find critical words (count == 1)
            critical_words = np.where(counts == 1)[0]
            if len(critical_words) == 0:
                # No critical words, try more removals?
                # If no critical words, all counts >= 2.
                # Greedy removal should have cleared them?
                # Unless removal creates new critical words? No, removal decreases counts.
                # If all counts >= 2, we can remove any codeword?
                # Wait, if I remove c, counts[covers[c]] become 1.
                # So we can remove c if min(counts[covers[c]]) >= 2.
                # If no critical words, min(counts) >= 2.
                # So we can remove ANY codeword?
                # Let's run greedy removal again just in case.
                greedy_remove_pass()
                if not selected_indices:
                    break
                # If still no critical words, we are in a state where we can remove more?
                # Or we are stuck in a local minimum where we can't remove?
                # If min(counts) >= 2, we CAN remove.
                # So greedy_remove_pass should have removed something.
                # If it didn't, it means no codeword satisfies min(counts[covers[c]]) >= 2?
                # That implies for every c, there is a word w in covers[c] with counts[w] == 1.
                # So there MUST be critical words.
                # Contradiction. So if len(critical_words)==0, we should have removed something.
                # Let's break to avoid infinite loop if logic fails.
                break
            
            # Pick a random critical word
            w_idx = random.choice(critical_words)
            
            # Find the unique codeword covering w_idx
            # candidates = covers[w_idx]
            # covering_c is the one in is_selected
            covering_c = -1
            for c in covers[w_idx]:
                if is_selected[c]:
                    covering_c = c
                    break
            
            if covering_c == -1:
                continue
                
            # Find a replacement c' not in cover that covers w_idx
            # c' must be in covers[w_idx] and not in is_selected
            replacements = [c for c in covers[w_idx] if not is_selected[c]]
            if not replacements:
                continue
                
            new_c = random.choice(replacements)
            
            # Tentative Swap
            # Remove covering_c, Add new_c
            # Update counts
            counts[covers_np[covering_c]] -= 1
            counts[covers_np[new_c]] += 1
            
            # Check validity: no counts should be 0
            if np.any(counts == 0):
                # Revert
                counts[covers_np[new_c]] -= 1
                counts[covers_np[covering_c]] += 1
            else:
                # Accept swap
                is_selected[covering_c] = False
                is_selected[new_c] = True
                selected_indices.remove(covering_c)
                selected_indices.append(new_c)
                
                # Try to remove more
                old_size = len(selected_indices)
                greedy_remove_pass()
                new_size = len(selected_indices)
                
                if new_size < old_size:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # If no improvement, we might want to revert the swap?
                    # Or just keep it and try another?
                    # Keeping it might help escape local minima.
                    # But if we just swap and don't reduce size, we are just moving.
                    # Let's keep it.

    # Return result
    return [all_words[idx].tolist() for idx in selected_indices]

# EVOLVE-BLOCK-END
