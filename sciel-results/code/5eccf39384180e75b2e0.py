import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    if radius != 1 or q != 3 or n != 8:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute sphere indices
    sphere_indices = [[] for _ in range(total_words)]
    for i, word in enumerate(all_words):
        inds = [i]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    j = word_to_idx[new_word]
                    inds.append(j)
        sphere_indices[i] = inds
        
    # Initial Greedy Cover
    # We use a list of uncovered words to speed up? 
    # For 6561, bitset is fine.
    uncovered_mask = (1 << total_words) - 1
    code = []
    cover_count = np.zeros(total_words, dtype=np.int8)
    
    # Fast Greedy using bitsets
    # Precompute sphere masks
    sphere_masks = [0] * total_words
    for i in range(total_words):
        m = 0
        for j in sphere_indices[i]:
            m |= (1 << j)
        sphere_masks[i] = m

    while uncovered_mask:
        best_idx = -1
        best_count = 0
        # Optimization: Only check indices that intersect uncovered_mask
        # But we don't have an inverted index. 
        # Given 6561 is small, linear scan is okay.
        for i in range(total_words):
            if sphere_masks[i] & uncovered_mask:
                count = (sphere_masks[i] & uncovered_mask).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
        
        if best_idx == -1:
            break # Should not happen if radius covers everything
            
        code.append(best_idx)
        uncovered_mask &= ~sphere_masks[best_idx]
        for j in sphere_indices[best_idx]:
            cover_count[j] += 1
            
    best_size = len(code)
    
    # Local Search
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < 100:
        improved = False
        
        # Phase 1: Remove Redundant Codewords
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                c_idx = code[i]
                # Check if any word in sphere(c_idx) has count 1
                redundant = True
                for w_idx in sphere_indices[c_idx]:
                    if cover_count[w_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for w_idx in sphere_indices[c_idx]:
                        cover_count[w_idx] -= 1
                    code.pop(i)
                    best_size = len(code)
                    improved = True
                    changed = True
                    break # Restart scan as indices changed
        
        # Phase 2: Swap/Repair Local Search
        # Try to remove a random codeword and repair
        for _ in range(100): # Limit iterations per phase
            if time.time() - start_time > 100:
                break
            if not code:
                break
            i = random.randint(0, len(code) - 1)
            c_idx = code[i]
            
            # Temporarily remove
            for w_idx in sphere_indices[c_idx]:
                cover_count[w_idx] -= 1
            
            # Find uncovered
            unc_mask = 0
            # Efficiently find uncovered? 
            # We can just check if any count is 0.
            # But we need the mask for greedy repair.
            # Rebuilding mask from scratch is O(N).
            # Alternative: Keep a list of uncovered indices?
            # Let's just build the mask.
            unc_indices = []
            for w in range(total_words):
                if cover_count[w] == 0:
                    unc_indices.append(w)
            
            if not unc_indices:
                # We removed a redundant word without knowing it was redundant? 
                # This means Phase 1 missed it or it became redundant due to previous changes.
                # In any case, size decreased.
                code.pop(i)
                best_size = len(code)
                improved = True
                break # Restart local search loop
            
            # Greedily repair
            unc_mask = 0
            for w in unc_indices:
                unc_mask |= (1 << w)
                
            # Pick best replacement
            best_repl = -1
            best_count = 0
            for j in range(total_words):
                if j in code: # Optional: prefer new words? No, any word.
                    pass
                count = (sphere_masks[j] & unc_mask).bit_count()
                if count > best_count:
                    best_count = count
                    best_repl = j
                    
            if best_repl == -1:
                # Restore
                for w_idx in sphere_indices[c_idx]:
                    cover_count[w_idx] += 1
                continue
                
            # Apply replacement
            code[i] = best_repl
            for w_idx in sphere_indices[best_repl]:
                cover_count[w_idx] += 1
                
            # Check if we can remove more now?
            # The loop will handle it in Phase 1 next iteration.
            # But we might have increased size? No, we swapped 1 for 1.
            # However, the new word might make others redundant.
            # We don't know if this swap is "better" yet. 
            # It's only better if it allows subsequent removals.
            # To accept this swap, we should try to reduce again.
            # If reduction succeeds, accept. If not, revert?
            # Simpler: Just keep the state. Phase 1 will clean up.
            # If Phase 1 doesn't find anything, we are stuck at same size.
            # We should revert if no improvement in size is found after a few tries.
            # For now, let's just let it run.
            
            # To avoid cycling, we can track best size.
            # If we don't improve size, we might want to revert.
            # But "Swap" is a neutral move.
            # Let's try to reduce immediately.
            # (Already done in Phase 1 at top of loop, but we are inside Phase 2).
            # Let's do a quick reduction check.
            reduced = False
            for k in range(len(code)):
                c_k = code[k]
                redundant = True
                for w_idx in sphere_indices[c_k]:
                    if cover_count[w_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for w_idx in sphere_indices[c_k]:
                        cover_count[w_idx] -= 1
                    code.pop(k)
                    best_size = len(code)
                    improved = True
                    reduced = True
                    break
            if reduced:
                break # Restart local search loop
        
    return [list(all_words[idx]) for idx in code]
