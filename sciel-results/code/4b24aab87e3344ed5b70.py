import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Build the direct-sum construction
    # Tetracode is a perfect 1-error-correcting code of length 4 over GF(3)
    # It has 9 words.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Construct initial code: direct sum with all tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))
            
    m = len(code) # 81 for q=3, n=6
    total_words = q ** n
    
    # Precompute word indices and coverage masks/indices for ALL possible words
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    index_to_word = [None] * total_words
    word_to_idx_map = {}
    
    for idx, w in enumerate(all_words):
        index_to_word[idx] = w
        word_to_idx_map[w] = idx
        
    # Precompute coverage masks and index lists for all possible words
    cover_masks = [0] * total_words
    cover_indices = [[] for _ in range(total_words)]
    
    for idx, w in enumerate(all_words):
        mask = 1 << idx
        cover_indices[idx].append(idx)
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_list[pos] = new_val
                n_idx = word_to_idx_map[tuple(w_list)]
                mask |= (1 << n_idx)
                cover_indices[idx].append(n_idx)
            w_list[pos] = orig_val
        cover_masks[idx] = mask
        
    # Map initial code to indices
    initial_indices = [word_to_idx_map[c] for c in code]
    
    best_indices = initial_indices[:]
    best_count = len(best_indices)
    
    # Local Search Function
    def local_search(start_indices):
        active_set = set(start_indices)
        cov_count = np.zeros(total_words, dtype=np.int32)
        
        # Initialize coverage
        for idx in active_set:
            for n_idx in cover_indices[idx]:
                cov_count[n_idx] += 1
                
        # 1. Greedy Removal of Redundant Codewords
        removed_any = True
        while removed_any:
            removed_any = False
            # Shuffle to handle ties randomly
            active_list = list(active_set)
            random.shuffle(active_list)
            for idx in active_list:
                if idx not in active_set:
                    continue
                # Check if redundant
                redundant = True
                for n_idx in cover_indices[idx]:
                    if cov_count[n_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    active_set.remove(idx)
                    for n_idx in cover_indices[idx]:
                        cov_count[n_idx] -= 1
                    removed_any = True
        
        # 2. Swap Local Search (Remove-Add)
        # We try to swap an active codeword with an inactive one if it reduces size or maintains size but improves "potential"
        # Since we want to minimize size, we only swap if we can remove one and add one, but that keeps size same.
        # Wait, the goal is to REDUCE size.
        # Greedy removal already reduced size as much as possible.
        # To reduce further, we need a "2-exchange" or similar?
        # No, standard covering code optimization often involves:
        # If we can't remove any single word, we might need to swap one word for another to enable a future removal.
        # Or, we start with a larger set and remove.
        # The initial set is 81. The optimal is 73.
        # If greedy removal gets stuck at 81, it means no single word is redundant.
        # We need to perform swaps that *enable* removals.
        
        improved = True
        while improved:
            improved = False
            active_list = list(active_set)
            random.shuffle(active_list)
            
            for idx_out in active_list:
                if idx_out not in active_set:
                    continue
                
                # Determine words that would become uncovered if we remove idx_out
                # These are words where cov_count == 1
                deficit_mask = 0
                for n_idx in cover_indices[idx_out]:
                    if cov_count[n_idx] == 1:
                        deficit_mask |= (1 << n_idx)
                        
                if deficit_mask == 0:
                    # This shouldn't happen if greedy removal ran to completion, 
                    # but if it did, just remove it.
                    active_set.remove(idx_out)
                    for n_idx in cover_indices[idx_out]:
                        cov_count[n_idx] -= 1
                    improved = True
                    break
                
                # Find a candidate idx_in that covers all deficit words
                # We only consider inactive codewords
                # To be fast, we can iterate all inactive and check if (cover_masks[idx_in] & deficit_mask) == deficit_mask
                # This is a subset check.
                
                # Optimization: Only try candidates that cover at least one deficit word?
                # The check is cheap enough with bitmasks.
                
                found_swap = False
                # Iterate inactive set
                # To avoid O(N^2) if N is large, we can just iterate all words not in active_set
                # N=729, Active~80. Inactive~650.
                # 80 * 650 = 52k checks per pass. Very fast.
                
                for idx_in in range(total_words):
                    if idx_in in active_set:
                        continue
                    
                    # Check if idx_in covers all deficit words
                    if (cover_masks[idx_in] & deficit_mask) == deficit_mask:
                        # Check if this swap is "beneficial".
                        # It maintains size. Does it help future removals?
                        # Heuristic: Prefer swaps that increase coverage counts of other words?
                        # Or just take the first one found?
                        # Let's take the one that covers the most *additional* words (overlap with current covered set is minimized? No, we want to maximize redundancy).
                        # Actually, any valid swap that allows a future removal is good.
                        # Let's just pick one randomly or the first one to keep it simple and fast.
                        
                        # Perform Swap
                        active_set.remove(idx_out)
                        for n_idx in cover_indices[idx_out]:
                            cov_count[n_idx] -= 1
                        
                        active_set.add(idx_in)
                        for n_idx in cover_indices[idx_in]:
                            cov_count[n_idx] += 1
                        
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
        
        # Final Greedy Removal after swaps
        removed_any = True
        while removed_any:
            removed_any = False
            active_list = list(active_set)
            random.shuffle(active_list)
            for idx in active_list:
                if idx not in active_set:
                    continue
                redundant = True
                for n_idx in cover_indices[idx]:
                    if cov_count[n_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    active_set.remove(idx)
                    for n_idx in cover_indices[idx]:
                        cov_count[n_idx] -= 1
                    removed_any = True
                    
        return list(active_set)

    # Run multiple restarts
    # Start with the initial code
    # Also try starting from random subsets? No, must be valid.
    # We can perturb the initial solution by random swaps and then run local search.
    
    current_indices = initial_indices[:]
    
    # First, run local search on the initial solution
    result = local_search(current_indices)
    if len(result) < best_count:
        best_count = len(result)
        best_indices = result[:]
        
    # Now, perform random restarts/perturbations
    # We can take the best solution found, remove a random word (making it invalid), 
    # then use a greedy repair to make it valid again, then optimize.
    # Or, simpler: Randomly swap 2 words in the best solution, then run local search.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Perturb best_indices
        perturbed = best_indices[:]
        if len(perturbed) >= 2:
            i1, i2 = random.sample(range(len(perturbed)), 2)
            # Swap i1 and i2? No, that doesn't change the set.
            # We need to change the set.
            # Remove one, add a random inactive one.
            idx_out = perturbed.pop(random.randrange(len(perturbed)))
            # Find a random inactive word
            active_set_pert = set(perturbed)
            # Pick a random word not in active_set_pert
            while True:
                idx_in = random.randrange(total_words)
                if idx_in not in active_set_pert:
                    break
            perturbed.append(idx_in)
            
        # The perturbed set might not be a valid cover.
        # local_search assumes valid input? 
        # My local_search implementation above assumes valid input for the "deficit" logic?
        # If input is invalid, cov_count will have zeros.
        # The "Greedy Removal" step might remove words that are needed?
        # If cov_count[n_idx] == 1, it removes it. If it was 0, it's not 1, so it won't remove it based on that check?
        # Wait, if cov_count is 0, the word is uncovered.
        # The "redundant" check `if cov_count[n_idx] == 1` will be False for 0.
        # So it won't remove words that are the *only* cover for a covered word.
        # But it won't fix uncovered words.
        # So we need a "Repair" step if we perturb.
        
        # Let's modify local_search to handle invalid inputs by adding words first?
        # Or, simpler: Only perturb by SWAPPING valid words?
        # If we swap two valid words, the set size is same, but coverage changes.
        # It might become invalid.
        
        # Alternative: Use the "Add-Remove" strategy directly.
        # 1. Start with best_indices.
        # 2. Randomly remove one word.
        # 3. Find the best word to add back to restore coverage (Greedy Set Cover step).
        # 4. Run local search.
        
        # Let's implement a repair function.
        
        def repair_and_optimize(start_set):
            active_set = set(start_set)
            cov_count = np.zeros(total_words, dtype=np.int32)
            for idx in active_set:
                for n_idx in cover_indices[idx]:
                    cov_count[n_idx] += 1
            
            # Repair: Add words until valid
            # Find uncovered words
            uncovered = [i for i in range(total_words) if cov_count[i] == 0]
            while uncovered:
                # Pick an uncovered word
                target = uncovered[0]
                # Find candidate that covers most uncovered words
                best_cand = -1
                best_score = -1
                # Iterate all words not in active_set
                for cand in range(total_words):
                    if cand in active_set:
                        continue
                    # Count how many uncovered words cand covers
                    score = 0
                    cand_mask = cover_masks[cand]
                    # Iterate uncovered list? Or use mask intersection?
                    # Let's use mask.
                    # uncovered_mask = 0
                    # for u in uncovered: uncovered_mask |= (1 << u)
                    # score = (cand_mask & uncovered_mask).bit_count()
                    
                    # To avoid building uncovered_mask every time, we can just count intersections with the list
                    # But list is small initially.
                    # Let's just check if it covers `target` and pick the one with highest overlap with current uncovered set.
                    
                    # Simpler: Just pick the first word that covers `target`.
                    # This is a valid greedy step.
                    if (cover_masks[cand] >> target) & 1:
                        best_cand = cand
                        break # Just take the first one to be fast?
                        # Better: take the one that covers the most uncovered.
                        # But for speed, first valid is okay.
                
                if best_cand == -1:
                    # Should not happen
                    break
                    
                active_set.add(best_cand)
                for n_idx in cover_indices[best_cand]:
                    cov_count[n_idx] += 1
                    if cov_count[n_idx] == 1:
                        # It was uncovered, now covered
                        # Remove from uncovered list? 
                        # Easier to just recalculate uncovered at end of loop or maintain set.
                        pass
                        
                # Update uncovered list efficiently?
                # Just recalculate if list is small?
                # Or maintain a set of uncovered.
                # Let's just recalculate uncovered for simplicity, N=729 is small.
                uncovered = [i for i in range(total_words) if cov_count[i] == 0]
                
            # Now run local search (Removal + Swaps)
            # Reuse the logic from before
            # ... (same as local_search but starting from this state)
            
            # To avoid code duplication, we can just call the main local_search logic 
            # but we need to pass the state.
            # Let's refactor local_search to accept active_set and cov_count.
            
            return active_set, cov_count

        # This is getting complex. Let's stick to the simpler "Perturb by Swap" which keeps size constant, 
        # then run the existing local_search which *assumes* valid input?
        # If we swap two words, the set might become invalid.
        # My local_search `deficit` logic:
        # `if cov_count[n_idx] == 1` -> deficit.
        # If cov_count is 0, it's not 1. So it won't flag it as deficit for removal.
        # So it won't remove the word causing the hole.
        # It will just get stuck.
        
        # So we MUST repair.
        
        # Let's implement a fast repair inside the loop.
        
        # 1. Perturb: Remove one random word from best_indices.
        # 2. Repair: Add the best word to cover the new holes.
        # 3. Optimize: Run local_search.
        
        # We can reuse the `local_search` function if we modify it to accept initial state?
        # Or just write a combined function.
        
        # Let's write a helper `optimize_set` that takes a set (valid or not) and returns optimized set.
        
        def optimize_set(init_set):
            active_set = set(init_set)
            cov_count = np.zeros(total_words, dtype=np.int32)
            for idx in active_set:
                for n_idx in cover_indices[idx]:
                    cov_count[n_idx] += 1
            
            # Repair Phase
            uncovered_mask = 0
            for i in range(total_words):
                if cov_count[i] == 0:
                    uncovered_mask |= (1 << i)
            
            while uncovered_mask:
                # Find word that covers most uncovered bits
                best_cand = -1
                best_count = -1
                # Sample a few candidates to be fast? Or all?
                # All is 729 checks. 729 * 120s budget is fine.
                for cand in range(total_words):
                    if cand in active_set:
                        continue
                    # Count overlap with uncovered_mask
                    # (cover_masks[cand] & uncovered_mask).bit_count()
                    # Python 3.8+ has int.bit_count()
                    cnt = (cover_masks[cand] & uncovered_mask).bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_cand = cand
                        if cnt == uncovered_mask.bit_count():
                            break # Perfect match found
                
                if best_cand == -1:
                    break # Should not happen
                    
                active_set.add(best_cand)
                for n_idx in cover_indices[best_cand]:
                    cov_count[n_idx] += 1
                    if cov_count[n_idx] == 1:
                        uncovered_mask &= ~(1 << n_idx)
                        
            # Optimization Phase (Same as before)
            # 1. Greedy Removal
            removed_any = True
            while removed_any:
                removed_any = False
                active_list = list(active_set)
                random.shuffle(active_list)
                for idx in active_list:
                    if idx not in active_set:
                        continue
                    redundant = True
                    for n_idx in cover_indices[idx]:
                        if cov_count[n_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        active_set.remove(idx)
                        for n_idx in cover_indices[idx]:
                            cov_count[n_idx] -= 1
                        removed_any = True
            
            # 2. Swap Search
            improved = True
            while improved:
                improved = False
                active_list = list(active_set)
                random.shuffle(active_list)
                for idx_out in active_list:
                    if idx_out not in active_set:
                        continue
                    deficit_mask = 0
                    for n_idx in cover_indices[idx_out]:
                        if cov_count[n_idx] == 1:
                            deficit_mask |= (1 << n_idx)
                            
                    if deficit_mask == 0:
                        active_set.remove(idx_out)
                        for n_idx in cover_indices[idx_out]:
                            cov_count[n_idx] -= 1
                        improved = True
                        break
                        
                    found_swap = False
                    for idx_in in range(total_words):
                        if idx_in in active_set:
                            continue
                        if (cover_masks[idx_in] & deficit_mask) == deficit_mask:
                            active_set.remove(idx_out)
                            for n_idx in cover_indices[idx_out]:
                                cov_count[n_idx] -= 1
                            active_set.add(idx_in)
                            for n_idx in cover_indices[idx_in]:
                                cov_count[n_idx] += 1
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                        
            # 3. Final Greedy Removal
            removed_any = True
            while removed_any:
                removed_any = False
                active_list = list(active_set)
                random.shuffle(active_list)
                for idx in active_list:
                    if idx not in active_set:
                        continue
                    redundant = True
                    for n_idx in cover_indices[idx]:
                        if cov_count[n_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        active_set.remove(idx)
                        for n_idx in cover_indices[idx]:
                            cov_count[n_idx] -= 1
                        removed_any = True
                        
            return active_set

        # Run initial optimization
        best_set = optimize_set(initial_indices)
        if len(best_set) < best_count:
            best_count = len(best_set)
            best_indices = list(best_set)
            
        # Main Loop
        while time.time() < deadline:
            # Perturb
            perturbed = best_indices[:]
            if len(perturbed) > 0:
                idx_out = perturbed.pop(random.randrange(len(perturbed)))
                # Add a random word
                # To ensure diversity, pick a word that is NOT in best_set
                # But we don't have best_set as a set here, just list.
                # Let's just pick random.
                idx_in = random.randrange(total_words)
                perturbed.append(idx_in)
                
            # Optimize
            new_set = optimize_set(perturbed)
            
            if len(new_set) < best_count:
                best_count = len(new_set)
                best_indices = list(new_set)
                
    # Convert to lists of ints
    return [list(index_to_word[idx]) for idx in best_indices]
