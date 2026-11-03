import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17

        # Precompute powers of 3
        pow3 = [3 ** i for i in range(n + 1)]

        def int_to_word(idx):
            w = []
            for _ in range(n):
                w.append(idx % 3)
                idx //= 3
            return tuple(reversed(w))

        def word_to_int(w):
            result = 0
            for d in w:
                result = result * 3 + d
            return result

        # Build sphere index matrix: shape (6561, 17)
        # spheres[i] contains the indices of the 17 words covered by codeword i
        spheres = np.empty((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
            w = int_to_word(idx)
            spheres[idx, 0] = idx
            k = 1
            for pos in range(n):
                orig = w[pos]
                base = pow3[n - 1 - pos]
                for d in range(q):
                    if d != orig:
                        spheres[idx, k] = idx + (d - orig) * base
                        k += 1

        # Precompute coverers: for each word u, list of codewords c such that u in spheres[c]
        # This is the inverse of the sphere relation.
        # coverers[u] is a list of indices.
        coverers = [[] for _ in range(total_space)]
        for c in range(total_space):
            for u in spheres[c]:
                coverers[u].append(c)
        
        # Convert to numpy arrays for faster access if possible, but lists of lists are okay for small intersections
        # Actually, for fast intersection, we might want sets, but sets are slow to create.
        # We will use lists and check membership.

        def randomized_greedy_init():
            uncovered = np.ones(total_space, dtype=bool)
            uncovered_count = total_space
            code_indices = []
            
            # Precompute counts for all candidates initially? 
            # No, update incrementally.
            
            # To speed up greedy, we can just pick the best uncovered word, then pick a random codeword covering it?
            # Or pick the codeword that covers the most uncovered words.
            
            # Efficient Greedy:
            # 1. Find an uncovered word u.
            # 2. Among all codewords covering u, pick the one that covers the most other uncovered words.
            
            while uncovered_count > 0:
                if time.time() > deadline - 100: # Leave time for search
                    break
                
                # Pick a random uncovered word to anchor the search
                # This helps in randomization
                uncovered_indices = np.where(uncovered)[0]
                if len(uncovered_indices) == 0:
                    break
                anchor_u = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
                
                # Candidates are codewords covering anchor_u
                candidates = coverers[anchor_u]
                
                best_c = -1
                best_gain = -1
                
                # Evaluate candidates
                # To speed up, we can sample if candidates is large? 
                # coverers[u] has size roughly 17 * (total_space / sphere_size) ? 
                # No, each word is covered by 17 codewords? 
                # Wait. The relation is symmetric? 
                # If u is in sphere(c), is c in sphere(u)? 
                # Yes, Hamming distance is symmetric. 
                # So coverers[u] is exactly spheres[u]. 
                # So len(coverers[u]) is always 17.
                
                # So we only check 17 candidates!
                for c in candidates:
                    # Count how many uncovered words c covers
                    gain = int(np.sum(uncovered[spheres[c]]))
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                        
                if best_c >= 0 and best_gain > 0:
                    code_indices.append(best_c)
                    uncovered[spheres[best_c]] = False
                    uncovered_count -= best_gain
                else:
                    # Should not happen if uncovered_count > 0
                    break
                    
            return code_indices

        def local_search(code_indices, max_time=None):
            if max_time is None:
                max_time = deadline - 10
                
            # Convert to list for easy removal
            code_set = list(code_indices)
            m = len(code_set)
            
            # Build cover_count array
            cover_count = np.zeros(total_space, dtype=np.int32)
            for c in code_set:
                cover_count[spheres[c]] += 1
            
            # Map index to position in code_set for O(1) removal?
            # Or just use list.remove which is O(n). m is ~400, so O(n) is fine.
            
            improved = True
            while improved and time.time() < max_time:
                improved = False
                # Shuffle to avoid deterministic cycles?
                random.shuffle(code_set)
                
                for i in range(len(code_set)):
                    if time.time() > max_time:
                        return code_set, cover_count, False
                    
                    c = code_set[i]
                    
                    # Try to remove c
                    cover_count[spheres[c]] -= 1
                    
                    # Check for uncovered words
                    # Instead of scanning whole array, we can track it?
                    # But scanning 6561 ints is fast (~0.01ms).
                    uncovered_mask = cover_count == 0
                    uncovered_words = np.where(uncovered_mask)[0]
                    
                    if len(uncovered_words) == 0:
                        # Success! Remove c
                        code_set.pop(i)
                        improved = True
                        # No need to update cover_count further as we already decremented
                        # But we popped from code_set, so indices shift? 
                        # We are iterating by index, so popping shifts indices.
                        # To be safe, break and restart loop or adjust index.
                        # Let's just break and restart the while loop to be safe with indices.
                        # Actually, if we pop i, the next element is at i. 
                        # So we should not increment i.
                        # But we are in a for loop.
                        # Let's use a while loop for the index.
                        # Or simply, if we remove, we break out of the for loop and restart the while loop.
                        break
                    else:
                        # Coverage lost. Try to repair.
                        # Find c_new that covers all uncovered_words.
                        # c_new must be in intersection of coverers[u] for u in uncovered_words.
                        
                        # Since len(uncovered_words) is usually small (1-3), we can find intersection.
                        # Start with the list of coverers for the first uncovered word.
                        candidates = set(coverers[uncovered_words[0]])
                        
                        for u in uncovered_words[1:]:
                            candidates.intersection_update(coverers[u])
                            if not candidates:
                                break
                            
                        if candidates:
                            # Pick a random candidate from the intersection
                            # Exclude c itself? If c is in candidates, adding it back just reverts.
                            # If c is the only candidate, we must revert.
                            # If there are others, we can swap.
                            cand_list = list(candidates)
                            if len(cand_list) > 1:
                                # Remove c from consideration if present
                                if c in cand_list:
                                    cand_list.remove(c)
                                if cand_list:
                                    c_new = random.choice(cand_list)
                                    # Add c_new
                                    cover_count[spheres[c_new]] += 1
                                    # Verify coverage restored
                                    # If yes, we have swapped c with c_new.
                                    # Size is same, but structure changed.
                                    # We accept this swap to allow future deletions.
                                    # Replace c with c_new in code_set
                                    code_set[i] = c_new
                                    improved = True
                                    # Break and restart to avoid index issues
                                    break
                                    # Note: We did NOT remove c from code_set, we replaced it.
                                    # And we did NOT decrement cover_count for c_new (we incremented).
                                    # Wait, we decremented cover_count for c at the start.
                                    # So net change: -1 for c, +1 for c_new.
                                    # This is correct for a swap.
                            else:
                                # Only c covers the uncovered words. Revert.
                                cover_count[spheres[c]] += 1
                        else:
                            # No single codeword covers all uncovered words.
                            # Revert.
                            cover_count[spheres[c]] += 1

            return code_set, cover_count, False

        def perturb(code_set, num_remove=5):
            # Remove num_remove random codewords
            # and re-add them using greedy logic on the uncovered spots?
            # Or just random re-add?
            # Let's remove and then run a mini-greedy to fill the gaps.
            
            m = len(code_set)
            if m < num_remove:
                return code_set
                
            indices_to_remove = random.sample(range(m), num_remove)
            # Sort indices to remove cleanly
            indices_to_remove.sort(reverse=True)
            
            # We need to track cover_count to know what's uncovered.
            # But we don't have cover_count here.
            # Let's just remove them from the set.
            # The local_search will handle the rest?
            # No, local_search assumes a valid code.
            # So perturbation must return a valid code.
            
            # Better Perturbation:
            # 1. Remove 5 random codewords.
            # 2. This creates holes.
            # 3. Use Greedy to fill the holes (minimize additions).
            # 4. The result is a valid code, possibly larger.
            # 5. Then Local Search will try to shrink it.
            
            # To do this efficiently, we need cover_count.
            # Let's rebuild cover_count for the perturbed set.
            
            new_set = list(code_set)
            for idx in indices_to_remove:
                new_set.pop(idx)
                
            # Rebuild cover_count
            cover_count = np.zeros(total_space, dtype=np.int32)
            for c in new_set:
                cover_count[spheres[c]] += 1
                
            # Greedy fill
            uncovered = cover_count == 0
            while np.any(uncovered):
                if time.time() > deadline - 10:
                    break
                u_idx = np.where(uncovered)[0]
                u = u_idx[random.randint(0, len(u_idx) - 1)]
                candidates = coverers[u]
                best_c = -1
                best_gain = -1
                for c in candidates:
                    gain = int(np.sum(uncovered[spheres[c]]))
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                if best_c >= 0:
                    new_set.append(best_c)
                    cover_count[spheres[best_c]] += 1
                    uncovered = cover_count == 0
                else:
                    break
                    
            return new_set

        # Main Logic
        best_code = None
        best_size = float('inf')
        
        # Initial Greedy
        init_code = randomized_greedy_init()
        best_size = len(init_code)
        best_code = init_code
        
        # Iterated Local Search
        current_code = init_code
        
        while time.time() < deadline - 5:
            # Local Search
            optimized_code, _, _ = local_search(current_code, max_time=deadline - 5)
            
            if len(optimized_code) < best_size:
                best_size = len(optimized_code)
                best_code = optimized_code
                # Restart local search from this new best?
                # Or just continue.
                current_code = optimized_code
            else:
                # Perturb
                current_code = perturb(optimized_code, num_remove=5)
                # If perturbation made it much worse, maybe revert to best?
                # No, ILS usually accepts worse solutions to escape.
                
        # Final check and return
        if best_code is None:
            best_code = init_code
            
        # Verify best_code
        # Convert to list of words
        final_indices = best_code
        final_code = [int_to_word(idx) for idx in final_indices]
        
        # Verify coverage
        verify_cover = np.zeros(total_space, dtype=bool)
        for idx in final_indices:
            verify_cover[spheres[idx]] = True
        if np.all(verify_cover):
            return [list(w) for w in final_code]
        else:
            # Fallback
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    else:
        # General case fallback
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
