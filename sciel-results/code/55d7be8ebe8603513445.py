import numpy as np
import time
import itertools

def construct(q, n, radius):
            # q=3, n=7, radius=1
            if q != 3 or n != 7 or radius != 1:
                # Fallback for other parameters (though task specifies 7 matches ternary)
                # Use seed logic for general case if possible, or simple greedy
                pass 
            
            # Precomputation
            words = list(itertools.product(range(q), repeat=n))
            word_to_idx = {w: i for i, w in enumerate(words)}
            idx_to_word = {i: w for i, w in enumerate(words)}
            
            # Precompute balls
            # A ball of radius 1 around w contains w and all words at dist 1
            balls = []
            for w in words:
                ball = [w]
                for i in range(n):
                    for val in range(q):
                        if val == w[i]: continue
                        new_w = list(w)
                        new_w[i] = val
                        ball.append(tuple(new_w))
                balls.append([word_to_idx[b] for b in ball])
            balls = np.array(balls, dtype=np.int32)
            
            # Greedy Set Cover
            coverage_count = np.zeros(len(words), dtype=np.int32)
            C = []
            
            # Optimization: Only consider centers that are not already covered? No, centers can be anywhere.
            # But if a center is already covered, it might still be useful to add if it covers many *other* uncovered words.
            # Actually, in set cover, we pick the set with max marginal gain.
            
            deadline = time.time() + 100 # 20s buffer
            
            # Greedy Phase
            while np.any(coverage_count == 0) and time.time() < deadline - 20:
                # Calculate gain for all candidates
                # gain[c] = number of words in balls[c] that are currently uncovered
                # uncovered_mask = (coverage_count == 0)
                # We can vectorize this:
                # For each ball, count how many elements are in the uncovered set.
                # balls is (2187, 15). coverage_count is (2187,).
                # We want sum(coverage_count[balls[i]] == 0) for each i.
                
                # This is a gather operation.
                # balls_indices = balls # (2187, 15)
                # current_counts = coverage_count[balls_indices] # (2187, 15)
                # gains = np.sum(current_counts == 0, axis=1)
                
                current_counts = coverage_count[balls]
                gains = np.sum(current_counts == 0, axis=1)
                
                # Pick best
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break # Should not happen if not fully covered
                
                C.append(best_idx)
                # Update coverage
                coverage_count[balls[best_idx]] += 1
            
            # Local Search Phase
            best_C = C[:]
            best_size = len(C)
            
            # Helper to check if removing c is valid
            # We maintain coverage_count.
            # To check if c is redundant:
            # 1. Decrement counts for balls[c]
            # 2. Check if any count becomes -1 (i.e., was 0)
            # 3. If yes, revert and not redundant. If no, keep removal.
            
            # To allow "swaps" to escape local minima:
            # If no removal is possible, try a random swap.
            
            while time.time() < deadline - 5:
                found_removal = False
                # Try to remove a random codeword
                # Shuffle C to try different orders? Or just iterate?
                # Iterating is deterministic. Randomizing helps.
                indices = np.random.permutation(len(C))
                
                for i in indices:
                    c = C[i]
                    # Check redundancy
                    ball_c = balls[c]
                    # Check if any word in ball_c has count 1 (will become 0)
                    # Actually, if count is 0, it's already uncovered (shouldn't happen).
                    # If count is 1, removing c makes it 0.
                    # We need to ensure NO word becomes 0.
                    # So we check if np.any(coverage_count[ball_c] == 1)
                    # If True, removing c uncovers some words. Not redundant.
                    # If False, removing c is safe.
                    
                    if not np.any(coverage_count[ball_c] == 1):
                        # Redundant
                        coverage_count[ball_c] -= 1
                        C.pop(i)
                        found_removal = True
                        best_size = len(C)
                        break # Restart search with new state
                
                if not found_removal:
                    # Try a random swap to shake
                    if len(C) > 0:
                        i = np.random.randint(len(C))
                        c_remove = C[i]
                        
                        # Pick random c_add not in C
                        # To be efficient, pick random from 0..2186 and check if in C
                        # C is a list, so `in` is O(N). N~200. 2187 checks is fine.
                        # Or maintain a set of C for O(1) lookup.
                        C_set = set(C)
                        while True:
                            c_add = np.random.randint(len(words))
                            if c_add not in C_set:
                                break
                        
                        # Check if swap is valid
                        # New set = (C \ {c_remove}) U {c_add}
                        # We need to check if all words are covered by this new set.
                        # This is expensive to check from scratch.
                        # Approximation: Check if the "damage" from removing c_remove is covered by c_add.
                        # Damage: Words in Ball(c_remove) that are covered ONLY by c_remove.
                        # If Ball(c_add) covers all these damage words, and c_add doesn't uncover anything else (it can't, it's adding),
                        # then the swap is valid?
                        # No, we also need to ensure that words NOT in Ball(c_remove) remain covered.
                        # Since we only remove c_remove, words not in Ball(c_remove) are unaffected.
                        # So we only need to check:
                        # 1. Are all words in Ball(c_remove) that were covered ONLY by c_remove now covered by c_add?
                        # 2. (Implicitly) Are all other words still covered? Yes, because we didn't touch their covering codewords.
                        
                        # So:
                        # unique_to_c_remove = words w in Ball(c_remove) such that coverage_count[w] == 1
                        # If unique_to_c_remove is empty, c_remove is redundant (handled above).
                        # If not empty, check if all w in unique_to_c_remove are in Ball(c_add).
                        
                        unique_words = [w for w in balls[c_remove] if coverage_count[w] == 1]
                        if unique_words:
                            # Check if c_add covers all unique_words
                            ball_add_set = set(balls[c_add])
                            if all(w in ball_add_set for w in unique_words):
                                # Swap is valid
                                # Update state
                                coverage_count[balls[c_remove]] -= 1
                                coverage_count[balls[c_add]] += 1
                                C[i] = c_add
                                C_set.remove(c_remove)
                                C_set.add(c_add)
                                # Note: Size doesn't change, but structure might improve
                        else:
                            # This case shouldn't happen if we checked redundancy first, 
                            # but if we did, we would have removed it.
                            pass

            # Final check and conversion to list of lists
            # Ensure C is valid (paranoia)
            # Convert indices to tuples/words
            
            return [list(idx_to_word[c]) for c in C]
