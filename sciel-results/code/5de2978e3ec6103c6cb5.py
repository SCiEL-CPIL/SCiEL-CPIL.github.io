"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17
        
        # Precompute all spheres using bitmask representation
        # Each word in {0,1,2}^8 maps to an index 0..6560
        # Precompute for each index, which indices are within distance 1
        def word_to_int(w):
            """Convert a tuple of digits to an integer (base-3 encoding)."""
            result = 0
            for d in w:
                result = result * 3 + d
            return result
        
        def int_to_word(idx):
            """Convert an integer back to a tuple of digits (base-3)."""
            w = []
            for _ in range(n):
                w.append(idx % 3)
                idx //= 3
            return tuple(reversed(w))
        
        # Precompute sphere masks: for each word index, the list of indices it covers
        # This is a list of 6561 lists, each with 17 elements
        spheres = []
        for idx in range(total_space):
            w = int_to_word(idx)
            covered = [idx]
            for pos in range(n):
                for d in range(q):
                    if d != w[pos]:
                        new_w = list(w)
                        new_w[pos] = d
                        covered.append(word_to_int(tuple(new_w)))
            spheres.append(covered)
        
        # Greedy set cover: repeatedly pick the codeword that covers the most uncovered words
        uncovered = np.ones(total_space, dtype=bool)
        uncovered_count = total_space
        code_indices = []
        
        # For efficiency, we'll maintain an array of how many times each word is covered
        # by the selected codewords, and use a priority-based selection
        
        # Precompute: for each candidate word, its sphere as a numpy array
        sphere_arrays = [np.array(s, dtype=np.int32) for s in spheres]
        
        # Greedy selection (vectorized)
        # Precompute sphere indices as a 2D array: sphere_idx_2d[i, j] = j-th word in sphere of word i
        sphere_idx_2d = np.array(spheres, dtype=np.int32)  # shape (6561, 17)
        
        # Use a slightly randomized greedy to potentially find better solutions
        greedy_rng = random.Random(123)
        
        while uncovered_count > 0:
            if time.time() > deadline - 30:  # Leave time for local search
                break
            
            # For each candidate word, count how many of its sphere words are uncovered
            # uncovered[sphere_idx_2d[i, :]] gives the uncovered status for sphere i
            # We want: for each i, sum(uncovered[sphere_idx_2d[i, :]])
            # This is: (uncovered[sphere_idx_2d]).sum(axis=1)
            # uncovered is bool array of size 6561, sphere_idx_2d is (6561, 17)
            # uncovered[sphere_idx_2d] gives a (6561, 17) bool array
            counts = uncovered[sphere_idx_2d].sum(axis=1)
            
            best_count = int(counts.max())
            
            if best_count <= 0:
                break
            
            # Among all words with the max count, pick one at random for diversity
            best_candidates = np.where(counts == best_count)[0]
            if len(best_candidates) > 1:
                best_idx = int(best_candidates[greedy_rng.randint(0, len(best_candidates) - 1)])
            else:
                best_idx = int(best_candidates[0])
            
            code_indices.append(best_idx)
            # Mark covered words
            uncovered[sphere_idx_2d[best_idx]] = False
            uncovered_count = int(uncovered.sum())
        
        # Now we have a greedy code. Let's try to reduce it via local search.
        # Convert to list of words
        code = [int_to_word(idx) for idx in code_indices]
        
        # Local search: try to remove redundant codewords (vectorized)
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[spheres[idx]] += 1
        
        m = len(code_indices)
        active = np.ones(m, dtype=bool)
        
        # Precompute sphere index matrix for the current code
        sphere_idx_matrix = np.array([spheres[idx] for idx in code_indices], dtype=np.int32)  # shape (m, 17)
        
        # Run many passes of redundant codeword removal with randomization
        rng = random.Random(42)
        for pass_num in range(1000):
            if time.time() > deadline - 2:
                break
            
            if not np.any(active):
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            # For each active codeword, find the minimum cover count among words it covers
            active_spheres = sphere_idx_matrix[active_indices]  # (num_active, 17)
            active_cover_counts = cover_count[active_spheres]  # (num_active, 17)
            min_covers = np.min(active_cover_counts, axis=1)  # (num_active,)
            
            # Find which active codewords are redundant (min_cover >= 2)
            redundant_mask = min_covers >= 2
            redundant_local_indices = np.where(redundant_mask)[0]
            
            if len(redundant_local_indices) == 0:
                break
            
            # Randomize the order of removal to escape local optima
            if len(redundant_local_indices) > 1:
                order = list(range(len(redundant_local_indices)))
                rng.shuffle(order)
                redundant_local_indices = redundant_local_indices[order]
            
            # Remove them one at a time, checking after each removal
            removed_any = False
            for li in redundant_local_indices:
                if time.time() > deadline - 2:
                    break
                gi = active_indices[li]
                active[gi] = False
                cover_count[spheres[code_indices[gi]]] -= 1
                removed_any = True
            
            if not removed_any:
                break
            
            if not np.any(active):
                break
        
        # Additional local search: try swap moves to enable further removals
        # A swap: remove codeword i, add codeword j (not in code) such that coverage is maintained
        # and the new code has a redundant codeword that can be removed
        if time.time() < deadline - 5:
            current_size = int(np.sum(active))
            for swap_round in range(200):
                if time.time() > deadline - 3:
                    break
                
                active_indices = np.where(active)[0]
                if len(active_indices) < 2:
                    break
                
                # Find codewords that are "critical" (have at least one word covered only by them)
                active_spheres = sphere_idx_matrix[active_indices]
                active_cover_counts = cover_count[active_spheres]
                min_covers = np.min(active_cover_counts, axis=1)
                
                # Codewords with min_cover == 1 are critical
                critical_local = np.where(min_covers == 1)[0]
                if len(critical_local) == 0:
                    break
                
                # For each critical codeword, try to find a replacement
                # that covers its unique words and allows removal of another codeword
                found_improvement = False
                for ci in critical_local:
                    if time.time() > deadline - 3:
                        break
                    
                    gi = active_indices[ci]
                    gi_sphere = spheres[code_indices[gi]]
                    
                    # Find words covered only by this codeword
                    unique_words = [w for w in gi_sphere if cover_count[w] == 1]
                    if len(unique_words) == 0:
                        continue
                    
                    # Try to find a non-active codeword that covers all unique_words
                    # and whose addition allows removal of at least one other codeword
                    # This is expensive, so sample a few candidates
                    inactive_indices = np.where(~active)[0]
                    if len(inactive_indices) == 0:
                        break
                    
                    # Sample up to 50 inactive candidates
                    if len(inactive_indices) > 50:
                        sample = rng.sample(list(inactive_indices), 50)
                    else:
                        sample = list(inactive_indices)
                    
                    for si in sample:
                        if time.time() > deadline - 3:
                            break
                        
                        si_sphere = spheres[code_indices[si]]
                        si_sphere_set = set(si_sphere)
                        
                        # Check if si covers all unique words of gi
                        if all(w in si_sphere_set for w in unique_words):
                            # Try the swap: remove gi, add si
                            # Check if this creates any redundant codeword
                            # Temporarily update cover_count
                            old_cover = cover_count.copy()
                            cover_count[gi_sphere] -= 1
                            cover_count[si_sphere] += 1
                            
                            # Check if any active codeword (other than gi, including si if it was active... 
                            # but si is inactive, so just check existing active ones) becomes redundant
                            new_active_indices = np.where(active)[0]
                            new_active_indices = [x for x in new_active_indices if x != gi]
                            
                            if len(new_active_indices) > 0:
                                new_spheres = sphere_idx_matrix[new_active_indices]
                                new_cover_counts = cover_count[new_spheres]
                                new_min_covers = np.min(new_cover_counts, axis=1)
                                new_redundant = np.where(new_min_covers >= 2)[0]
                                
                                if len(new_redundant) > 0:
                                    # Perform the swap and remove the redundant codeword
                                    active[gi] = False
                                    active[si] = True
                                    
                                    # Remove one redundant codeword
                                    ri = new_active_indices[new_redundant[0]]
                                    active[ri] = False
                                    cover_count[spheres[code_indices[ri]]] -= 1
                                    
                                    found_improvement = True
                                    break
                            
                            # Revert cover_count
                            cover_count = old_cover
                        if found_improvement:
                            break
                    
                    if found_improvement:
                        break
                
                if not found_improvement:
                    break
        
        final_indices = [code_indices[i] for i in range(m) if active[i]]
        final_code = [int_to_word(idx) for idx in final_indices]
        
        # Verify coverage
        if len(final_code) < total_space:
            verify_cover = np.zeros(total_space, dtype=bool)
            for idx in final_indices:
                verify_cover[spheres[idx]] = True
            if np.all(verify_cover):
                return [list(w) for w in final_code]
        
        # Fallback: return greedy code
        return [list(w) for w in code]
    
    else:
        # General case: use the tetracode-based construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback: return all words (valid but not minimal)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
