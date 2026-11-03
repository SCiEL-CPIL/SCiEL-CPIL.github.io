import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 142  # 8s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17

        # Precompute all spheres: for each word index, the list of indices it covers
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

        # Precompute powers of 3
        pow3 = [3 ** i for i in range(n + 1)]

        # Build sphere index matrix: shape (6561, 17)
        sphere_idx_2d = np.empty((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
            # Get digits
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            digits = digits[::-1] # Most significant first for standard index calc? 
            # Wait, standard index: d0 * 3^7 + d1 * 3^6 ... 
            # My int_to_word does: w[0] is MSB?
            # Let's check int_to_word:
            # w = []
            # loop n times: w.append(idx % 3); idx //= 3
            # So w[0] is LSB, w[n-1] is MSB.
            # Then reversed(w) is passed to list.
            # So list(w) has w[0] as last element?
            # Let's trace: idx=1. n=2.
            # w.append(1), idx=0. w.append(0), idx=0.
            # w = [1, 0]. reversed(w) = [0, 1].
            # So word is [0, 1].
            # Index of [0, 1] in base 3 is 1. Correct.
            # So word[i] corresponds to digit at position i (0-indexed from left).
            # Index = sum(word[i] * 3^(n-1-i)).
            
            sphere_idx_2d[idx, 0] = idx
            k = 1
            for pos in range(n):
                orig = digits[pos]
                base = pow3[n - 1 - pos]
                for d in range(q):
                    if d != orig:
                        sphere_idx_2d[idx, k] = idx + (d - orig) * base
                        k += 1

        # Phase 1: Greedy Set Cover with Randomization
        # We will run a few greedy passes to find a good starting point.
        best_initial_code = None
        best_initial_size = total_space
        
        greedy_deadline = start_time + 10
        for run in range(5):
            if time.time() > greedy_deadline:
                break
            
            uncovered = np.ones(total_space, dtype=bool)
            uncovered_count = total_space
            code_indices = []
            
            while uncovered_count > 0:
                if time.time() > greedy_deadline:
                    break
                # Count how many uncovered words each candidate covers
                # Vectorized: sum of uncovered over each row of sphere_idx_2d
                counts = uncovered[sphere_idx_2d].sum(axis=1)
                
                # Add slight randomness to break ties or prefer slightly worse if close?
                # Or just pick max.
                # To diversify, we can pick from top K candidates randomly.
                # But simple max is fast. Let's stick to max for speed, but maybe 
                # if we have time, try picking from top 5.
                
                # For now, strict max.
                best_idx = int(np.argmax(counts))
                best_count = int(counts[best_idx])
                
                if best_count <= 0:
                    break
                
                code_indices.append(best_idx)
                uncovered[sphere_idx_2d[best_idx]] = False
                uncovered_count = int(uncovered.sum())
            
            if len(code_indices) < best_initial_size:
                best_initial_size = len(code_indices)
                best_initial_code = code_indices[:]
            
            if best_initial_size <= 486: # We hit the known best, stop early?
                # Maybe try to improve further?
                pass

        if best_initial_code is None:
            # Fallback: simple greedy
            uncovered = np.ones(total_space, dtype=bool)
            uncovered_count = total_space
            best_initial_code = []
            while uncovered_count > 0:
                counts = uncovered[sphere_idx_2d].sum(axis=1)
                best_idx = int(np.argmax(counts))
                if counts[best_idx] == 0: break
                best_initial_code.append(best_idx)
                uncovered[sphere_idx_2d[best_idx]] = False
                uncovered_count = int(uncovered.sum())
        
        code_indices = best_initial_code
        m = len(code_indices)
        
        # Initialize cover_count
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[sphere_idx_2d[idx]] += 1
            
        # Track active status
        active = np.ones(m, dtype=bool)
        
        # Helper to calculate N1 (words covered by exactly 1 codeword)
        def calc_n1():
            return np.sum(cover_count == 1)
        
        current_n1 = calc_n1()
        current_size = m
        current_cost = current_size * 10000 + current_n1
        
        best_size = current_size
        best_n1 = current_n1
        best_cover_count = cover_count.copy()
        best_active = active.copy()
        
        # Phase 2: Local Search
        # Strategy: 
        # 1. Try to remove redundant codewords (Deletion)
        # 2. If no deletion, try Swap (Remove one, Add one) to reduce N1 or maintain size but improve N1
        # We use Simulated Annealing on the cost function.
        
        T = 50.0
        cooling_rate = 0.9995
        T_min = 0.1
        sa_deadline = deadline - 10
        
        # Precompute list of all possible codeword indices (0 to 6560)
        all_indices = np.arange(total_space)
        
        while time.time() < sa_deadline and T > T_min:
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            # 1. Attempt Deletion
            # Find a codeword that is redundant
            # A codeword is redundant if all words in its sphere have cover_count >= 2
            # We can check all active codewords, but that's slow.
            # Instead, we can just pick a random active codeword and try to remove it.
            # If it works, great. If not, try swap.
            
            remove_pos = active_indices[random.randint(0, len(active_indices) - 1)]
            remove_idx = code_indices[remove_pos]
            
            # Remove it
            active[remove_pos] = False
            remove_sphere = sphere_idx_2d[remove_idx]
            cover_count[remove_sphere] -= 1
            
            # Check if valid
            if np.all(cover_count > 0):
                # Valid deletion
                new_size = int(np.sum(active))
                new_n1 = calc_n1()
                new_cost = new_size * 10000 + new_n1
                
                # Always accept deletion (size reduced)
                if new_cost < best_size * 10000 + best_n1:
                    best_size = new_size
                    best_n1 = new_n1
                    best_cover_count = cover_count.copy()
                    best_active = active.copy()
                    
                current_size = new_size
                current_n1 = new_n1
                current_cost = new_cost
                T *= cooling_rate
                continue
            
            # Invalid deletion: some words are uncovered
            uncovered_mask = cover_count == 0
            uncovered_indices = np.where(uncovered_mask)[0]
            
            # 2. Attempt Swap
            # We need to find a codeword c_in that covers all uncovered_indices
            # Candidates for c_in are the union of spheres of uncovered_indices
            # To speed up, we can sample or use a heuristic.
            # Heuristic: For each uncovered word, look at its sphere. 
            # The best c_in is the one that covers the most uncovered words.
            
            # Efficient approach:
            # Create a candidate set.
            # Instead of building the full union, we can just iterate through 
            # the neighbors of the first few uncovered words.
            
            best_gain = 0
            best_cand = -1
            
            # Sample up to 10 uncovered words to generate candidates
            # If the set of uncovered words is small, use all.
            num_uncovered = len(uncovered_indices)
            if num_uncovered == 0:
                # Should not happen here
                active[remove_pos] = True
                cover_count[remove_sphere] += 1
                continue
            
            sample_size = min(10, num_uncovered)
            sampled_uncovered = uncovered_indices[random.sample(range(num_uncovered), sample_size)]
            
            # Get candidates from these samples
            # Note: A candidate c_in must be in the sphere of EVERY uncovered word? 
            # No, c_in must cover the uncovered words. 
            # If we add c_in, it covers its sphere. 
            # So c_in must be such that Sphere(c_in) covers all uncovered_indices.
            # This means c_in must be in the INTERSECTION of the spheres of all uncovered words?
            # No. 
            # Word w is covered by c if w is in Sphere(c). 
            # So for all u in uncovered_indices, u must be in Sphere(c_in).
            # This implies c_in must be in Sphere(u) for all u in uncovered_indices.
            # So c_in must be in the INTERSECTION of Sphere(u) for all u in uncovered_indices.
            
            # Let's find the intersection.
            # Start with the sphere of the first uncovered word.
            cand_set = set(sphere_idx_2d[sampled_uncovered[0]].tolist())
            
            for u in sampled_uncovered[1:]:
                if not cand_set: break
                # Intersect with sphere of u
                # For efficiency, convert to numpy array for intersection? 
                # Sets are fine for small sizes (17 elements).
                cand_set.intersection_update(sphere_idx_2d[u].tolist())
            
            # If we sampled all uncovered words, cand_set is the exact intersection.
            # If we sampled a subset, cand_set is a superset of the true intersection.
            # We need to verify candidates against ALL uncovered words.
            
            if cand_set:
                # Convert to list for iteration
                candidates = list(cand_set)
                
                # Evaluate each candidate
                # We want to maximize the number of uncovered words it covers.
                # Since it's in the intersection of the sampled ones, it covers those.
                # We need to check if it covers the *other* uncovered words.
                
                for c in candidates:
                    # Check if c covers all uncovered_indices
                    # c's sphere is sphere_idx_2d[c]
                    # We need to see if uncovered_indices is a subset of sphere_idx_2d[c]
                    # This is a bit slow if done naively.
                    
                    # Optimization: 
                    # If num_uncovered is small, we can just check.
                    # Let's check coverage count.
                    # How many of the uncovered_indices are in sphere_idx_2d[c]?
                    # We can use a temporary array or set.
                    
                    # For small num_uncovered, set intersection is fast.
                    # But we have many candidates potentially.
                    
                    # Let's just pick the candidate that is in the intersection of ALL uncovered words.
                    # If we didn't sample all, we need to filter.
                    
                    # To keep it fast, let's assume we sampled enough.
                    # If num_uncovered <= 10, we sampled all, so cand_set is exact.
                    # If num_uncovered > 10, we sampled 10. 
                    # In that case, finding a single codeword that covers >10 specific words 
                    # is rare. If it exists, it's a very strong candidate.
                    
                    # Let's just verify the first candidate in the intersection 
                    # against the full uncovered set.
                    # Actually, let's just iterate and find the best.
                    
                    # To avoid O(C * U) complexity, we can precompute a "coverage mask" 
                    # but that's 6561 bits.
                    
                    # Let's just do a simple check for the first few candidates.
                    # If we find one that covers all, accept.
                    
                    # For speed, we'll just pick the first candidate in the intersection 
                    # and verify it. If it fails, we reject the swap.
                    # This is a heuristic.
                    
                    c_check = int(c)
                    # Check if all uncovered_indices are in sphere_idx_2d[c_check]
                    # We can do this by:
                    # np.isin(uncovered_indices, sphere_idx_2d[c_check]).all()
                    
                    if np.isin(uncovered_indices, sphere_idx_2d[c_check]).all():
                        best_cand = c_check
                        break # Found a valid swap candidate
            
            if best_cand != -1:
                # Add best_cand
                # Check if it's already in the code (inactive)
                if best_cand in code_indices:
                    # Find its position
                    # This is slow. 
                    # Better to maintain a set of active indices.
                    # But we have `active` array.
                    # We can just check if it was the one we removed? 
                    # If best_cand == remove_idx, it's a no-op.
                    if best_cand == remove_idx:
                        # Undo removal
                        active[remove_pos] = True
                        cover_count[remove_sphere] += 1
                    else:
                        # Find index of best_cand in code_indices
                        # This is O(M). M is ~500. Acceptable.
                        # We can optimize this later.
                        try:
                            add_pos = code_indices.index(best_cand)
                            active[add_pos] = True
                            cover_count[sphere_idx_2d[best_cand]] += 1
                        except ValueError:
                            # Not in code_indices, add it
                            code_indices.append(best_cand)
                            add_pos = len(code_indices) - 1
                            active = np.append(active, True)
                            cover_count[sphere_idx_2d[best_cand]] += 1
                            
                        new_size = int(np.sum(active))
                        new_n1 = calc_n1()
                        new_cost = new_size * 10000 + new_n1
                        delta_cost = new_cost - current_cost
                        
                        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                            # Accept
                            current_size = new_size
                            current_n1 = new_n1
                            current_cost = new_cost
                            
                            if new_cost < best_size * 10000 + best_n1:
                                best_size = new_size
                                best_n1 = new_n1
                                best_cover_count = cover_count.copy()
                                best_active = active.copy()
                        else:
                            # Reject, undo
                            active[add_pos] = False
                            cover_count[sphere_idx_2d[best_cand]] -= 1
                            # Restore removed one? No, we didn't restore it yet.
                            # Wait, in the "Remove then Repair" logic:
                            # We removed `remove_idx`. 
                            # If we reject the swap, we must restore `remove_idx`.
                            active[remove_pos] = True
                            cover_count[remove_sphere] += 1
                else:
                    # best_cand not in code_indices
                    code_indices.append(best_cand)
                    add_pos = len(code_indices) - 1
                    active = np.append(active, True)
                    cover_count[sphere_idx_2d[best_cand]] += 1
                    
                    new_size = int(np.sum(active))
                    new_n1 = calc_n1()
                    new_cost = new_size * 10000 + new_n1
                    delta_cost = new_cost - current_cost
                    
                    if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                        current_size = new_size
                        current_n1 = new_n1
                        current_cost = new_cost
                        
                        if new_cost < best_size * 10000 + best_n1:
                            best_size = new_size
                            best_n1 = new_n1
                            best_cover_count = cover_count.copy()
                            best_active = active.copy()
                    else:
                        active[add_pos] = False
                        cover_count[sphere_idx_2d[best_cand]] -= 1
                        active[remove_pos] = True
                        cover_count[remove_sphere] += 1
            else:
                # No valid swap found, undo removal
                active[remove_pos] = True
                cover_count[remove_sphere] += 1
            
            T *= cooling_rate

        # Restore best state
        active = best_active.copy()
        cover_count = best_cover_count.copy()
        
        # Phase 3: Final Cleanup
        final_deadline = deadline - 3
        for pass_num in range(1000):
            if time.time() > final_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            active_spheres = sphere_idx_2d[code_indices[active_indices]]
            active_cover_counts = cover_count[active_spheres]
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            if not np.any(redundant_mask):
                break
            redundant_local_indices = np.where(redundant_mask)[0]
            redundant_global_indices = active_indices[redundant_local_indices]
            active[redundant_global_indices] = False
            for ri in redundant_global_indices:
                cover_count[sphere_idx_2d[code_indices[ri]]] -= 1

        final_indices = [code_indices[i] for i in range(len(active)) if active[i]]
        final_code = [int_to_word(idx) for idx in final_indices]

        # Verify coverage
        verify_cover = np.zeros(total_space, dtype=bool)
        for idx in final_indices:
            verify_cover[sphere_idx_2d[idx]] = True
        if np.all(verify_cover):
            return [list(w) for w in final_code]

        # Fallback
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    else:
        # General case
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
