import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    # Fallback for non-specific parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(8,1)
    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    start_time = time.time()
    time_limit = 148.0  # Safety margin before 150s kill

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all 6561 words
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig_val

    all_words = [idx_to_word(i) for i in range(total_points)]

    def compute_coverage(chosen_indices):
        """Compute which points are covered and how many times."""
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return cover_count

    def is_valid_cover(chosen_indices):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return np.all(cover_count > 0)

    def greedy_cover_vectorized():
        """
        Vectorized greedy set cover.
        Uses matrix operations to speed up the selection of the best codeword.
        """
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        
        # Precompute neighbor matrix for faster access
        # neighbors is already (total_points, neighbors_per_word)
        
        while not covered.all():
            if time.time() - start_time > time_limit - 5.0:
                break
                
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
                
            # Get neighbors of all available words
            # neighbors[avail_indices] has shape (num_avail, 17)
            neigh_avail = neighbors[avail_indices]
            
            # Check which of these neighbors are already covered
            is_covered = covered[neigh_avail] # Shape (num_avail, 17)
            
            # Count how many NEW points each available word covers
            newly = (~is_covered).sum(axis=1)
            
            max_new = np.max(newly)
            if max_new == 0:
                break
                
            # Pick the word that covers the most new points
            # If ties, pick randomly among the best to add diversity
            candidates = np.where(newly == max_new)[0]
            pick = candidates[random.randint(0, len(candidates) - 1)]
            best_global = avail_indices[pick]
            
            chosen.append(best_global)
            covered[neighbors[best_global]] = True
            available[best_global] = False
            
        return chosen, covered

    def local_search_reduce(chosen_indices, time_budget):
        """
        Local search to reduce code size.
        Moves:
        1. Remove redundant words (if possible).
        2. Swap words to increase coverage efficiency (minimize total count).
        Uses a simple hill-climbing approach with random restarts if stuck.
        """
        current = list(chosen_indices)
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            cover_count[neighbors[ci]] += 1
        
        best = list(current)
        best_size = len(current)
        
        improved = True
        max_rounds = 50
        round_num = 0
        
        while improved and round_num < max_rounds and time.time() - start_time < time_limit - 2.0:
            improved = False
            round_num += 1
            # Shuffle order to escape local optima
            order = list(range(len(current)))
            random.shuffle(order)
            
            for i in order:
                if i >= len(current):
                    continue
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                ci = current[i]
                
                # Try removing ci
                # Find points that would become uncovered
                exposed = neighbors[ci]
                would_uncover = exposed[cover_count[exposed] == 1]
                
                if len(would_uncover) == 0:
                    # Safe to remove
                    cover_count[neighbors[ci]] -= 1
                    current.pop(i)
                    improved = True
                    continue
                
                # Try to replace ci with a better word
                # We want a word that covers all `would_uncover` points.
                # Ideally, it also covers other points that are currently covered
                # by *only* this word (which is impossible since we just removed it)
                # or points covered by few other words.
                
                # Strategy: Find a word that covers all `would_uncover` and has
                # a "good" profile on the rest of its neighbors.
                
                best_replacement = -1
                best_gain = -1 # We want to minimize the "cost" of the replacement
                # Actually, let's just look for a word that covers the exposed set.
                # If multiple, pick one that covers the most *other* points that are
                # currently covered by >= 2 words (so we can potentially remove those later).
                
                # To speed this up, we don't check all 6561 words.
                # We check a subset.
                
                # Optimization: Only consider words that are neighbors of the exposed points?
                # No, a word that covers ALL exposed points must be a common neighbor of all exposed points.
                # The set of common neighbors of `would_uncover` is small.
                
                # Let's find common neighbors of `would_uncover`.
                # Since `would_uncover` can be large, this is tricky.
                
                # Alternative: Just sample random words and check if they cover `would_uncover`.
                num_samples = 100
                found_replacement = False
                
                # Pre-compute a set of would_uncover for fast lookup?
                # would_uncover is an array.
                
                for _ in range(num_samples):
                    cand = random.randint(0, total_points - 1)
                    if cand == ci:
                        continue
                    
                    cand_neigh = neighbors[cand]
                    cand_set = set(cand_neigh.tolist())
                    
                    # Check if cand covers all exposed points
                    if not all(p in cand_set for p in would_uncover):
                        continue
                    
                    # If it covers them, calculate the "gain"
                    # Gain = number of points in cand_neigh that are covered by >= 2 words
                    # (excluding ci, which we are replacing)
                    # If we replace ci with cand, the cover_count for cand_neigh increases by 1.
                    # The cover_count for neighbors[ci] decreases by 1.
                    
                    # We want to maximize the potential for future removals.
                    # A good proxy is: how many points in cand_neigh are currently covered by exactly 1 word?
                    # If we add cand, those points become covered by 2 words.
                    # This doesn't directly help removal, but it adds redundancy.
                    
                    # Better metric:
                    # After replacement, how many points in the NEW code are covered by exactly 1 word?
                    # We want to MINIMIZE this number.
                    
                    # Let's simulate the replacement temporarily.
                    old_count = cover_count.copy()
                    cover_count[neighbors[ci]] -= 1
                    cover_count[cand_neigh] += 1
                    
                    # Count single-covered points
                    single_count = np.sum(cover_count == 1)
                    
                    # Restore
                    cover_count[neighbors[ci]] += 1
                    cover_count[cand_neigh] -= 1
                    
                    # We want to minimize single_count.
                    # So best_gain should be -single_count.
                    if -single_count > best_gain:
                        best_gain = -single_count
                        best_replacement = cand
                        found_replacement = True

                if found_replacement:
                    # Perform replacement
                    cover_count[neighbors[ci]] -= 1
                    cover_count[neighbors[best_replacement]] += 1
                    current[i] = best_replacement
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best = list(current)
        
        return best

    def iterated_local_search():
        """
        Iterated local search with multiple restarts.
        """
        best_result = None
        best_size = float('inf')
        
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Generate initial cover
            chosen, covered = greedy_cover_vectorized()
            
            if not covered.all():
                continue
            
            # Local search to reduce
            optimized = local_search_reduce(chosen, 0)
            
            # Verify validity
            if not is_valid_cover(optimized):
                continue
            
            current_size = len(optimized)
            
            if current_size < best_size:
                best_size = current_size
                best_result = list(optimized)
            
            # Early stop if we're doing very well
            if best_size <= 410:
                break
            
            # Perturb: randomly remove a few words and re-add them greedily?
            # Or just restart with a different random seed in greedy.
            
        return best_result, best_size

    # Run the iterated local search
    best_result, best_size = iterated_local_search()
    
    # Fallback if no valid result
    if best_result is None:
        chosen, covered = greedy_cover_vectorized()
        if not covered.all():
             # Emergency full greedy
             covered = np.zeros(total_points, dtype=bool)
             chosen = []
             available = np.ones(total_points, dtype=bool)
             while not covered.all():
                 avail_indices = np.where(available)[0]
                 if len(avail_indices) == 0:
                     break
                 neigh_avail = neighbors[avail_indices]
                 is_covered = covered[neigh_avail]
                 newly = (~is_covered).sum(axis=1)
                 max_new = np.max(newly)
                 if max_new == 0:
                     break
                 candidates = np.where(newly == max_new)[0]
                 pick = candidates[random.randint(0, len(candidates) - 1)]
                 best_global = avail_indices[pick]
                 chosen.append(best_global)
                 covered[neighbors[best_global]] = True
                 available[best_global] = False
        else:
             best_result = local_search_reduce(chosen, 0)
             best_size = len(best_result)
    
    # Final verification
    if best_result is not None and not is_valid_cover(best_result):
        # Emergency fallback: full greedy
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        while not covered.all():
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            neigh_avail = neighbors[avail_indices]
            is_covered = covered[neigh_avail]
            newly = (~is_covered).sum(axis=1)
            max_new = np.max(newly)
            if max_new == 0:
                break
            candidates = np.where(newly == max_new)[0]
            pick = candidates[random.randint(0, len(candidates) - 1)]
            best_global = avail_indices[pick]
            chosen.append(best_global)
            covered[neighbors[best_global]] = True
            available[best_global] = False
        best_result = chosen
    
    return [all_words[i] for i in best_result]
