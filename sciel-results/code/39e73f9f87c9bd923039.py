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
    time_limit = 145.0  # Safety margin before 150s kill

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
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return cover_count

    def is_valid_cover(chosen_indices):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return np.all(cover_count > 0)

    def structured_initial_cover():
        """
        Construct an initial cover using a structured approach:
        1. Start with the tetracode product (16 * 27 = 432 words) which is a valid code but might not be minimal.
        Actually, the product code of tetracode(4,1) and full space(4) is 9 * 81 = 729? No.
        Tetracode has 9 words. Product with full space of length 4 (81 words) gives 729.
        Wait, K_3(4,1) = 9 (tetracode). K_3(8,1) <= K_3(4,1) * K_3(4,1) = 81? No, product code radius is sum of radii.
        If we take product of two codes with radius 1, the resulting code has radius 2.
        We need radius 1.
        
        Let's use a simpler structured start:
        Greedy is fast enough to get a good start. Let's just use a fast greedy.
        """
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        
        # Vectorized greedy
        while not covered.all():
            if time.time() - start_time > time_limit - 20.0:
                break
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
            
        return chosen

    def local_search_swap(chosen_indices):
        """
        1-exchange local search:
        Try to remove a word. If it fails (some points become uncovered),
        try to replace it with a word that covers those points and is "better"
        (e.g., covers more points that are currently single-covered, or simply
        allows further removals).
        
        To be efficient, we focus on:
        1. Removing redundant words (cover_count > 1 for all neighbors).
        2. Swapping a word with another that covers the same "critical" points 
           but potentially allows more redundancy elsewhere.
        """
        current = list(chosen_indices)
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            cover_count[neighbors[ci]] += 1
        
        best = list(current)
        best_size = len(current)
        
        improved = True
        max_rounds = 100
        round_num = 0
        
        while improved and round_num < max_rounds and time.time() - start_time < time_limit - 2.0:
            improved = False
            round_num += 1
            order = list(range(len(current)))
            random.shuffle(order)
            
            for i in order:
                if i >= len(current):
                    continue
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                ci = current[i]
                
                # Try removing ci
                exposed = neighbors[ci]
                would_uncover = exposed[cover_count[exposed] == 1]
                
                if len(would_uncover) == 0:
                    # Safe to remove
                    cover_count[neighbors[ci]] -= 1
                    current.pop(i)
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best = list(current)
                    continue
                
                # Try to replace ci with a better word
                # We need a word that covers all `would_uncover`.
                # Let's sample candidates.
                # A good candidate is a neighbor of the exposed points?
                # Actually, any word that covers all `would_uncover` works.
                
                # Heuristic: Try random words, or words close to `ci`?
                # Let's try random words first, limited count for speed.
                best_replacement = -1
                best_gain = -1
                
                # Optimization: Instead of checking all 6561, check a subset.
                # We can prioritize words that are neighbors of the exposed points?
                # No, the replacement word must cover ALL exposed points.
                # The set of such words is the intersection of the neighbor sets of all exposed points.
                # This intersection is likely small or empty.
                
                # Let's just sample 50 random words.
                for _ in range(50):
                    cand = random.randint(0, total_points - 1)
                    if cand == ci:
                        continue
                    
                    cand_neigh = neighbors[cand]
                    # Check if cand covers all would_uncover
                    # would_uncover is an array of indices
                    # We can use a set for fast lookup
                    cand_set = set(cand_neigh.tolist())
                    
                    if not all(p in cand_set for p in would_uncover):
                        continue
                    
                    # Calculate gain:
                    # If we replace ci with cand:
                    # cover_count[neighbors[ci]] -= 1
                    # cover_count[cand_neigh] += 1
                    # We want to minimize the number of points with cover_count == 1.
                    
                    old_count = cover_count.copy()
                    cover_count[neighbors[ci]] -= 1
                    cover_count[cand_neigh] += 1
                    
                    single_count = np.sum(cover_count == 1)
                    
                    # Restore
                    cover_count[neighbors[ci]] += 1
                    cover_count[cand_neigh] -= 1
                    
                    # We want to minimize single_count, so maximize -single_count
                    if -single_count > best_gain:
                        best_gain = -single_count
                        best_replacement = cand

                if best_replacement != -1:
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
        best_result = None
        best_size = float('inf')
        
        num_restarts = 10 # Increase restarts since each is faster
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Generate initial cover
            chosen = structured_initial_cover()
            
            if not is_valid_cover(chosen):
                continue
            
            # Local search to reduce
            optimized = local_search_swap(chosen)
            
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
        
        return best_result, best_size

    # Run the iterated local search
    best_result, best_size = iterated_local_search()
    
    # Fallback if no valid result
    if best_result is None:
        chosen = structured_initial_cover()
        if not is_valid_cover(chosen):
             # Emergency full greedy (should not happen if structured_initial_cover works)
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
             best_result = local_search_swap(chosen)
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
