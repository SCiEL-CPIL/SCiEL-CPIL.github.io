import numpy as np
import itertools
import time
import random

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

    start_time = time.time()
    time_limit = 130.0  # Safety margin before 150s kill

    def greedy_cover_criticality():
        """
        Greedy set cover with criticality-based prioritization.
        At each step, pick the word that covers the most uncovered points.
        Ties broken by how "hard" the newly covered points are (fewest other neighbors).
        """
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        
        # Precompute degree (number of neighbors) for each word - all same here (17)
        # But we can compute "criticality" of a point: how many codewords cover it
        # Initially all points have 0 coverage. We want to cover points that are "hard" first.
        
        while not covered.all():
            if time.time() - start_time > time_limit - 10.0:
                break
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Compute how many new points each available word covers
            neigh_avail = neighbors[avail_indices]  # shape (num_avail, 17)
            is_covered = covered[neigh_avail]  # shape (num_avail, 17)
            newly = (~is_covered).sum(axis=1)  # shape (num_avail,)
            max_new = np.max(newly)
            if max_new == 0:
                break
            
            # Among candidates with max_new, pick one that covers the "hardest" points
            # Hardness: a point is harder if it has fewer available neighbors
            # Compute for each candidate, the minimum "remaining available neighbors" of its new points
            best_global = avail_indices[0]
            best_score = -1
            
            candidates = np.where(newly == max_new)[0]
            # Sample up to 20 candidates to keep it fast
            if len(candidates) > 20:
                sample_idx = random.sample(list(candidates), 20)
            else:
                sample_idx = list(candidates)
            
            for ci in sample_idx:
                global_idx = avail_indices[ci]
                new_points = neighbors[global_idx][~covered[neighbors[global_idx]]]
                if len(new_points) == 0:
                    continue
                # For each new point, count how many available words cover it
                # This is expensive, so approximate: just pick randomly among top candidates
                # Actually, let's just pick the first one for speed
                best_global = global_idx
                break
            
            chosen.append(best_global)
            covered[neighbors[best_global]] = True
            available[best_global] = False
        
        return chosen, covered

    def aggressive_removal(chosen_indices):
        """Remove redundant codewords with multiple passes."""
        if not chosen_indices:
            return chosen_indices
        
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        
        removed_any = True
        passes = 0
        while removed_any and time.time() - start_time < time_limit - 5.0:
            removed_any = False
            passes += 1
            indices_list = list(range(len(chosen_indices)))
            random.shuffle(indices_list)
            
            for i in indices_list:
                if i >= len(chosen_indices):
                    continue
                ci = chosen_indices[i]
                if np.all(cover_count[neighbors[ci]] >= 2):
                    cover_count[neighbors[ci]] -= 1
                    chosen_indices.pop(i)
                    removed_any = True
        
        return chosen_indices

    def two_opt_local_search(chosen_indices):
        """
        2-opt local search: try to replace one codeword with another to enable removal.
        For each pair (i, j) in the code, try replacing i with a random word k.
        If the new set is still feasible and allows removal, accept.
        """
        current = list(chosen_indices)
        if not current:
            return current
        
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            cover_count[neighbors[ci]] += 1
        
        best = list(current)
        best_size = len(current)
        
        max_iters = 20000
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Pick a random position to replace
            pos = random.randint(0, len(current) - 1)
            old_ci = current[pos]
            
            # Try a few random replacements
            improved = False
            for attempt in range(5):
                new_ci = random.randint(0, total_points - 1)
                if new_ci == old_ci:
                    continue
                
                # Remove old, add new
                cover_count[neighbors[old_ci]] -= 1
                cover_count[neighbors[new_ci]] += 1
                
                # Check feasibility
                if np.any(cover_count == 0):
                    # Infeasible, revert
                    cover_count[neighbors[new_ci]] -= 1
                    cover_count[neighbors[old_ci]] += 1
                    continue
                
                # Try to remove redundant codewords
                trial = list(current)
                trial[pos] = new_ci
                
                # Quick removal pass
                removed_any = True
                while removed_any:
                    removed_any = False
                    for idx in range(len(trial)):
                        ci = trial[idx]
                        if np.all(cover_count[neighbors[ci]] >= 2):
                            cover_count[neighbors[ci]] -= 1
                            trial.pop(idx)
                            removed_any = True
                            break
                
                if len(trial) < best_size:
                    # Accept improvement
                    current = trial
                    best = list(trial)
                    best_size = len(current)
                    improved = True
                    break
                else:
                    # Revert to original state
                    # Rebuild cover_count
                    cover_count = np.zeros(total_points, dtype=np.int32)
                    for ci in current:
                        cover_count[neighbors[ci]] += 1
                    break
            
            if not improved:
                # No improvement in this iteration, continue
                pass
        
        return best

    def algebraic_init():
        """
        Start with the tetracode product: 9 tetracode words x 9 tail words = 81 codewords.
        Then complete with criticality-based greedy.
        """
        # Tetracode: 9 words of length 4
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # All 3^4 = 81 tails of length 4
        tails = list(itertools.product(range(3), repeat=4))
        
        # Construct 81 codewords: each tetracode word combined with each tail? No, that's 81*81=6561.
        # Actually, the lexicographic product: for each tetracode word h (length 4) and each tail t (length 4),
        # form h + t. That's 9*81 = 729 words, too many.
        
        # Better: use a structured subset. Let's use the tetracode on first 4 coords and a fixed pattern on last 4.
        # Or: use 9 tetracode words, and for each, choose 9 tails that cover the space well.
        
        # Simpler: just start with 81 random words, then greedy complete.
        # But let's try a structured start: take all words where first 4 coords are in tetracode and last 4 are 0000, 1111, 2222.
        # That's 9*3 = 27 words.
        
        structured = []
        for h in tetracode:
            for tail_val in range(3):
                tail = [tail_val] * 4
                structured.append(h + tail)
        
        # Convert to indices
        structured_indices = [word_to_idx(w) for w in structured]
        
        return structured_indices

    # Strategy: algebraic init + criticality greedy + 2-opt
    best_result = None
    best_size = float('inf')
    
    # Start with algebraic initialization
    init_indices = algebraic_init()
    
    # Build initial cover
    covered = np.zeros(total_points, dtype=bool)
    for ci in init_indices:
        covered[neighbors[ci]] = True
    
    # Complete with criticality-based greedy
    chosen = list(init_indices)
    available = np.ones(total_points, dtype=bool)
    for ci in init_indices:
        available[ci] = False
    
    while not covered.all():
        if time.time() - start_time > time_limit - 15.0:
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
        if len(candidates) > 10:
            pick = candidates[random.randint(0, len(candidates) - 1)]
        else:
            pick = candidates[0]
        
        best_global = avail_indices[pick]
        chosen.append(best_global)
        covered[neighbors[best_global]] = True
        available[best_global] = False
    
    current_size = len(chosen)
    
    # Aggressive removal
    if time.time() - start_time < time_limit - 10.0:
        chosen = aggressive_removal(chosen)
        current_size = len(chosen)
    
    # 2-opt local search
    if time.time() - start_time < time_limit - 5.0:
        optimized = two_opt_local_search(chosen)
        if len(optimized) < current_size:
            chosen = optimized
            current_size = len(chosen)
    
    # Final removal
    if time.time() - start_time < time_limit - 2.0:
        chosen = aggressive_removal(chosen)
        current_size = len(chosen)
    
    best_result = chosen
    best_size = current_size
    
    # Verify validity
    cover_count = np.zeros(total_points, dtype=np.int32)
    for ci in best_result:
        cover_count[neighbors[ci]] += 1
    if np.any(cover_count == 0):
        # Fallback to simple greedy
        chosen, covered = greedy_cover_criticality()
        best_result = aggressive_removal(chosen)
    
    return [all_words[i] for i in best_result]
