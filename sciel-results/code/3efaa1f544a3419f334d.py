"""
Optimized 2-colouring of [1,256] to minimize monochromatic 4-term APs.
Combines fast delta-based local search with multi-bit (3-bit) Simulated Annealing.
"""
import random
import time
import numpy as np

N = 256
TIME_LIMIT = 115.0  # Safety margin before 120s kill
SEED = 42


def construct():
    random.seed(SEED)
    np.random.seed(SEED)
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # 1. Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps_arr = np.array(aps, dtype=np.int32)
    num_aps = len(aps)

    # 2. Precompute incidence structure for fast delta calculation
    # ap_indices[i] = list of indices of APs that contain position i
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos].append(idx)
    
    # Convert to numpy arrays for faster iteration if needed, but lists are fine for small K
    # We also need the other 3 positions for each AP to quickly check mono status
    # ap_other[pos_idx, ap_local_idx] = (j, k, l) where i is the flipped one?
    # Actually, for multi-bit, we just need the AP indices.
    
    # Helper to check if an AP is monochromatic given a colors array
    # We will use this in the multi-bit delta calculation on the affected subset.
    
    def is_mono_vec(aps_subset, colors):
        """Check if a set of APs (indices) are monochromatic. Returns count."""
        if len(aps_subset) == 0:
            return 0
        # aps_subset is a list of indices into aps_arr
        sub_arr = aps_arr[aps_subset]
        c0 = colors[sub_arr[:, 0]]
        c1 = colors[sub_arr[:, 1]]
        c2 = colors[sub_arr[:, 2]]
        c3 = colors[sub_arr[:, 3]]
        return np.sum((c0 == c1) & (c1 == c2) & (c2 == c3))

    def count_mono_full(colors):
        """Full vectorized count."""
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))

    def calc_delta_multi_bit(colors, flip_indices):
        """
        Calculate the change in mono-AP count if we flip the bits at flip_indices.
        Returns (delta, affected_ap_indices).
        """
        # Find all APs affected by these flips
        affected_set = set()
        for i in flip_indices:
            affected_set.update(ap_indices[i])
        
        if not affected_set:
            return 0, []
        
        affected_list = list(affected_set)
        
        # Count mono in affected APs before flip
        old_cost = is_mono_vec(affected_list, colors)
        
        # Flip
        for i in flip_indices:
            colors[i] = 1 - colors[i]
            
        # Count mono in affected APs after flip
        new_cost = is_mono_vec(affected_list, colors)
        
        # Unflip (we want to return the delta, and the caller decides if to keep)
        # But wait, in SA we usually apply the flip if accepted. 
        # Let's design this to apply the flip permanently if accepted.
        # So, let's return the delta and the list of affected APs.
        # The caller will unflip if rejected.
        
        return new_cost - old_cost, affected_list

    def polish_1bit(colors, current_score, time_limit=0.5):
        """
        Greedy local search using 1-bit flips with fast delta.
        """
        polish_start = time.time()
        improved = True
        # We will iterate through positions. To speed up, we can just do passes.
        # Since N=256, a full pass is 256 delta calcs.
        # Each delta calc is O(deg(i)) ~ 100 ops. Very fast.
        
        while improved and (time.time() - polish_start) < time_limit:
            improved = False
            # Randomize order to avoid bias
            order = np.random.permutation(N)
            for i in order:
                if time.time() > deadline - 5:
                    return colors, current_score
                
                old_c = colors[i]
                # Calculate delta for flipping i
                # Affected APs are ap_indices[i]
                affected = ap_indices[i]
                if not affected:
                    continue
                
                # Before flip
                # We need to check which of these APs are mono and will change.
                # An AP is mono if all 4 are same.
                # If we flip i, an AP containing i changes mono status only if the other 3 were already mono.
                # If other 3 are same color X:
                #   If old_c == X, AP was mono, becomes non-mono (delta -1)
                #   If old_c != X, AP was non-mono, becomes mono (delta +1)
                
                # Vectorized check for neighbors
                # Get the other 3 positions for each AP in affected
                # This is tricky with standard numpy without precomputed "others" array.
                # Let's stick to the set-based approach for 1-bit too, it's fast enough.
                # Or precompute "others" for each i.
                
                # Precomputing others is better.
                # Let's do it outside the loop if possible? No, depends on colors.
                # But the *indices* of others are static.
                
                # Let's use the full subset count method for 1-bit as well, it's simpler and robust.
                # is_mono_vec is very fast for small lists.
                delta, _ = calc_delta_multi_bit(colors, [i])
                
                if delta < 0:
                    colors[i] = 1 - colors[i]
                    current_score += delta
                    improved = True
                # else: unflip is not needed because calc_delta_multi_bit unflips? 
                # No, my calc_delta_multi_bit flips, calculates, but doesn't unflip.
                # I need to be careful. 
                # Let's rewrite calc_delta_multi_bit to NOT flip, just calculate.
                
                # REVISION: 
                # calc_delta should just return the delta.
                # Then we apply the flip if we want.
                
        return colors, current_score

    # Let's refactor the delta calculation to be cleaner.
    def get_delta(colors, flip_indices):
        """Returns the change in score if flip_indices are flipped. Does not modify colors."""
        affected_set = set()
        for i in flip_indices:
            affected_set.update(ap_indices[i])
        
        if not affected_set:
            return 0
        
        affected_list = list(affected_set)
        
        # Save original colors for affected indices to restore? No, we just need to simulate.
        # We can just flip, count, unflip.
        # But flipping/unflipping in Python loop is slow.
        
        # Alternative:
        # old_count = is_mono_vec(affected_list, colors)
        # Create a copy of colors? No, too expensive.
        # Just flip, count, unflip.
        
        # Flip
        for i in flip_indices:
            colors[i] = 1 - colors[i]
        
        new_count = is_mono_vec(affected_list, colors)
        
        # Unflip
        for i in flip_indices:
            colors[i] = 1 - colors[i]
            
        old_count = is_mono_vec(affected_list, colors)
        
        return new_count - old_count

    def polish_1bit_optimized(colors, current_score, time_limit=0.5):
        """
        Greedy local search using 1-bit flips.
        """
        polish_start = time.time()
        improved = True
        while improved and (time.time() - polish_start) < time_limit:
            improved = False
            order = np.random.permutation(N)
            for i in order:
                if time.time() > deadline - 5:
                    return colors, current_score
                
                delta = get_delta(colors, [i])
                if delta < 0:
                    colors[i] = 1 - colors[i]
                    current_score += delta
                    improved = True
        return colors, current_score

    def sa_3bit(colors, current_score, start_temp=2.0, end_temp=0.1, max_iter=100, time_limit=1.0):
        """
        Simulated Annealing with 3-bit flips.
        """
        sa_start = time.time()
        T = start_temp
        best_score = current_score
        best_colors = colors.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5 or (time.time() - sa_start) > time_limit:
                break
            
            # Pick 3 distinct random indices
            flip_indices = np.random.choice(N, 3, replace=False)
            
            delta = get_delta(colors, flip_indices)
            
            # Acceptance probability
            if delta < 0:
                accept = True
            else:
                # Boltzmann factor
                # Avoid division by zero
                if T < 1e-9:
                    accept = False
                else:
                    accept = np.random.random() < np.exp(-delta / T)
            
            if accept:
                for i in flip_indices:
                    colors[i] = 1 - colors[i]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
            
            # Cool down
            T = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        return best_colors, best_score

    # 3. Initialization
    pop_size = 20
    population = []
    scores = []
    
    # Diverse initial patterns
    patterns = []
    # Random
    for _ in range(5):
        patterns.append(np.random.randint(0, 2, N).astype(np.int8))
    # Structured
    patterns.append(np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8))
    patterns.append(np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8))
    patterns.append(np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8))
    patterns.append(np.array([(i % 4) < 2 for i in range(N)], dtype=np.int8))
    patterns.append(np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8))
    
    for p in patterns[:pop_size]:
        c = p.copy()
        s = count_mono_full(c)
        population.append(c)
        scores.append(s)
    
    # Polish all initial individuals
    for i in range(pop_size):
        population[i], scores[i] = polish_1bit_optimized(population[i], scores[i], time_limit=0.5)
        
    best_score = min(scores)
    best_colors = population[scores.index(best_score)].copy()

    # 4. Main GA Loop
    gen = 0
    while time.time() < deadline - 10:
        gen += 1
        
        # Selection: Tournament
        def tournament_select(pop, scores, k=3):
            idxs = np.random.choice(len(pop), k, replace=False)
            best_idx = idxs[np.argmin([scores[i] for i in idxs])]
            return pop[best_idx].copy()
        
        new_pop = []
        new_scores = []
        
        for _ in range(pop_size):
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            
            # Crossover
            mask = np.random.random(N) < 0.5
            child = np.where(mask, p1, p2).astype(np.int8)
            
            # Mutation: 3-bit flip
            # Sometimes do 1-bit, sometimes 3-bit? 
            # Knowledge says 3-bit is better for escaping. Let's stick to 3-bit for mutation.
            if np.random.random() < 0.8:
                flip_idx = np.random.choice(N, 3, replace=False)
            else:
                flip_idx = np.random.choice(N, 1, replace=False)
            
            for i in flip_idx:
                child[i] = 1 - child[i]
                
            s = count_mono_full(child)
            
            # Refinement: Polish then SA
            child, s = polish_1bit_optimized(child, s, time_limit=0.2)
            
            # If time allows, try SA escape
            if time.time() < deadline - 15:
                child, s = sa_3bit(child, s, start_temp=1.5, end_temp=0.05, max_iter=50, time_limit=0.5)
            
            new_pop.append(child)
            new_scores.append(s)
            
            if s < best_score:
                best_score = s
                best_colors = child.copy()
        
        # Update Population
        # Combine old and new, keep best pop_size
        all_cands = [(c, s) for c, s in zip(population, scores)] + \
                    [(c, s) for c, s in zip(new_pop, new_scores)]
        
        all_cands.sort(key=lambda x: x[1])
        
        population = [c for c, s in all_cands[:pop_size]]
        scores = [s for c, s in all_cands[:pop_size]]
        
        # Periodically, run a deeper SA on the global best to try to break out of the plateau
        if gen % 10 == 0 and time.time() < deadline - 20:
            deep_c = best_colors.copy()
            deep_s = best_score
            deep_c, deep_s = sa_3bit(deep_c, deep_s, start_temp=3.0, end_temp=0.1, max_iter=200, time_limit=2.0)
            if deep_s < best_score:
                best_score = deep_s
                best_colors = deep_c
                # Insert into population
                population[0] = best_colors
                scores[0] = best_score
                # Re-sort
                sorted_idx = np.argsort(scores)
                population = [population[i] for i in sorted_idx]
                scores = [scores[i] for i in sorted_idx]

    # Final Polish
    if time.time() < deadline - 2:
        best_colors, best_score = polish_1bit_optimized(best_colors, best_score, time_limit=2.0)
        # One last SA push
        if time.time() < deadline - 1:
            best_colors, best_score = sa_3bit(best_colors, best_score, start_temp=1.0, end_temp=0.01, max_iter=100, time_limit=1.0)

    return [int(x) for x in best_colors]

# EVOLVE-BLOCK-END
