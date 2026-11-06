import itertools
import random
import time

def construct():
    n = 16
    deadline = time.time() + 115.0
    
    # Precompute all triples and their properties
    all_triples = list(itertools.combinations(range(n), 3))
    triple_masks = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples}
    
    # Precompute supersets for each triple mask
    supersets_cache = {}
    for t in all_triples:
        mask = triple_masks[t]
        free = (~mask) & 0xFFFF
        subs = []
        sub = free
        while True:
            subs.append(mask | sub)
            if sub == 0:
                break
            sub = (sub - 1) & free
        supersets_cache[t] = subs

    # Precompute popcount for all 16-bit masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute submasks for each mask (for fast alpha calculation)
    # submasks[mask] = list of all submasks of mask (including mask itself and 0)
    submasks_cache = [None] * (1 << n)
    for mask in range(1 << n):
        subs = []
        sub = mask
        while True:
            subs.append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & mask
        submasks_cache[mask] = subs

    def calculate_alpha(cover_count):
        """
        Find max popcount of mask where cover_count[mask] == 0.
        Uses subset iteration for efficiency.
        """
        max_alpha = 0
        # Try masks in descending order of popcount
        # Group masks by popcount for efficiency
        for pc in range(n, max_alpha, -1):
            # Find any mask with this popcount that has cover_count == 0
            found = False
            # Enumerate all masks with exactly pc bits
            # Use combinatorial generation
            for combo in itertools.combinations(range(n), pc):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                if cover_count[mask] == 0:
                    max_alpha = pc
                    found = True
                    break
            if found:
                break
        return max_alpha

    def calculate_alpha_fast(cover_count, current_best):
        """
        Faster alpha calculation: only check if we can beat current_best.
        Returns the new alpha value.
        """
        # Check if there's an independent set of size current_best + 1
        for pc in range(n, current_best, -1):
            for combo in itertools.combinations(range(n), pc):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                if cover_count[mask] == 0:
                    return pc
        return current_best

    def get_max_independent_set(cover_count):
        """Return the actual mask of a maximum independent set."""
        best_mask = 0
        best_size = 0
        for pc in range(n, 0, -1):
            for combo in itertools.combinations(range(n), pc):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                if cover_count[mask] == 0:
                    return mask
        return 0

    def generate_random_packing():
        used_pairs = set()
        triples = []
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for t in candidates:
            a, b, c = t
            ps = frozenset([(a, b), (a, c), (b, c)])
            if not (ps & used_pairs):
                triples.append(t)
                used_pairs.update(ps)
        return triples, used_pairs

    def is_valid_addition(t, used_pairs):
        a, b, c = t
        ps = frozenset([(a, b), (a, c), (b, c)])
        if ps & used_pairs:
            return False
        return True

    best_alpha = 17
    best_triples = []
    rng = random.Random(42)
    
    # Main search loop
    while time.time() < deadline:
        triples, used_pairs = generate_random_packing()
        
        # Initialize cover_count
        cover_count = [0] * (1 << n)
        for t in triples:
            for sup in supersets_cache[t]:
                cover_count[sup] += 1
        
        current_alpha = calculate_alpha(cover_count)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            
        if best_alpha <= 5:
            break
            
        # Local Search with multiple strategies
        search_deadline = min(time.time() + 2.0, deadline - 2.0)
        iterations = 0
        
        while time.time() < search_deadline:
            iterations += 1
            
            # Strategy 1: Single triple swap
            if len(triples) >= 2 and rng.random() < 0.6:
                idx = rng.randrange(len(triples))
                t_old = triples[idx]
                
                # Remove t_old
                used_pairs.difference_update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                for sup in supersets_cache[t_old]:
                    cover_count[sup] -= 1
                    
                # Find a replacement - prioritize triples that hit the current max independent set
                max_is_mask = get_max_independent_set(cover_count)
                
                # Sort candidates by how many points they share with the max IS
                candidates = [t for t in all_triples if t != t_old]
                # Filter to valid additions
                valid_candidates = []
                for t in candidates:
                    if is_valid_addition(t, used_pairs):
                        valid_candidates.append(t)
                
                if valid_candidates:
                    # Score candidates by how many points they hit in the max IS
                    def candidate_score(t):
                        mask = triple_masks[t]
                        hits = popcount[mask & max_is_mask]
                        return -hits  # We want to maximize hits, so negate for sorting
                    
                    # Shuffle first for diversity, then sort
                    rng.shuffle(valid_candidates)
                    valid_candidates.sort(key=candidate_score)
                    
                    # Try top candidates
                    tried = 0
                    found = False
                    for t_new in valid_candidates[:20]:
                        tried += 1
                        triples[idx] = t_new
                        used_pairs.update(frozenset([(t_new[0], t_new[1]), (t_new[0], t_new[2]), (t_new[1], t_new[2])]))
                        
                        for sup in supersets_cache[t_new]:
                            cover_count[sup] += 1
                            
                        new_alpha = calculate_alpha_fast(cover_count, current_alpha)
                        
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_triples = triples[:]
                            
                        if new_alpha <= current_alpha:
                            current_alpha = new_alpha
                            found = True
                            break
                        else:
                            # Revert
                            used_pairs.difference_update(frozenset([(t_new[0], t_new[1]), (t_new[0], t_new[2]), (t_new[1], t_new[2])]))
                            for sup in supersets_cache[t_new]:
                                cover_count[sup] -= 1
                            triples[idx] = t_old
                            used_pairs.update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                            for sup in supersets_cache[t_old]:
                                cover_count[sup] += 1
                    
                    if not found:
                        # Revert
                        used_pairs.update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                        for sup in supersets_cache[t_old]:
                            cover_count[sup] += 1
                        triples[idx] = t_old
            
            # Strategy 2: Double triple swap
            elif len(triples) >= 3 and rng.random() < 0.5:
                idx1, idx2 = rng.sample(range(len(triples)), 2)
                t_old1 = triples[idx1]
                t_old2 = triples[idx2]
                
                # Remove both
                for t_old in [t_old1, t_old2]:
                    used_pairs.difference_update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                    for sup in supersets_cache[t_old]:
                        cover_count[sup] -= 1
                
                # Find replacements
                max_is_mask = get_max_independent_set(cover_count)
                
                valid_candidates = []
                for t in all_triples:
                    if t != t_old1 and t != t_old2 and is_valid_addition(t, used_pairs):
                        valid_candidates.append(t)
                
                if len(valid_candidates) >= 2:
                    # Try pairs of candidates
                    rng.shuffle(valid_candidates)
                    found = False
                    for i in range(min(10, len(valid_candidates))):
                        t_new1 = valid_candidates[i]
                        if not is_valid_addition(t_new1, used_pairs):
                            continue
                        # Temporarily add t_new1
                        used_pairs.update(frozenset([(t_new1[0], t_new1[1]), (t_new1[0], t_new1[2]), (t_new1[1], t_new1[2])]))
                        
                        for j in range(i+1, min(i+10, len(valid_candidates))):
                            t_new2 = valid_candidates[j]
                            if not is_valid_addition(t_new2, used_pairs):
                                continue
                            
                            # Add t_new2
                            used_pairs.update(frozenset([(t_new2[0], t_new2[1]), (t_new2[0], t_new2[2]), (t_new2[1], t_new2[2])]))
                            for sup in supersets_cache[t_new1]:
                                cover_count[sup] += 1
                            for sup in supersets_cache[t_new2]:
                                cover_count[sup] += 1
                            
                            new_alpha = calculate_alpha_fast(cover_count, current_alpha)
                            
                            if new_alpha < best_alpha:
                                best_alpha = new_alpha
                                best_triples = triples[:]
                            
                            if new_alpha <= current_alpha:
                                triples[idx1] = t_new1
                                triples[idx2] = t_new2
                                current_alpha = new_alpha
                                found = True
                                break
                            else:
                                # Revert t_new2
                                used_pairs.difference_update(frozenset([(t_new2[0], t_new2[1]), (t_new2[0], t_new2[2]), (t_new2[1], t_new2[2])]))
                                for sup in supersets_cache[t_new2]:
                                    cover_count[sup] -= 1
                        
                        if found:
                            break
                        
                        # Revert t_new1
                        used_pairs.difference_update(frozenset([(t_new1[0], t_new1[1]), (t_new1[0], t_new1[2]), (t_new1[1], t_new1[2])]))
                    
                    if not found:
                        # Restore originals
                        for t_old in [t_old1, t_old2]:
                            used_pairs.update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                            for sup in supersets_cache[t_old]:
                                cover_count[sup] += 1
                        triples[idx1] = t_old1
                        triples[idx2] = t_old2
            
            # Strategy 3: Directly target the max independent set
            else:
                max_is_mask = get_max_independent_set(cover_count)
                max_is_points = [p for p in range(n) if (max_is_mask >> p) & 1]
                
                if len(max_is_points) >= 3:
                    # Try to find a triple that hits 2 or more points in the max IS
                    # and is valid to add (replacing some existing triple)
                    for idx in range(len(triples)):
                        t_old = triples[idx]
                        # Remove t_old
                        used_pairs.difference_update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                        for sup in supersets_cache[t_old]:
                            cover_count[sup] -= 1
                        
                        found = False
                        # Try triples that hit at least 2 points in max IS
                        for t_new in all_triples:
                            if t_new == t_old:
                                continue
                            mask = triple_masks[t_new]
                            hits = popcount[mask & max_is_mask]
                            if hits >= 2 and is_valid_addition(t_new, used_pairs):
                                triples[idx] = t_new
                                used_pairs.update(frozenset([(t_new[0], t_new[1]), (t_new[0], t_new[2]), (t_new[1], t_new[2])]))
                                for sup in supersets_cache[t_new]:
                                    cover_count[sup] += 1
                                
                                new_alpha = calculate_alpha_fast(cover_count, current_alpha)
                                
                                if new_alpha < best_alpha:
                                    best_alpha = new_alpha
                                    best_triples = triples[:]
                                
                                if new_alpha <= current_alpha:
                                    current_alpha = new_alpha
                                    found = True
                                    break
                                else:
                                    # Revert
                                    used_pairs.difference_update(frozenset([(t_new[0], t_new[1]), (t_new[0], t_new[2]), (t_new[1], t_new[2])]))
                                    for sup in supersets_cache[t_new]:
                                        cover_count[sup] -= 1
                                    triples[idx] = t_old
                                    used_pairs.update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                                    for sup in supersets_cache[t_old]:
                                        cover_count[sup] += 1
                        
                        if not found:
                            # Restore
                            used_pairs.update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                            for sup in supersets_cache[t_old]:
                                cover_count[sup] += 1
                            triples[idx] = t_old
                        
                        if found:
                            break
            
            # Early termination if we hit a good value
            if best_alpha <= 5:
                break
    
    return best_triples
