"""
Maximize comparable pairs among 40 distinct subsets of {0..11}.
Strategy: Efficient dynamic local search + Simulated Annealing.
"""
import time
import random
import itertools

N = 12
M = 40
NUM_MASKS = 1 << N  # 4096
ALL_MASKS = list(range(NUM_MASKS))
DEADLINE_MARGIN = 5.0  # seconds safety margin

def _precompute_comparability():
    """
    Precompute for each mask m:
    - comp_mask[m]: a 4096-bit integer where bit j is set iff m and j are strictly comparable
    """
    comp_mask = [0] * NUM_MASKS
    
    for m1 in range(NUM_MASKS):
        for m2 in range(m1 + 1, NUM_MASKS):
            if (m1 & m2) == m1:
                if m1 != m2:
                    comp_mask[m1] |= (1 << m2)
                    comp_mask[m2] |= (1 << m1)
            elif (m1 & m2) == m2:
                if m1 != m2:
                    comp_mask[m1] |= (1 << m2)
                    comp_mask[m2] |= (1 << m1)
    
    return comp_mask

def _count_comparable(family_masks, comp_mask):
    """Count unordered pairs where one is strict subset of the other."""
    n = len(family_masks)
    count = 0
    for i in range(n):
        mi = family_masks[i]
        for j in range(i + 1, n):
            mj = family_masks[j]
            if (comp_mask[mi] >> mj) & 1:
                count += 1
    return count

def _greedy_construct(seed, comp_mask):
    """
    Greedy construction: start with a random mask, then repeatedly add the mask
    that maximizes comparable pairs with the current family.
    """
    rng = random.Random(seed)
    
    in_family = [False] * NUM_MASKS
    family = []
    
    # Pick a random starting mask
    first = rng.choice(ALL_MASKS)
    family.append(first)
    in_family[first] = True
    
    # Score for each mask: number of family members it's comparable with
    scores = [0] * NUM_MASKS
    for y in ALL_MASKS:
        if y == first:
            continue
        if (comp_mask[first] >> y) & 1:
            scores[y] = 1
    
    for _ in range(M - 1):
        best_score = -1
        best_candidates = []
        for y in ALL_MASKS:
            if not in_family[y]:
                if scores[y] > best_score:
                    best_score = scores[y]
                    best_candidates = [y]
                elif scores[y] == best_score:
                    best_candidates.append(y)
        
        if not best_candidates:
            break
        
        chosen = rng.choice(best_candidates)
        family.append(chosen)
        in_family[chosen] = True
        
        # Update scores
        for y in ALL_MASKS:
            if not in_family[y]:
                if (comp_mask[chosen] >> y) & 1:
                    scores[y] += 1
    
    # Fill if needed
    if len(family) < M:
        remaining = [m for m in ALL_MASKS if not in_family[m]]
        rng.shuffle(remaining)
        family.extend(remaining[:M - len(family)])
    
    return family

def _local_search_1swap(family, comp_mask, deadline):
    """
    Efficient 1-swap local search with dynamic score updates.
    """
    family_set = set(family)
    current_count = _count_comparable(family, comp_mask)
    
    # Precompute scores: for each mask not in family, how many family members it's comparable with
    scores = [0] * NUM_MASKS
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        for f in family:
            if (comp_mask[m] >> f) & 1:
                scores[m] += 1
    
    # Also compute scores for family members (d(A, F))
    family_scores = [0] * M
    for i in range(M):
        f = family[i]
        for j in range(M):
            if i == j: continue
            if (comp_mask[f] >> family[j]) & 1:
                family_scores[i] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # For each i, find the best swap
        best_total_gain = 0
        best_i = -1
        best_add = -1
        
        for i in range(M):
            removed = family[i]
            old_comp = family_scores[i]
            
            # Find best candidate to add
            best_gain = 0
            best_add_candidate = -1
            for m in range(NUM_MASKS):
                if m in family_set:
                    continue
                # gain = scores[m] - is_comp_with_removed - old_comp
                is_comp_with_removed = (comp_mask[m] >> removed) & 1
                gain = scores[m] - is_comp_with_removed - old_comp
                if gain > best_gain:
                    best_gain = gain
                    best_add_candidate = m
            
            if best_gain > best_total_gain:
                best_total_gain = best_gain
                best_i = i
                best_add = best_add_candidate
        
        if best_total_gain > 0 and best_i != -1 and best_add != -1:
            # Perform swap
            removed = family[best_i]
            added = best_add
            
            family[best_i] = added
            family_set.remove(removed)
            family_set.add(added)
            
            # Update scores for non-family masks
            for m in range(NUM_MASKS):
                if m in family_set:
                    continue
                if (comp_mask[m] >> removed) & 1:
                    scores[m] -= 1
                if (comp_mask[m] >> added) & 1:
                    scores[m] += 1
            
            # Update family_scores
            # The score of the added element in the new family
            new_score_added = 0
            for j in range(M):
                if j == best_i: continue
                if (comp_mask[added] >> family[j]) & 1:
                    new_score_added += 1
            family_scores[best_i] = new_score_added
            
            # The scores of other family elements change if they were comparable with removed or added
            for j in range(M):
                if j == best_i: continue
                if (comp_mask[family[j]] >> removed) & 1:
                    family_scores[j] -= 1
                if (comp_mask[family[j]] >> added) & 1:
                    family_scores[j] += 1
            
            current_count += best_total_gain
            improved = True
    
    return family, current_count

def _simulated_annealing(family, comp_mask, deadline, seed):
    """
    Simulated annealing to escape local optima.
    """
    rng = random.Random(seed)
    family = list(family)
    family_set = set(family)
    current_count = _count_comparable(family, comp_mask)
    best_count = current_count
    best_family = list(family)
    
    # Precompute scores
    scores = [0] * NUM_MASKS
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        for f in family:
            if (comp_mask[m] >> f) & 1:
                scores[m] += 1
    
    # Family scores
    family_scores = [0] * M
    for i in range(M):
        f = family[i]
        for j in range(M):
            if i == j: continue
            if (comp_mask[f] >> family[j]) & 1:
                family_scores[i] += 1

    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Random 1-swap move
        i = rng.randint(0, M - 1)
        removed = family[i]
        
        # Pick random non-family mask
        # Optimization: maintain list of non-family masks? 
        # For simplicity, pick random mask and check, or use a list.
        # Given 4096 masks and 40 in family, rejection sampling is fast enough (4096/40 ~ 100 tries? No, 4096-40=4056 valid. 1/4056 chance of hit? No, 40/4096 ~ 1% invalid. So ~1.01 tries).
        while True:
            add = rng.randint(0, NUM_MASKS - 1)
            if add not in family_set:
                break
        
        old_comp = family_scores[i]
        is_comp_with_removed = (comp_mask[add] >> removed) & 1
        gain = scores[add] - is_comp_with_removed - old_comp
        
        # Accept or reject
        if gain >= 0 or (gain > -20 * T and rng.random() < 2.718281828 ** (gain / T)):
            # Accept
            family[i] = add
            family_set.remove(removed)
            family_set.add(add)
            
            # Update scores
            for m in range(NUM_MASKS):
                if m in family_set:
                    continue
                if (comp_mask[m] >> removed) & 1:
                    scores[m] -= 1
                if (comp_mask[m] >> add) & 1:
                    scores[m] += 1
            
            # Update family_scores
            new_score_added = 0
            for j in range(M):
                if j == i: continue
                if (comp_mask[add] >> family[j]) & 1:
                    new_score_added += 1
            family_scores[i] = new_score_added
            
            for j in range(M):
                if j == i: continue
                if (comp_mask[family[j]] >> removed) & 1:
                    family_scores[j] -= 1
                if (comp_mask[family[j]] >> add) & 1:
                    family_scores[j] += 1
            
            current_count += gain
            
            if current_count > best_count:
                best_count = current_count
                best_family = list(family)
        
        T *= cooling
    
    return best_family, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    # Precompute comparability
    comp_mask = _precompute_comparability()
    
    best_count = 0
    best_family = None
    
    # Phase 1: Multiple greedy restarts
    # We have ~85 seconds.
    # Each greedy is fast (~0.1s). Local search is ~0.5s. SA is ~5s.
    # Let's do 50 restarts of greedy + local search.
    # Then take the best few and run SA on them.
    
    candidates = []
    
    restart_deadline = deadline - 20.0
    
    for i in range(100):
        if time.time() > restart_deadline:
            break
        
        family = _greedy_construct(i, comp_mask)
        # Quick local search
        ls_deadline = time.time() + 0.5
        family, count = _local_search_1swap(family, comp_mask, ls_deadline)
        
        if count > best_count:
            best_count = count
            best_family = list(family)
        
        candidates.append((count, list(family)))
        
        if len(candidates) > 20:
            # Keep top 20
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[:20]
    
    # Phase 2: Simulated Annealing on top candidates
    if candidates:
        sa_deadline_per_candidate = (deadline - time.time()) / len(candidates)
        if sa_deadline_per_candidate < 1.0:
            # Not enough time for many, just do the best one
            candidates = [candidates[0]]
            sa_deadline_per_candidate = deadline - time.time()
        
        for count, family in candidates:
            if time.time() > deadline:
                break
            
            sa_deadline = time.time() + sa_deadline_per_candidate
            family, count = _simulated_annealing(family, comp_mask, sa_deadline, 42)
            
            if count > best_count:
                best_count = count
                best_family = list(family)
    
    # Final local search on the best family found
    if best_family is not None:
        final_deadline = deadline - 0.5
        best_family, best_count = _local_search_1swap(best_family, comp_mask, final_deadline)
    
    # Fallback
    if best_family is None:
        vertices = list(range(6))
        all_sets = []
        for r in range(7):
            for combo in itertools.combinations(vertices, r):
                all_sets.append(list(combo))
        all_sets.sort(key=len)
        best_family = all_sets[:M]
    
    # Convert masks to lists of ints
    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
