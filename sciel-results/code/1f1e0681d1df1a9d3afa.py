"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

N = 12
M = 40
FULL_MASK = (1 << N) - 1

def mask_to_list(mask):
    return [b for b in range(N) if (mask >> b) & 1]

def list_to_mask(lst):
    m = 0
    for v in lst:
        m |= (1 << v)
    return m

def is_subset(a, b):
    # a subset of b (strict? no, just subset)
    return (a & b) == a

def is_strict_subset(a, b):
    return a != b and (a & b) == a

def count_pairs(masks):
    # masks: list of ints
    cnt = 0
    n = len(masks)
    for i in range(n):
        ai = masks[i]
        for j in range(i + 1, n):
            aj = masks[j]
            if (ai & aj) == ai and ai != aj:
                cnt += 1
            elif (ai & aj) == aj and ai != aj:
                cnt += 1
    return cnt

def incremental_score(masks):
    # Compute total comparable pairs
    return count_pairs(masks)

def greedy_init():
    # Start with empty set and build up
    # We want a chain-like structure to maximize comparability
    # A full chain of 13 sets (0..12 elements) gives 78 pairs
    # But we need 40 distinct sets.
    
    # Strategy: Pick a "spine" of sets that are comparable, then add others that fit into gaps or are comparable to many.
    # Actually, the maximum number of comparable pairs in a family of size M from a Boolean lattice of size N
    # is achieved by taking an initial segment of the poset ordered by inclusion? No, that's for antichains (Sperner).
    # For maximizing comparable pairs, we want a "chain" as much as possible.
    # A chain has size N+1 = 13. Pairs in a chain: 13*12/2 = 78.
    # We need 40 sets.
    # If we take a chain of 13, we have 78 pairs.
    # We need 27 more sets.
    # Where to add them?
    # If we add a set that is comparable to k existing sets, we gain k pairs.
    # To maximize, we should add sets that are comparable to as many existing sets as possible.
    
    # Let's start with a maximal chain: {}, {0}, {0,1}, {0,1,2}, ..., {0..11}
    chain = []
    cur = 0
    for i in range(N + 1):
        chain.append(cur)
        if i < N:
            cur |= (1 << i)
    
    # Now we have 13 sets. We need 27 more.
    # The sets in the chain are S_0, S_1, ..., S_12 where S_i has size i and contains {0..i-1}.
    # Any other set T is comparable to S_i if T subset S_i or S_i subset T.
    # Since S_i are nested, if T is comparable to S_i and S_j (i<j), it must be comparable to the "middle" ones too?
    # Not necessarily, but likely.
    
    # Let's generate all possible sets and score them by how many chain members they are comparable to.
    # Then pick the top 27.
    
    all_masks = range(1 << N)
    candidates = []
    
    # Precompute chain membership for speed? No, just loop.
    for m in all_masks:
        if m in chain:
            continue
        comp_count = 0
        for cm in chain:
            if (m & cm) == m or (m & cm) == cm:
                comp_count += 1
        candidates.append((comp_count, m))
    
    # Sort by comp_count descending
    candidates.sort(key=lambda x: -x[0])
    
    # Pick top 27
    selected = []
    for _, m in candidates:
        if len(selected) >= M - len(chain):
            break
        # Check if adding this set creates duplicates? No, distinct from chain.
        # But we must ensure distinctness among selected too.
        if m not in selected:
            selected.append(m)
    
    family = chain + selected
    # Ensure exactly M
    if len(family) > M:
        family = family[:M]
    
    return family

def local_search(family, time_limit):
    masks = family[:]
    best_score = count_pairs(masks)
    start_time = time.time()
    
    # Precompute subset relationships?
    # For incremental update:
    # When replacing mask[i] with new_mask:
    # Delta = (new comparable pairs with others) - (old comparable pairs with others)
    
    def delta_score(i, new_mask, masks):
        old_mask = masks[i]
        if old_mask == new_mask:
            return 0
        
        d = 0
        for j in range(M):
            if i == j:
                continue
            mj = masks[j]
            
            # Check old
            old_comp = 0
            if (old_mask & mj) == old_mask and old_mask != mj:
                old_comp = 1
            elif (old_mask & mj) == mj and old_mask != mj:
                old_comp = 1
            
            # Check new
            new_comp = 0
            if (new_mask & mj) == new_mask and new_mask != mj:
                new_comp = 1
            elif (new_mask & mj) == mj and new_mask != mj:
                new_comp = 1
            
            d += new_comp - old_comp
        
        return d

    # Candidate generation:
    # 1. Random mask
    # 2. Neighbor of existing mask (flip 1 bit)
    # 3. Random subset/superset of existing mask
    
    # To speed up, we can precompute the score contribution of each mask to the total?
    # Score = sum_{i<j} comp(i,j).
    # If we remove i, we lose sum_{j!=i} comp(i,j).
    # If we add i', we gain sum_{j!=i} comp(i',j).
    # This is O(M) per candidate evaluation.
    # M=40. 1600 pairs.
    # Evaluating a candidate takes O(M) = 40 ops.
    # We can do many iterations.
    
    # Let's try to be smarter.
    # We can maintain an array `comp_count[j]` = number of masks in family comparable to masks[j].
    # Then total score = sum(comp_count[j]) / 2.
    # When removing i:
    #   For each j != i, if comp(i,j), then comp_count[j] decreases by 1.
    #   The cost of removing i is sum_{j!=i} comp(i,j).
    # When adding i' (new mask):
    #   For each j != i, if comp(i',j), then comp_count[j] increases by 1.
    #   The gain is sum_{j!=i} comp(i',j).
    #   Note: j includes the position where i was removed? No, i is removed first.
    #   So the set of "others" is the same.
    
    # Let's implement this incremental structure.
    
    # Re-initialize comp_count
    comp_count = [0] * M
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            mi = masks[i]
            mj = masks[j]
            if (mi & mj) == mi and mi != mj:
                comp_count[i] += 1
            elif (mi & mj) == mj and mi != mj:
                comp_count[i] += 1
    
    current_score = sum(comp_count) // 2
    
    # Verify
    assert current_score == best_score, f"Score mismatch: {current_score} vs {best_score}"
    
    # Local search loop
    # We will try to replace each mask with a better one.
    # Candidate generation for mask i:
    # - Random mask
    # - Mask with one bit flipped
    # - Mask with one bit added/removed
    
    # To avoid duplicates, we check if new_mask is already in masks (excluding i).
    
    # Time limit guard
    deadline = start_time + time_limit
    
    while time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(M))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            old_mask = masks[i]
            
            # Generate candidates
            candidates = set()
            
            # 1. Random
            candidates.add(random.randint(0, FULL_MASK))
            
            # 2. Neighbors (flip 1 bit)
            for b in range(N):
                candidates.add(old_mask ^ (1 << b))
            
            # 3. Add/remove one bit (subset/superset of size 1 diff)
            # Already covered by flip? No, flip changes size by +/- 1.
            # So neighbors cover size +/- 1.
            
            # 4. Random subset/superset
            # Pick random size
            for _ in range(5):
                # Random subset
                subset = 0
                for b in range(N):
                    if (old_mask >> b) & 1 and random.random() < 0.5:
                        subset |= (1 << b)
                candidates.add(subset)
                
                # Random superset
                superset = old_mask
                for b in range(N):
                    if not ((old_mask >> b) & 1) and random.random() < 0.5:
                        superset |= (1 << b)
                candidates.add(superset)
            
            # Remove old mask from consideration of duplicates?
            # We need to ensure new_mask is not in masks[j] for j != i.
            # Let's create a set of existing masks excluding i.
            existing = set(masks[j] for j in range(M) if j != i)
            
            best_delta = 0
            best_new_mask = None
            
            for cand in candidates:
                if cand in existing:
                    continue
                
                # Calculate delta
                # Gain = sum_{j != i} comp(cand, masks[j])
                # Loss = sum_{j != i} comp(old_mask, masks[j])
                
                # We can use comp_count?
                # comp_count[j] includes comp(i,j).
                # So sum_{j!=i} comp(i,j) = comp_count[i].
                # Wait, comp_count[i] is the number of j such that comp(i,j).
                # Yes.
                
                loss = comp_count[i]
                
                gain = 0
                for j in range(M):
                    if i == j:
                        continue
                    mj = masks[j]
                    if (cand & mj) == cand and cand != mj:
                        gain += 1
                    elif (cand & mj) == mj and cand != mj:
                        gain += 1
                
                delta = gain - loss
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = cand
            
            if best_new_mask is not None and best_delta > 0:
                # Update
                # 1. Remove old_mask from comp_count
                for j in range(M):
                    if i == j:
                        continue
                    mj = masks[j]
                    if (old_mask & mj) == old_mask and old_mask != mj:
                        comp_count[j] -= 1
                    elif (old_mask & mj) == mj and old_mask != mj:
                        comp_count[j] -= 1
                
                # 2. Replace
                masks[i] = best_new_mask
                current_score += best_delta
                
                # 3. Add best_new_mask to comp_count
                for j in range(M):
                    if i == j:
                        continue
                    mj = masks[j]
                    if (best_new_mask & mj) == best_new_mask and best_new_mask != mj:
                        comp_count[j] += 1
                    elif (best_new_mask & mj) == mj and best_new_mask != mj:
                        comp_count[j] += 1
                
                # 4. Update comp_count[i]
                # comp_count[i] was loss. Now it should be gain.
                comp_count[i] = gain
                
                improved = True
        
        if not improved:
            # Try a few random restarts or perturbations if stuck?
            # Or just break if we've been stuck for a while?
            # For now, let's do a random perturbation.
            # Pick a random index and replace with a random mask, even if it decreases score, to escape local optima.
            # But we must be careful not to degrade too much.
            # Let's try 100 random perturbations.
            for _ in range(100):
                if time.time() > deadline:
                    break
                i = random.randint(0, M - 1)
                old_mask = masks[i]
                new_mask = random.randint(0, FULL_MASK)
                
                existing = set(masks[j] for j in range(M) if j != i)
                if new_mask in existing:
                    continue
                
                loss = comp_count[i]
                gain = 0
                for j in range(M):
                    if i == j:
                        continue
                    mj = masks[j]
                    if (new_mask & mj) == new_mask and new_mask != mj:
                        gain += 1
                    elif (new_mask & mj) == mj and new_mask != mj:
                        gain += 1
                
                delta = gain - loss
                
                # Accept if delta > 0 or with some probability (simulated annealing style)
                # Since we don't have temperature, let's just accept if delta >= -5 (small degradation)
                if delta >= -5:
                    # Update
                    for j in range(M):
                        if i == j:
                            continue
                        mj = masks[j]
                        if (old_mask & mj) == old_mask and old_mask != mj:
                            comp_count[j] -= 1
                        elif (old_mask & mj) == mj and old_mask != mj:
                            comp_count[j] -= 1
                    
                    masks[i] = new_mask
                    current_score += delta
                    
                    for j in range(M):
                        if i == j:
                            continue
                        mj = masks[j]
                        if (new_mask & mj) == new_mask and new_mask != mj:
                            comp_count[j] += 1
                        elif (new_mask & mj) == mj and new_mask != mj:
                            comp_count[j] += 1
                    
                    comp_count[i] = gain
                    break # Restart the main loop after perturbation
        
        if current_score > best_score:
            best_score = current_score
            best_masks = masks[:]
    
    return best_masks

def construct():
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin
    
    # 1. Greedy Init
    family = greedy_init()
    
    # 2. Local Search
    best_masks = local_search(family, time_limit)
    
    # 3. Convert to lists of ints
    result = []
    for m in best_masks:
        result.append(mask_to_list(m))
    
    # Ensure types are correct
    # "integers must be Python int"
    # "sequences must be list or tuple"
    
    return result

# EVOLVE-BLOCK-END
