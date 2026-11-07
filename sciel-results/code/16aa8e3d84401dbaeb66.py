import time
import random
import itertools

def construct():
    n = 12
    m = 40
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    # Precompute comparability for all pairs of masks
    # total_masks = 4096
    total_masks = 1 << n
    
    # For each mask, precompute the list of masks it is comparable with
    # comparable[i] is a set/list of j such that i and j are comparable (one is subset of other)
    # But storing 4096 * 4096 bits is 2M bits = 256KB, which is fine.
    # However, computing it naively is 16M operations.
    # Let's optimize: 
    # A mask 'a' is a subset of 'b' iff (a & b) == a.
    # We can precompute for each mask 'a', the list of all 'b' such that a is subset of b.
    # Then comparable(a,b) is true if b in superset[a] OR a in superset[b].
    
    # Actually, a faster approach for local search:
    # Represent the family as a list of 40 masks.
    # Score = sum over i<j of [ (fam[i] & fam[j]) == fam[i] or (fam[i] & fam[j]) == fam[j] ]
    
    # To speed up evaluation of a move (replacing mask i with new_mask):
    # Delta = - (comparable(new_mask, old) for j != i) + (comparable(new_mask, fam[j]) for j != i)
    # Wait, we replace fam[i] with new_mask.
    # Old contribution involving i: sum_{j != i} C(fam[i], fam[j])
    # New contribution involving i: sum_{j != i} C(new_mask, fam[j])
    # So Delta = sum_{j != i} (C(new_mask, fam[j]) - C(fam[i], fam[j]))
    
    # Precompute C(a, b) for all a, b? No, that's 16M lookups.
    # But we can precompute for a fixed new_mask, how many j in current family are comparable to it.
    
    # Let's try a simple local search first.
    # Initialize with a random valid family (distinct masks).
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def count_comparable(fam, idx, new_mask):
        # Count how many j != idx are comparable to new_mask
        c = 0
        for j in range(m):
            if j == idx:
                continue
            if is_comparable(new_mask, fam[j]):
                c += 1
        return c

    def total_score(fam):
        s = 0
        for i in range(m):
            for j in range(i+1, m):
                if is_comparable(fam[i], fam[j]):
                    s += 1
        return s

    # Initial family: 
    # A good heuristic: pick masks of similar sizes? 
    # Or start with a chain? 
    # Let's start with a random set of 40 distinct masks.
    # Or better: start with a greedy construction.
    
    # Greedy: 
    # Add masks one by one, picking the one that maximizes comparable pairs with existing set.
    
    def greedy_construct():
        fam = []
        used = set()
        # Precompute all masks
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)
        
        # Start with empty set? Or a random one?
        # Let's try starting with mask 0 (empty set) as it is subset of everything.
        # Then add masks that contain it? No, we want distinct subsets.
        # The empty set is comparable to ALL other sets.
        # So including empty set is always good (adds m-1 pairs).
        # Similarly, the full set (all 12 bits) is comparable to all.
        
        # Let's force include empty set and full set?
        # If we include empty set, it contributes 39 pairs.
        # If we include full set, it contributes 39 pairs.
        # But empty and full are comparable to each other.
        
        # Let's try a different init: 
        # Pick 40 masks randomly, but biased towards smaller or larger sets?
        # Actually, let's just do random restarts with local search.
        
        pass

    # Local Search
    best_fam = None
    best_score = -1
    
    # Try a few random initializations
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Initialize
        # Strategy: Pick masks of varying sizes?
        # Let's pick 40 distinct random masks.
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)
        fam = all_masks[:m]
        
        # Optional: Ensure diversity? 
        # Let's just run local search.
        
        cur_score = total_score(fam)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomly shuffle order of indices to avoid bias
            indices = list(range(m))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                old_mask = fam[i]
                old_contrib = 0
                for j in range(m):
                    if j != i and is_comparable(old_mask, fam[j]):
                        old_contrib += 1
                
                # Try replacing with a random new mask
                # To be efficient, we should pick new_mask from a pool.
                # Which pool? 
                # Maybe masks that are "close" to old_mask?
                # Or just random?
                # Let's try random first.
                
                # Optimization: Precompute a list of candidate new masks?
                # Or just pick a few random ones.
                
                # Let's try a specific candidate: 
                # 1. Random mask
                # 2. old_mask with one bit flipped?
                
                # Let's just do random mask for now.
                new_mask = random.randint(0, total_masks - 1)
                if new_mask == old_mask:
                    continue
                
                new_contrib = 0
                for j in range(m):
                    if j != i and is_comparable(new_mask, fam[j]):
                        new_contrib += 1
                
                delta = new_contrib - old_contrib
                if delta > 0:
                    fam[i] = new_mask
                    cur_score += delta
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_fam = fam[:]
        
        if cur_score > best_score:
            best_score = cur_score
            best_fam = fam[:]
        
        # If we found a perfect score, break?
        # Max possible pairs: C(40, 2) = 780.
        if best_score >= 780:
            break

    # Convert best_fam to list of lists of integers
    result = []
    for mask in best_fam:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
