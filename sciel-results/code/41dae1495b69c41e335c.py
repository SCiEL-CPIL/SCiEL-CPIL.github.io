import random
import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Points 0..15
    points = list(range(16))
    
    # Pre-compute all triples as bitmasks for speed
    all_triples = list(itertools.combinations(points, 3))
    triple_masks = []
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Helper to check linearity using bitmasks
    def is_linear(triples):
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                if triple_masks[i] & triple_masks[j] and (triple_masks[i] & triple_masks[j]) != (1 << min(min(triples[i]), min(triples[j]))) and (triple_masks[i] & triple_masks[j]) != (1 << max(min(triples[i]), min(triples[j]))):
                    # More robust: check if intersection has more than 1 bit
                    inter = triple_masks[i] & triple_masks[j]
                    if inter & (inter - 1): # More than one bit set
                        return False
        return True

    # Helper to calculate alpha (independence number)
    def calc_alpha(triple_indices):
        if not triple_indices:
            return 16
        
        # Create a boolean array or use bit operations
        # For each subset mask, check if it contains any triple
        # We can precompute for each triple index, but simpler:
        # Iterate masks from high popcount down, return first independent
        
        # Sort by popcount descending to find max alpha quickly
        # But we need to check all to be sure? No, we can break early if we find one of size k.
        # We want the MAX size. So we check sizes 16, 15, ..., 1.
        
        # To make it fast, we can precompute a list of masks sorted by popcount?
        # Or just iterate all masks and keep max. 2^16 is 65536.
        # 65536 * len(triples) is too slow if len(triples) is large.
        
        # Better: For each triple, mark all supersets as "covered".
        # Then find the largest mask not covered.
        
        covered = [False] * (1 << 16)
        
        # For each triple in the system, mark all supersets
        for idx in triple_indices:
            t_mask = triple_masks[idx]
            # Supersets of t_mask
            # Iterate over all masks that are supersets of t_mask
            # This is still expensive if done naively.
            
            # Alternative: For each mask, check if it contains any triple.
            # We can optimize by breaking early.
            
        # Let's stick to the direct check but optimize the inner loop.
        # Precompute a list of (mask, popcount) sorted by popcount descending?
        # No, we can just iterate all masks.
        
        max_alpha = 0
        # To speed up, we can precompute popcounts
        popcounts = [0] * (1 << 16)
        for i in range(1, 1 << 16):
            popcounts[i] = popcounts[i >> 1] + (i & 1)
            
        # We want to find the largest mask such that no triple is a subset.
        # Iterate masks in descending order of popcount?
        # That requires sorting 65536 items, which is okay once, but we call this many times.
        
        # Let's just iterate all masks.
        for mask in range(1 << 16):
            is_independent = True
            for idx in triple_indices:
                t_mask = triple_masks[idx]
                if (mask & t_mask) == t_mask:
                    is_independent = False
                    break
            if is_independent:
                pc = popcounts[mask]
                if pc > max_alpha:
                    max_alpha = pc
        
        return max_alpha

    # Helper to check if a triple can be added to a set
    # triples is a list of indices
    def can_add(triples, new_idx):
        new_mask = triple_masks[new_idx]
        for idx in triples:
            if (new_mask & triple_masks[idx]) and ((new_mask & triple_masks[idx]) & (new_mask & triple_masks[idx] - 1)):
                return False
        return True

    # Random initial maximal packing
    def random_initial():
        triples = []
        available = list(range(len(all_triples)))
        random.shuffle(available)
        for idx in available:
            if can_add(triples, idx):
                triples.append(idx)
        return triples

    # Local Search
    def local_search(triples):
        current_alpha = calc_alpha(triples)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try adding a triple
            candidates = [idx for idx in range(len(all_triples)) if can_add(triples, idx)]
            if candidates:
                random.shuffle(candidates)
                best_idx = None
                best_a = current_alpha
                # Try a few candidates
                for idx in candidates[:20]:
                    new_triples = triples + [idx]
                    a = calc_alpha(new_triples)
                    if a < best_a:
                        best_a = a
                        best_idx = idx
                if best_idx is not None:
                    triples.append(best_idx)
                    current_alpha = best_a
                    improved = True
                    continue
            
            # 2. Try removing a triple (if alpha is high, maybe we need to restructure?)
            # Usually removing increases alpha, so we only do this if we want to swap.
            
            # 3. Try swapping: remove one, add one
            # This is complex. Let's try a simple kick: remove a random triple, then try to add.
            if len(triples) > 0:
                # Pick a random triple to remove
                remove_idx = random.choice(triples)
                triples.remove(remove_idx)
                
                # Now try to add a better triple
                candidates = [idx for idx in range(len(all_triples)) if can_add(triples, idx)]
                if candidates:
                    random.shuffle(candidates)
                    best_idx = None
                    best_a = current_alpha # Note: alpha might have increased after removal
                    # Re-calculate alpha after removal?
                    # To save time, just try to find a triple that leads to lower alpha than original best?
                    # Or just try to maximize density?
                    # Let's just try to find any triple that reduces alpha below the pre-removal alpha?
                    # Actually, we should compare against the current state's alpha.
                    current_alpha_after_removal = calc_alpha(triples)
                    
                    for idx in candidates[:20]:
                        new_triples = triples + [idx]
                        a = calc_alpha(new_triples)
                        if a < best_a:
                            best_a = a
                            best_idx = idx
                    if best_idx is not None:
                        triples.append(best_idx)
                        current_alpha = best_a
                        improved = True
                    else:
                        # Revert? Or keep removed if it helped?
                        # If we didn't add, alpha is current_alpha_after_removal.
                        # If current_alpha_after_removal < original current_alpha, keep it.
                        # But we lost track of original.
                        # Let's just revert if we didn't add to maintain linearity and size?
                        # Actually, if we removed and didn't add, the system is smaller, alpha is larger.
                        # So we should revert unless the removal itself lowered alpha (impossible).
                        # So we must add or revert.
                        triples.append(remove_idx)
                        # current_alpha remains the pre-removal value?
                        # We didn't update it.
                        pass

        return triples, current_alpha

    best_alpha = 16
    best_triples = []
    
    # Main loop
    while time.time() < deadline:
        init_triples = random_initial()
        improved_triples, alpha = local_search(init_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved_triples
            if best_alpha <= 5:
                break

    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return [list(all_triples[i]) for i in best_triples]

# EVOLVE-BLOCK-END
