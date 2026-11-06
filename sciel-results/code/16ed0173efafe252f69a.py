# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s safety margin before 60s limit
    
    # Precompute all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    mask_to_idx = {}
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
        mask_to_idx[mask] = len(all_masks) - 1
    
    N = len(all_masks)
    
    def is_sunflower(a, b, c):
        """Check if a, b, c form a sunflower (pairwise intersections are equal sets)."""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc

    def build_family_greedy(seed):
        random.seed(seed)
        indices = list(range(N))
        random.shuffle(indices)
        
        family = []
        
        for idx in indices:
            if time.time() > deadline:
                break
            m = all_masks[idx]
            valid = True
            # Check against all pairs in family
            for i in range(len(family)):
                if not valid: break
                mi = all_masks[family[i]]
                for j in range(i + 1, len(family)):
                    mj = all_masks[family[j]]
                    if is_sunflower(mi, mj, m):
                        valid = False
                        break
            
            if valid:
                family.append(idx)
                
        return family

    def local_search(family, seed):
        """Simple swap local search."""
        random.seed(seed)
        family = family[:]
        family_set = set(family)
        available = [i for i in range(N) if i not in family_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to swap one out, one in
            random.shuffle(family)
            for a_idx in family:
                if time.time() > deadline:
                    break
                # Remove a_idx
                family.remove(a_idx)
                family_set.discard(a_idx)
                
                # Try to add any available that doesn't conflict
                found_add = False
                random.shuffle(available)
                for b_idx in available:
                    mb = all_masks[b_idx]
                    valid = True
                    # Check pairs in current family (which is F \ {a})
                    for i in range(len(family)):
                        if not valid: break
                        mi = all_masks[family[i]]
                        for j in range(i + 1, len(family)):
                            mj = all_masks[family[j]]
                            if is_sunflower(mi, mj, mb):
                                valid = False
                                break
                    if valid:
                        # Add b_idx
                        family.append(b_idx)
                        family_set.add(b_idx)
                        available.remove(b_idx)
                        # a_idx is now available
                        available.append(a_idx)
                        improved = True
                        found_add = True
                        break
                        
                if not found_add:
                    # Re-add a_idx
                    family.append(a_idx)
                    family_set.add(a_idx)
                    # a_idx was removed from family, so it's not in available.
                    # We don't add it to available because it's back in family.
                    # Wait, if we remove a_idx from family, it should be in available?
                    # No, available is the complement of family.
                    # Initially available = complement of family.
                    # We removed a_idx from family. a_idx is NOT in available.
                    # We tried to add b_idx. If failed, we didn't touch available.
                    # So we just put a_idx back in family.
                    # a_idx is still not in available. Correct.
        
        return family

    best_family = []
    
    # Phase 1: Greedy with multiple seeds
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline - 10:
            break
        fam = build_family_greedy(seed)
        if len(fam) > len(best_family):
            best_family = fam
    
    # Phase 2: Local search on best
    if best_family and time.time() < deadline - 5:
        improved_fam = local_search(best_family, 42)
        if len(improved_fam) > len(best_family):
            best_family = improved_fam
            
    # Phase 3: More restarts if time permits
    for seed in range(num_restarts, num_restarts + 30):
        if time.time() > deadline - 2:
            break
        fam = build_family_greedy(seed)
        if time.time() < deadline - 2:
            fam = local_search(fam, seed + 100)
        if len(fam) > len(best_family):
            best_family = fam
            
    # Convert to result format
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
# EVOLVE-BLOCK-END
