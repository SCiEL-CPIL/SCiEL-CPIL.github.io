"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _make_bitmasks():
    """Generate all 4-element subsets of [1,16] as bitmasks, indexed."""
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    return all_sets

def _is_sunflower(a, b, c):
    """Check if three 4-bitmasks form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _check_conflict(mask, chosen):
    """Check if adding mask to chosen creates a sunflower triple."""
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            if _is_sunflower(chosen[i], chosen[j], mask):
                return True
    return False

def _greedy_build(all_masks, order):
    """Greedy: add masks in given order if no conflict."""
    chosen = []
    for mask in order:
        if not _check_conflict(mask, chosen):
            chosen.append(mask)
    return chosen

def _local_search(chosen, all_masks, time_limit):
    """Try to improve by swapping or adding/removing sets."""
    chosen_set = set(chosen)
    start = time.time()
    
    for iteration in range(50):
        if time.time() - start > time_limit:
            break
        
        # Try adding a random unchosen mask
        unchosen = [m for m in all_masks if m not in chosen_set]
        if unchosen:
            random.shuffle(unchosen)
            for mask in unchosen[:20]:
                if not _check_conflict(mask, chosen):
                    chosen.append(mask)
                    chosen_set.add(mask)
                    break
        
        # Try swapping: remove one, add two
        if len(chosen) < 250:
            continue
        
        # Try single swap
        for i in range(len(chosen)):
            if time.time() - start > time_limit:
                break
            removed = chosen[i]
            chosen_temp = chosen[:i] + chosen[i+1:]
            chosen_temp_set = set(chosen_temp)
            
            unchosen = [m for m in all_masks if m not in chosen_temp_set]
            if not unchosen:
                continue
            random.shuffle(unchosen)
            
            for mask in unchosen[:10]:
                if not _check_conflict(mask, chosen_temp):
                    chosen_temp.append(mask)
                    chosen = chosen_temp
                    chosen_set = set(chosen)
                    break

    return chosen

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    all_masks = _make_bitmasks()
    random.seed(42)
    
    best = []
    
    # Multiple greedy passes with different orderings
    for trial in range(5):
        if time.time() > deadline:
            break
        
        order = all_masks[:]
        random.shuffle(order)
        
        chosen = _greedy_build(all_masks, order)
        
        if len(chosen) > len(best):
            best = chosen[:]
        
        # Local search improvement
        if time.time() > deadline - 10:
            break
        improved = _local_search(best, all_masks, max(0, deadline - time.time() - 5))
        if len(improved) > len(best):
            best = improved
    
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best:
        elems = [i+1 for i in range(16) if mask & (1 << i)]
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
