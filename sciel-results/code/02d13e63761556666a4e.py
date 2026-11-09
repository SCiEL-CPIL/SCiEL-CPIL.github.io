"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Precompute all 4-element subsets as bitmasks
    all_sets = [frozenset(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    all_sets_mask = [0] * n_sets
    for i, s in enumerate(all_sets):
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        all_sets_mask[i] = mask
    
    mask_to_set = {all_sets_mask[i]: all_sets[i] for i in range(n_sets)}
    
    def _greedy(order_indices, deadline):
        """Greedy construction with fast bitmask conflict checking."""
        chosen_masks = []
        chosen_pairs = []  # (mask_i, mask_j, core) for each pair of chosen sets
        
        for idx in order_indices:
            if time.time() > deadline:
                break
            
            C = all_sets_mask[idx]
            conflict = False
            
            for (Ai, Bj, core) in chosen_pairs:
                if (Ai & C) == core and (Bj & C) == core:
                    conflict = True
                    break
            
            if not conflict:
                for Ai in chosen_masks:
                    core = Ai & C
                    chosen_pairs.append((Ai, C, core))
                chosen_masks.append(C)
        
        return [mask_to_set[m] for m in chosen_masks]
    
    best_family = []
    
    # Try lexicographic order first
    if time.time() < deadline:
        family = _greedy(list(range(n_sets)), deadline)
        if len(family) > len(best_family):
            best_family = family
    
    # Try multiple random orderings
    rng = random.Random(42)
    while time.time() < deadline:
        order = list(range(n_sets))
        rng.shuffle(order)
        family = _greedy(order, deadline)
        if len(family) > len(best_family):
            best_family = family
            if len(best_family) >= 250:
                break
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
