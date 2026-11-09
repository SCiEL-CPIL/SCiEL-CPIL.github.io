"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import random

def _greedy_build(all_sets, n_sets, order):
    """Greedy: iterate in given order, add set if no sunflower with any existing pair."""
    chosen = []
    chosen_frozensets = []
    # For fast lookup: for each chosen set, precompute nothing yet
    # We'll check conflicts on the fly
    
    for idx in order:
        s = all_sets[idx]
        conflict_found = False
        for i in range(len(chosen)):
            if conflict_found:
                break
            a = chosen_frozensets[i]
            as_ = a & s
            for j in range(i + 1, len(chosen)):
                b = chosen_frozensets[j]
                ab = a & b
                bs = b & s
                if ab == as_ == bs:
                    conflict_found = True
                    break
        if not conflict_found:
            chosen.append(idx)
            chosen_frozensets.append(s)
    
    return chosen

def construct():
    all_sets = [frozenset(c) for c in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    best_size = 0
    best_chosen = []
    
    rng = random.Random(42)
    
    # Multiple restarts with different random orderings
    for trial in range(200):
        order = list(range(n_sets))
        rng.shuffle(order)
        
        chosen = _greedy_build(all_sets, n_sets, order)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen[:]
    
    return [sorted(all_sets[i]) for i in best_chosen]
# EVOLVE-BLOCK-END
