"""Randomized Greedy with Multiple Restarts."""
# EVOLVE-BLOCK-START
import itertools
import random

def _pairwise_intersections_equal(a, b, c):
    """Check if three distinct 4-sets have all three pairwise intersections equal."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _is_sunflower_free(chosen, new_set):
    """Check if adding new_set to chosen creates a sunflower."""
    # Optimization: Only check against pairs in chosen
    for i in range(len(chosen)):
        a = chosen[i]
        for j in range(i + 1, len(chosen)):
            b = chosen[j]
            if _pairwise_intersections_equal(a, b, new_set):
                return False
    return True

def _randomized_greedy(random_seed):
    """Greedy with random ordering."""
    rng = random.Random(random_seed)
    all_4sets = [set(s) for s in itertools.combinations(range(1, 17), 4)]
    rng.shuffle(all_4sets)
    
    chosen = []
    for s in all_4sets:
        if _is_sunflower_free(chosen, s):
            chosen.append(s)
            
    return [sorted(s) for s in chosen]

def construct():
    """Construct a large sunflower-free family using randomized greedy."""
    best_family = []
    best_size = 0
    
    # Run multiple randomized greedy attempts
    # 60s limit. Each greedy run takes roughly 0.5-2s depending on family size.
    # Let's try 50-100 runs.
    import time
    start_time = time.time()
    max_time = 55.0 # Leave 5s margin
    
    seed = 0
    while time.time() - start_time < max_time:
        family = _randomized_greedy(seed)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
        seed += 1
        
    return best_family
# EVOLVE-BLOCK-END
