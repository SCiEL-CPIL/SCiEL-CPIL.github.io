"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-element subsets of [1,16] as bitmasks
_ALL_SETS = []
for combo in combinations(range(16), 4):
    mask = 0
    for bit in combo:
        mask |= (1 << bit)
    _ALL_SETS.append(mask)

_N_SETS = len(_ALL_SETS)  # 1820

# Precompute pairwise intersection info for all pairs of sets
# For each pair (i, j), store the intersection bitmask
_INTERSECTIONS = {}
for i in range(_N_SETS):
    for j in range(i + 1, _N_SETS):
        inter = _ALL_SETS[i] & _ALL_SETS[j]
        _INTERSECTIONS[(i, j)] = inter

# For sunflower check: three sets A, B, C form a sunflower if
# A ∩ B == A ∩ C == B ∩ C
# This is equivalent to: for all elements, the element appears in either 0 or 2 or 3 of the sets
# (appears in exactly 1 is NOT allowed for a sunflower; appears in 2 means it's in the "petal overlap" but not core)
# Wait, let me reconsider. A sunflower with core C and petals: each set = C ∪ P_i where P_i are disjoint petals.
# A ∩ B = C (since P_A ∩ P_B = empty)
# A ∩ C = C
# B ∩ C = C
# So all three pairwise intersections equal C (the core), which can be empty.
#
# So the condition is: A ∩ B == A ∩ C == B ∩ C
# Let's call this common intersection the "core".
# For three sets to NOT form a sunflower, at least one pairwise intersection must differ from the others.

def _is_sunflower_triple(i, j, k):
    """Check if sets i, j, k form a sunflower (all three pairwise intersections equal)."""
    ab = _INTERSECTIONS[(i, j)]
    ac = _INTERSECTIONS[(i, k)]
    bc = _INTERSECTIONS[(j, k)]
    return ab == ac == bc

def _greedy_with_order(order):
    """Greedy construction: add sets in the given order if no sunflower is formed."""
    chosen = []
    chosen_set = set()
    for idx in order:
        if idx in chosen_set:
            continue
        # Check against all pairs in chosen
        ok = True
        for a, b in combinations(chosen, 2):
            if _is_sunflower_triple(a, b, idx):
                ok = False
                break
        if ok:
            chosen.append(idx)
            chosen_set.add(idx)
    return chosen

def _local_search(chosen, max_iters=5000):
    """Local search: try to swap out a set for a better one, or add/remove to improve size."""
    chosen_set = set(chosen)
    best = list(chosen)
    best_size = len(best)
    
    for iteration in range(max_iters):
        # Try adding a random set not in chosen
        if best_size < _N_SETS:
            candidates = [i for i in range(_N_SETS) if i not in chosen_set]
            if candidates:
                idx = random.choice(candidates)
                # Check if adding idx creates a sunflower
                ok = True
                for a, b in combinations(best, 2):
                    if _is_sunflower_triple(a, b, idx):
                        ok = False
                        break
                if ok:
                    best.append(idx)
                    chosen_set.add(idx)
                    best_size = len(best)
        
        # Try swapping: remove one, add one
        if best_size > 0:
            remove_idx = random.choice(best)
            best_without = [x for x in best if x != remove_idx]
            chosen_set_without = chosen_set - {remove_idx}
            
            # Try to add a different set
            candidates = [i for i in range(_N_SETS) if i not in chosen_set_without]
            if candidates:
                # Shuffle and try a few
                random.shuffle(candidates)
                for add_idx in candidates[:50]:
                    ok = True
                    for a, b in combinations(best_without, 2):
                        if _is_sunflower_triple(a, b, add_idx):
                            ok = False
                            break
                    if ok:
                        # Perform swap
                        best = best_without + [add_idx]
                        chosen_set = chosen_set_without | {add_idx}
                        best_size = len(best)
                        break
    
    return best

def construct():
    deadline = time.time() + 85  # 85 second deadline (5s safety margin)
    
    # Start with a greedy construction in natural order
    order = list(range(_N_SETS))
    result = _greedy_with_order(order)
    best_result = list(result)
    best_size = len(best_result)
    
    # Try multiple random orderings with greedy
    start_time = time.time()
    attempts = 0
    while time.time() < deadline - 30:  # Leave 30s for local search
        random.shuffle(order)
        result = _greedy_with_order(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = list(result)
        attempts += 1
        if attempts % 100 == 0:
            break  # Don't do too many greedy passes
    
    # Local search refinement
    while time.time() < deadline - 5:
        refined = _local_search(best_result, max_iters=2000)
        if len(refined) > best_size:
            best_size = len(refined)
            best_result = refined
    
    # Convert bitmasks back to sorted lists
    return [sorted([i + 1 for i in range(16) if (mask >> i) & 1]) for mask in best_result]

# EVOLVE-BLOCK-END
