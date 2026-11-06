"""Optimized sunflower-free family construction for k=4, n=16."""
import time
import random
from itertools import combinations

# Deadline guard
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _precompute_conflicts():
    """
    Precompute all 4-subsets of [1,16] as bitmasks and their pairwise
    sunflower-conflict relationships.
    
    Two 4-sets A, B conflict if there exists a third 4-set C (distinct from A, B)
    such that {A,B,C} forms a sunflower (all three pairwise intersections equal).
    
    For k=4, sunflower configurations:
    - Core size 2: A∩B = A∩C = B∩C = {x,y} (size 2), petals disjoint
    - Core size 1: A∩B = A∩C = B∩C = {x} (size 1), petals disjoint  
    - Core size 0: A∩B = A∩C = B∩C = ∅ (empty), all pairwise disjoint
    """
    n = 16
    all_quads = list(combinations(range(1, n+1), 4))
    quad_to_mask = {}
    mask_to_quad = {}
    
    for i, q in enumerate(all_quads):
        mask = 0
        for x in q:
            mask |= (1 << (x - 1))
        quad_to_mask[q] = mask
        mask_to_quad[mask] = q
    
    # For each pair of 4-sets, check if they can form a sunflower with some third set
    # Two sets A, B with intersection S can form a sunflower with C iff:
    #   C must contain S (the common core)
    #   C must be disjoint from A\S and B\S (the petals)
    #   C must be a 4-set
    #   C must be distinct from A and B
    
    # Precompute for each pair (i,j) whether they are in conflict
    num_quads = len(all_quads)  # 1820
    masks = [quad_to_mask[q] for q in all_quads]
    
    # For efficiency, precompute the "forbidden" masks for each pair
    # A pair (A,B) conflicts if there exists C such that:
    #   C ∩ A = C ∩ B = A ∩ B
    # Let S = A ∩ B. Then C must:
    #   - contain S
    #   - be disjoint from (A\S) ∪ (B\S)
    #   - have exactly 4 elements
    #   - C != A and C != B
    
    # For each pair, compute the set of elements C can use:
    #   C must pick 4 - |S| elements from [1,16] \ S \ (A\S) \ (B\S)
    #   = [1,16] \ (A ∪ B)
    #   plus the |S| elements of S
    #   So C = S ∪ T where T ⊆ [1,16]\(A∪B), |T| = 4-|S|
    #   And C ≠ A, C ≠ B
    
    # Two sets A,B are in "sunflower conflict" if there exists at least one valid C.
    # But actually for the independent set problem, we need:
    # A and B are incompatible if NO valid C exists? No...
    
    # Wait, let me re-read the constraint:
    # "No three distinct sets may have all three pairwise intersections equal"
    # So we need: for any three distinct sets A, B, C in our family, 
    # it's NOT the case that A∩B = A∩C = B∩C.
    
    # So when adding a new set C, we check against all pairs (A,B) already chosen:
    # does A∩B = A∩C = B∩C?
    
    # For a pair (A,B) with intersection S = A∩B:
    # C conflicts with this pair iff A∩C = S and B∩C = S
    # i.e., C∩A = S and C∩B = S
    # i.e., C contains S and C is disjoint from A\S and B\S
    # i.e., C ⊇ S and C ∩ (A∪B \ S) = ∅
    # i.e., C = S ∪ T where T ⊆ complement(A∪B), |T| = 4 - |S|
    
    # So for a given pair (A,B), the set of C's that would form a sunflower is:
    #   {S ∪ T : T ⊆ [1,16]\(A∪B), |T| = 4-|S|}
    #   minus {A, B} themselves (if they happen to be in that set, but they won't be
    #   since A and B both contain elements outside S)
    
    # For the greedy algorithm: when we have chosen set C, it conflicts with pair (A,B)
    # iff C is in the sunflower set of (A,B).
    
    # Alternative approach: for each set C, precompute which pairs (A,B) it would
    # complete a sunflower with. But that's 1820 * (1820*1819/2) which is too much.
    
    # Better: for the greedy, when considering adding C to family F,
    # check all pairs (A,B) in F: is C∩A = C∩B = A∩B?
    # This is O(|F|^2) per insertion, O(|F|^3) total. For |F| up to ~250, that's ~15M checks.
    
    # Even better: for each C, precompute a bitmask of "compatible" sets.
    # Two sets A and B are "incompatible" (cannot coexist in a sunflower-free family
    # with a third set) -- no wait, the constraint is on triples.
    
    # Let me think differently. The constraint is:
    # For all distinct A, B, C in family: NOT (A∩B = A∩C = B∩C)
    
    # This is a 3-uniform hypergraph independence problem.
    # Each "bad triple" is a hyperedge. We want a maximum independent set in this
    # 3-uniform hypergraph on 1820 vertices.
    
    # For a greedy approach:
    # 1. Start with empty family
    # 2. For each candidate set (in some order), check if adding it creates a bad triple
    #    with any two existing sets
    # 3. If not, add it
    
    # The check for candidate C against existing family F:
    # For each pair (A,B) in F: check if A∩B = A∩C = B∩C
    # This is O(|F|^2) per candidate, O(N * |F|^2) total where N=1820
    
    # For |F| ~ 250: 1820 * 250^2 / 2 ≈ 56M intersection checks
    # With bitmasks, each check is a few ops. Should be feasible in ~5-10 seconds.
    
    # Let's precompute intersections between all pairs of 4-sets as bitmasks
    # inter[i][j] = masks[i] & masks[j]
    # Then check: inter[i][j] == inter[i][k] == inter[j][k]
    
    # But storing 1820x1820 = 3.3M entries is fine.
    
    # Actually, let's just do it on the fly with bitmasks for simplicity.
    
    return all_quads, masks, quad_to_mask, mask_to_quad

def _check_sunflower(masks, idx_a, idx_b, idx_c):
    """Check if sets at indices a, b, c form a sunflower."""
    ma = masks[idx_a]
    mb = masks[idx_b]
    mc = masks[idx_c]
    ab = ma & mb
    ac = ma & mc
    bc = mb & mc
    return ab == ac == bc

def _greedy_construct(masks, order, deadline):
    """Greedy construction following the given order of candidate indices."""
    chosen = []
    chosen_set = set()
    
    for idx in order:
        if time.time() > deadline:
            break
        # Check against all pairs in chosen
        conflict = False
        for i in range(len(chosen)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(chosen)):
                if _check_sunflower(masks, chosen[i], chosen[j], idx):
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            chosen.append(idx)
            chosen_set.add(idx)
    
    return chosen

def _local_search(masks, chosen, deadline):
    """Try to improve the family by removing one set and adding others."""
    chosen_set = set(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each set and see if we can add more
        for i in range(len(chosen)):
            if time.time() > deadline:
                break
            # Remove chosen[i]
            trial = [x for x in chosen if x != chosen[i]]
            trial_set = set(trial)
            # Try to add sets not in trial
            added = 0
            for idx in range(len(masks)):
                if time.time() > deadline:
                    break
                if idx in trial_set:
                    continue
                conflict = False
                for a in range(len(trial)):
                    for b in range(a+1, len(trial)):
                        if _check_sunflower(masks, trial[a], trial[b], idx):
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict:
                    trial.append(idx)
                    trial_set.add(idx)
                    added += 1
            if len(trial) > len(chosen):
                chosen = trial
                chosen_set = trial_set
                improved = True
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    all_quads, masks, quad_to_mask, mask_to_quad = _precompute_conflicts()
    num_quads = len(all_quads)
    
    # Try multiple random orderings
    best_family = []
    
    # First, try a deterministic ordering (by size of "flexibility")
    # Sets that are "more constrained" (fewer potential sunflower partners) 
    # might be better to place first or last
    
    # Simple approach: try several random permutations
    rng = random.Random(42)
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        order = list(range(num_quads))
        rng.shuffle(order)
        
        family = _greedy_construct(masks, order, deadline - 2)
        
        if len(family) > len(best_family):
            best_family = family
    
    # Local search on the best family
    if time.time() < deadline - 5:
        best_family = _local_search(masks, best_family, deadline - 2)
    
    # Convert back to sorted tuples
    result = [sorted(all_quads[idx]) for idx in best_family]
    result.sort()
    
    return result
