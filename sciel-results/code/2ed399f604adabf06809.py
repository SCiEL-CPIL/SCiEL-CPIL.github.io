"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower (all pairwise intersections equal)."""
    return (a & b) == (a & c) == (b & c)

def _greedy_construct_fast(order, deadline):
    """
    Fast greedy construction using a 'blocked' set.
    A set C is blocked if it forms a sunflower with any pair (A, B) in the current family.
    When adding S, we only check pairs (S, A) for A in family to update blocked sets.
    """
    family = []
    blocked = set()
    
    for s in order:
        if time.time() > deadline:
            break
        
        fs = frozenset(s)
        if fs in blocked:
            continue
            
        # Check if adding s creates a sunflower with any existing pair (s, a)
        # If it does, those sunflower sets become newly blocked.
        # We don't need to check if s itself is blocked by existing pairs because
        # we already checked `fs in blocked` above.
        # Wait, the logic is:
        # 1. If s is blocked by any pair in family, skip. (Checked by `fs in blocked`)
        # 2. Add s to family.
        # 3. For each a in family (before adding s), check if (s, a, c) is a sunflower.
        #    If yes, c is now blocked.
        
        family.append(s)
        
        for a in family[:-1]: # Check pairs (s, a)
            # The core is s & a
            # Any set c such that s & c == core and a & c == core is a sunflower petal.
            # We need to mark these c as blocked.
            core = s & a
            # Instead of iterating all c, we can just check specific candidates?
            # No, we must update the blocked set.
            # Efficient way: 
            # A set c forms a sunflower with s and a iff (s & c) == (a & c) == (s & a).
            # This is equivalent to:
            # 1. c contains core
            # 2. c is disjoint from (s \ core)
            # 3. c is disjoint from (a \ core)
            
            # However, iterating all sets to update blocked is O(N_sets).
            # We can precompute for each pair (s, a) the list of sets that form a sunflower with them?
            # That is too much memory/precomputation.
            
            # Let's stick to the previous logic but optimized:
            # When adding s, we check all existing a in family.
            # If (s, a, c) is a sunflower, c is blocked.
            # We can find c by checking all sets in `all_sets`? No, that's O(N_sets) per pair.
            # Total O(N_family^2 * N_sets) which is too slow if N_family is large.
            
            # Alternative:
            # We only need to know if a *candidate* is blocked.
            # The `blocked` set approach works if we can efficiently add to it.
            # For each pair (s, a), the sets c that form a sunflower are those c such that:
            # c contains (s & a) and c is disjoint from (s ^ a) \ (s & a) ? No.
            # s & c = s & a => c contains s&a and c is disjoint from s\(s&a).
            # a & c = s & a => c contains s&a and c is disjoint from a\(s&a).
            # So c must contain core and be disjoint from (s \ core) U (a \ core).
            
            # Let's just iterate through all sets to update blocked?
            # There are 1820 sets.
            # If family size is 200, we do 200 * 1820 = 364,000 checks per addition.
            # 200 additions * 364,000 = 72,800,000 checks.
            # This is feasible in Python if optimized (using bitmasks or sets).
            
            # Let's use bitmasks for speed.
            # Precompute bitmasks for all sets.
            pass # Placeholder for logic below
    return family

def _greedy_construct_bitmask(all_sets_masks, order_indices, deadline):
    """
    Greedy construction using bitmasks for fast intersection checks.
    order_indices: list of indices into all_sets_masks.
    """
    n_sets = len(all_sets_masks)
    # Precompute sunflower partners for each pair? No, too big.
    # We maintain a blocked bitmask (or set of indices).
    blocked = set()
    family_indices = []
    
    # To speed up "find all c that form sunflower with s and a",
    # we can precompute for each pair of sets (i, j) the list of k such that (i, j, k) is a sunflower.
    # This is O(N^3) precomputation which is 1820^3 ~ 6e9, too slow.
    
    # Instead, we compute on the fly but efficiently.
    # For a pair (s, a), the core is s & a.
    # The forbidden part is (s | a) \ core.
    # A set c is a sunflower petal if:
    # 1. c contains core
    # 2. c is disjoint from forbidden
    
    # Let's precompute for each set its bitmask.
    # We will iterate through all sets to update blocked.
    
    for idx in order_indices:
        if time.time() > deadline:
            break
            
        if idx in blocked:
            continue
            
        s_mask = all_sets_masks[idx]
        
        # Check if s conflicts with any existing pair in family.
        # This is implicitly handled by `blocked`.
        # If s forms a sunflower with any pair (a, b) in family, s would have been added to blocked when a or b was added.
        # Wait, when we added a, we checked pairs (a, existing). If s was one of those, s would be blocked.
        # So if idx is not in blocked, it is safe to add.
        
        family_indices.append(idx)
        
        # Update blocked set:
        # For each a in family (before adding s), check if (s, a, c) is a sunflower for any c.
        # If so, add c to blocked.
        s_mask = all_sets_masks[idx]
        for a_idx in family_indices[:-1]:
            a_mask = all_sets_masks[a_idx]
            core_mask = s_mask & a_mask
            forbidden_mask = (s_mask | a_mask) & ~core_mask
            
            # Find all c such that (c & core_mask) == core_mask and (c & forbidden_mask) == 0
            # We can iterate through all sets and check this.
            # Optimization: Precompute for each mask M, the list of sets that contain M?
            # Or just iterate. 1820 sets is small.
            
            for c_idx in range(n_sets):
                if c_idx == idx or c_idx == a_idx:
                    continue
                c_mask = all_sets_masks[c_idx]
                if (c_mask & core_mask) == core_mask and (c_mask & forbidden_mask) == 0:
                    blocked.add(c_idx)
                    
    return family_indices

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Precompute all 4-element subsets and their bitmasks
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Map set to index for quick lookup?
    # We work with indices in the greedy loop.
    
    # Bitmasks
    masks = []
    for s in all_sets:
        m = 0
        for x in s:
            m |= (1 << x)
        masks.append(m)
        
    best_family_indices = []
    best_size = 0
    
    # Strategy 1: Greedy with heuristic ordering
    # Heuristic: Sets with larger "forbidden" regions (when paired with others) are more restrictive?
    # Or sets that are "dense" in intersection?
    # Let's try: Sort by number of sets that intersect in exactly 2 elements?
    # Or just random orderings with the fast greedy.
    
    rng = random.Random(42)
    
    # Run multiple random greedy attempts
    # The fast greedy is O(N_family * N_sets * N_family) roughly?
    # Inner loop: for each a in family, iterate all c.
    # Cost per addition: len(family) * N_sets.
    # Total cost: sum(len(family)) * N_sets ~ (N_family^2 / 2) * N_sets.
    # If N_family = 200, cost ~ 20000 * 1820 = 36,400,000 operations.
    # This is fast enough to run many times.
    
    # Let's run 100 random attempts
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
            
        order = list(range(n_sets))
        rng.shuffle(order)
        
        family_indices = _greedy_construct_bitmask(masks, order, deadline - 5)
        
        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family_indices = family_indices
            if best_size >= 250:
                break
                
    # Convert indices back to sets
    if best_family_indices:
        return [sorted(all_sets[i]) for i in best_family_indices]
    else:
        return []
# EVOLVE-BLOCK-END
