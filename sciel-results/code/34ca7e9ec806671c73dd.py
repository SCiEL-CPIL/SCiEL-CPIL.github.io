import random
import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Points 0..15
    points = list(range(16))
    
    # Precompute all triples as bitmasks for fast operations
    all_triples = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                mask = (1 << i) | (1 << j) | (1 << k)
                all_triples.append((i, j, k, mask))
    
    # Helper to check if a triple mask can be added to a set of triple masks
    # Two triples conflict if they share more than 1 point, i.e., their masks share >= 2 bits
    def can_add_mask(triple_masks, new_mask):
        for tm in triple_masks:
            shared = (tm & new_mask).bit_count()
            if shared > 1:
                return False
        return True

    # Fast alpha computation: find the largest independent set
    # A set S is independent if no triple is fully contained in S
    # We use a branch-and-bound / greedy approach
    def calc_alpha_fast(triple_masks):
        if not triple_masks:
            return 16
        
        # For each subset of points, check if it's independent
        # Use the fact that we can iterate by size from large to small
        # and stop at the first independent set of that size
        
        # Precompute: for each point, which triples contain it
        # Actually, let's use a different approach:
        # A set S of size k is independent iff for every triple t, t is NOT a subset of S
        # Equivalently: for every triple t, at least one point of t is NOT in S
        
        # We'll iterate masks by descending popcount
        # Start from size 16 down to 0
        
        # Optimization: if we have many triples, small independent sets are likely
        # Let's try sizes from high to low, but with a cap
        
        # Actually, let's use a smarter approach:
        # The independence number is the max k such that there exists a k-subset with no triple contained in it.
        # We can check: for a given k, does there exist a k-subset that is independent?
        
        # For speed, let's use the following:
        # Build a "forbidden" structure: a set S is forbidden if it contains any triple.
        # We want the largest S that is not forbidden.
        
        # Use bit operations:
        # For each triple mask t, any superset of t is forbidden.
        # S is independent iff for all t in triple_masks, (S & t) != t, i.e., S does not contain all of t.
        
        # Let's iterate from size 16 down
        # But 2^16 = 65536 masks, and for each we check ~n triples.
        # With n up to ~80, that's ~5M operations per calc_alpha call.
        # We need this to be fast.
        
        # Better: precompute for each mask whether it's independent
        # But that depends on triple_masks which changes.
        
        # Alternative: use the complement. A set S is independent iff its complement C
        # hits every triple. So alpha = 16 - (minimum hitting set size).
        # Minimum hitting set is NP-hard but for 16 points and ~80 triples, we can use
        # a simple approach.
        
        # Let's just do the direct check but optimized:
        # Sort triples by some criterion? 
        # Or: for each mask, check if any triple is subset.
        
        # Key optimization: if a mask fails the check for one triple, skip it.
        # Also, we can precompute triple masks as a list.
        
        tm_list = list(triple_masks)
        n_tm = len(tm_list)
        
        # Iterate from size 16 down to 0
        # Generate masks by popcount using combinatorial generation is complex.
        # Instead, iterate all 65536 masks but group by popcount.
        
        # Actually, let's just iterate all masks and track max independent size.
        # 65536 * n_tm operations. With n_tm ~ 80, that's ~5.2M ops.
        # In Python, this might take ~0.5-1s. Too slow if called many times.
        
        # We need a faster alpha computation.
        
        # Idea: use a greedy upper bound first, then verify.
        # Greedy: start with all 16 points, remove points that are in many triples?
        # Actually, a simple upper bound: if we have n triples, each covering 3 points,
        # the independence number is at most 16 - something.
        
        # Let's use a different strategy: 
        # For local search, we don't need exact alpha every time.
        # We can use a fast heuristic estimate and only compute exact alpha when promising.
        
        # For now, let's do exact but optimized:
        max_size = 0
        # Iterate masks in order of decreasing popcount
        # Precompute popcount for all masks? 
        # Or just iterate and check.
        
        # Let's iterate all masks:
        for mask in range(1 << 16):
            # Check if mask is independent: no triple is subset of mask
            independent = True
            for tm in tm_list:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                size = mask.bit_count()
                if size > max_size:
                    max_size = size
                    # Early termination: if we've found size 16, that's max
                    # But we want the max, so we can't terminate early unless we reach 16
                    # and we know we can't do better. Actually 16 is the max possible.
                    if size == 16:
                        return 16
        
        return max_size

    # Even faster: use the hitting set dual
    # alpha = 16 - tau, where tau is the minimum size of a set that hits every triple
    # We can compute tau by trying all subsets of size 1, 2, 3, ...
    # A set H hits every triple iff for every triple t, H intersects t.
    # Equivalently, the complement of H is independent.
    # So tau = 16 - alpha.
    
    # To compute tau: find the smallest H such that every triple has at least one point in H.
    # Try |H| = 1, 2, 3, ...
    # For each size k, check if there exists a k-subset that hits all triples.
    
    def calc_alpha_via_hitting(triple_masks):
        if not triple_masks:
            return 16
        
        tm_list = list(triple_masks)
        
        # Try hitting set sizes from 0 upward
        # Size 0: only works if no triples
        # Size 1: check if there's a point in every triple
        # Size 2: check if there's a pair that hits every triple
        # etc.
        
        # For size k, we need to check all C(16,k) subsets
        # k=0: 1 subset
        # k=1: 16 subsets
        # k=2: 120 subsets
        # k=3: 560 subsets
        # k=4: 1820 subsets
        # k=5: 4368 subsets
        # k=6: 8008 subsets
        # k=7: 11440 subsets
        # k=8: 12870 subsets
        
        # For each candidate hitting set H, check if it hits all triples.
        # H hits triple t iff H & t != 0.
        
        for k in range(0, 17):
            # Generate all k-subsets of {0,...,15}
            # Check if any hits all triples
            for H in itertools.combinations(range(16), k):
                H_mask = 0
                for p in H:
                    H_mask |= (1 << p)
                
                hits_all = True
                for tm in tm_list:
                    if (H_mask & tm) == 0:
                        hits_all = False
                        break
                if hits_all:
                    return 16 - k  # alpha = 16 - tau
        
        return 0  # Should not reach here

    # Use the hitting set approach as it's often faster for small tau
    def calc_alpha(triple_masks):
        if not triple_masks:
            return 16
        # Try hitting set approach first (fast when tau is small)
        return calc_alpha_via_hitting(triple_masks)

    # Helper to get triple masks from a list of triples
    def get_triple_masks(triples):
        return [t[3] for t in triples]

    # Initial construction: Greedy packing with bitmasks
    def random_initial():
        triple_masks = []
        available = all_triples[:]
        random.shuffle(available)
        for t in available:
            if can_add_mask(triple_masks, t[3]):
                triple_masks.append(t[3])
        return triple_masks

    # Local search: try swaps (remove one triple, add another) and additions
    def local_search(triple_masks):
        current_alpha = calc_alpha(triple_masks)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline:
            improved = False
            no_improve_count += 1
            
            # Try adding a triple
            candidates = [t for t in all_triples if can_add_mask(triple_masks, t[3])]
            if candidates:
                random.shuffle(candidates)
                # Try a limited number of candidates
                for t in candidates[:20]:
                    new_masks = triple_masks + [t[3]]
                    a = calc_alpha(new_masks)
                    if a < current_alpha:
                        triple_masks = new_masks
                        current_alpha = a
                        improved = True
                        no_improve_count = 0
                        break
            
            # Try swap: remove one, add one
            if not improved and len(triple_masks) > 0:
                # Try removing a random triple and adding a different one
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(triple_masks) - 1)
                    removed = triple_masks[idx]
                    reduced = triple_masks[:idx] + triple_masks[idx+1:]
                    
                    # Find a triple that can be added to reduced
                    candidates = [t for t in all_triples if can_add_mask(reduced, t[3]) and t[3] != removed]
                    if candidates:
                        random.shuffle(candidates)
                        for t in candidates[:10]:
                            new_masks = reduced + [t[3]]
                            a = calc_alpha(new_masks)
                            if a < current_alpha:
                                triple_masks = new_masks
                                current_alpha = a
                                improved = True
                                no_improve_count = 0
                                break
                    if improved:
                        break
            
            # If no improvement for several iterations, do a random restart
            if no_improve_count > 20:
                break
        
        return triple_masks, current_alpha

    # Main loop
    best_alpha = 16
    best_triple_masks = []
    
    # First, try a deterministic good starting point
    # A near-Steiner triple system on 16 points
    # STS(15) exists (Steiner system S(2,3,15)), plus one point
    # Let's use a known construction or just start with random
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Random restart
        init_masks = random_initial()
        improved_masks, alpha = local_search(init_masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triple_masks = improved_masks
            if best_alpha <= 5:
                break
    
    # Convert best_triple_masks back to list of triples
    if not best_triple_masks:
        # Fallback: return a simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Convert masks to triples
    result = []
    for mask in best_triple_masks:
        pts = []
        for p in range(16):
            if mask & (1 << p):
                pts.append(p)
        result.append(pts)
    
    return result

# EVOLVE-BLOCK-END
