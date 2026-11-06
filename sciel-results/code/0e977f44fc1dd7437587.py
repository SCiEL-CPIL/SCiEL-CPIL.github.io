import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_points = list(range(16))
    all_triples_list = list(itertools.combinations(all_points, 3))
    num_triples = len(all_triples_list)
    
    # Precompute masks
    triple_masks = []
    for t in all_triples_list:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # Precompute conflict bitsets
    conflict_bits = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[i]
        for j in range(i + 1, num_triples):
            mj = triple_masks[j]
            inter = mi & mj
            if inter & (inter - 1):  # share > 1 point
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    
    # Precompute for each point, which triples contain it
    point_triples = [[] for _ in range(16)]
    for i, t in enumerate(all_triples_list):
        for p in t:
            point_triples[p].append(i)
    
    def count_ones(mask):
        return bin(mask).count('1')
    
    def find_max_independent_set(active_mask):
        """
        Find the size of the largest subset of points that contains no active triple.
        Uses branch-and-bound on the hitting set formulation.
        alpha = 16 - (minimum hitting set size)
        """
        if active_mask == 0:
            return 16
        
        # Get list of active triple indices
        active_indices = [i for i in range(num_triples) if (active_mask >> i) & 1]
        if not active_indices:
            return 16
        
        active_masks = [triple_masks[i] for i in active_indices]
        num_active = len(active_masks)
        
        # Minimum hitting set: find min number of points that hit every active triple.
        # Branch and bound: pick a triple, try each point in it.
        # State: remaining triples to hit (as a set of indices into active_masks),
        #        points used so far.
        
        # We'll use a recursive approach with memoization on the frozenset of remaining triple indices.
        # But that might be too large. Instead, use a greedy + local search for hitting set.
        
        # Actually, for 16 points, we can try all subsets of points of size k from 1 to 16,
        # and check if it's a hitting set. The first k that works gives alpha = 16 - k.
        
        # But we want the MINIMUM hitting set, so we try k from 1 upward.
        
        # For efficiency, we can use the fact that if a point is in no active triple,
        # it doesn't need to be in the hitting set.
        
        # Let's use a smarter approach: branch and bound.
        # Pick the active triple with the fewest points not yet "covered" by chosen points.
        # Actually, all triples have exactly 3 points.
        
        # Let's try: for k from 1 to 16, check if there's a hitting set of size k.
        # For k=1: check if there's a point in all active triples.
        # For k=2: check if there's a pair {p,q} such that every active triple contains p or q.
        # etc.
        
        # For small k, this is feasible.
        
        # Let's precompute: for each point p, the set of active triple indices that contain p.
        point_to_triples = [set() for _ in range(16)]
        for idx, tm in enumerate(active_masks):
            for p in range(16):
                if (tm >> p) & 1:
                    point_to_triples[p].add(idx)
        
        # Now, a hitting set S hits all active triples iff for every active triple idx,
        # there exists p in S such that idx is in point_to_triples[p].
        
        # Equivalently, the union of point_to_triples[p] for p in S must be all of {0..num_active-1}.
        
        # Let's represent the set of active triple indices as a bitmask (if num_active <= 64).
        # But num_active can be up to 560, so we can't use a single bitmask.
        
        # Alternative: for each subset S of points of size k, check if it's a hitting set.
        # For k=1: 16 subsets.
        # For k=2: 120 subsets.
        # For k=3: 560 subsets.
        # For k=4: 1820 subsets.
        # For k=5: 4368 subsets.
        # For k=6: 8008 subsets.
        
        # For each subset, check if it's a hitting set:
        #   For each active triple, check if it contains any point in S.
        #   This is O(num_active * k) per subset.
        
        # Total for k=1..6: (16 + 120 + 560 + 1820 + 4368 + 8008) * 560 * 6 ≈ 42 million ops.
        # This might be too slow in Python.
        
        # Let's optimize: precompute for each point p, the bitmask of active triples it hits.
        # But we can't use a bitmask if num_active > 64.
        
        # Alternative: use a different approach.
        # Let's use the fact that we're looking for the minimum hitting set.
        # We can use a greedy algorithm to get an upper bound, then branch and bound.
        
        # Greedy hitting set:
        #   While there are unhit triples:
        #     Pick the point that hits the most unhit triples.
        #     Add it to the hitting set.
        #     Mark all triples containing it as hit.
        
        # This gives an upper bound on the hitting set size.
        
        # Then, branch and bound to find the minimum.
        
        # For now, let's use a simpler approach: iterate k from 1 to 16, and for each k,
        # check all subsets of size k. Stop at the first k that works.
        
        # To speed up the check, we can precompute for each point p, the set of active triples it hits.
        # Then, for a subset S, the set of hit triples is the union of point_to_triples[p] for p in S.
        # We need this union to be all of {0..num_active-1}.
        
        # Let's represent the set of active triple indices as a frozenset.
        all_triple_indices = frozenset(range(num_active))
        
        for k in range(1, 17):
            for subset in itertools.combinations(all_points, k):
                # Check if subset is a hitting set
                hit_set = set()
                for p in subset:
                    hit_set.update(point_to_triples[p])
                if hit_set == all_triple_indices:
                    return 16 - k
            if time.time() > deadline - 5:
                return 16 - k  # Conservative estimate
        return 0
    
    def generate_initial():
        """Generate a random linear triple system using greedy method."""
        active_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if (conflict_bits[idx] & active_mask) == 0:
                active_mask |= (1 << idx)
        return active_mask
    
    def local_search(mask, max_iterations=100):
        """Local search to minimize independence number."""
        current_alpha = find_max_independent_set(mask)
        current_mask = mask
        
        for it in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Pick a random inactive triple, add it, remove conflicts, greedy refill.
            inactive_indices = [i for i in range(num_triples) if not ((current_mask >> i) & 1)]
            if not inactive_indices:
                break
            
            target_idx = random.choice(inactive_indices)
            
            # Remove conflicts
            conflicts = conflict_bits[target_idx] & current_mask
            new_mask = current_mask & ~conflicts
            new_mask |= (1 << target_idx)
            
            # Greedy refill
            refill_indices = list(range(num_triples))
            random.shuffle(refill_indices)
            added = 0
            for idx in refill_indices:
                if (new_mask >> idx) & 1:
                    continue
                if (conflict_bits[idx] & new_mask) == 0:
                    new_mask |= (1 << idx)
                    added += 1
                    if added > 50:
                        break
            
            new_alpha = find_max_independent_set(new_mask)
            
            if new_alpha <= current_alpha:
                current_alpha = new_alpha
                current_mask = new_mask
        
        return current_mask, current_alpha
    
    best_alpha = 16
    best_mask = 0
    
    # Multiple restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate initial solution
        current_mask = generate_initial()
        current_mask, current_alpha = local_search(current_mask, max_iterations=50)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_mask = current_mask
        
        # If we've reached a very good solution, do more polishing
        if best_alpha <= 6 and time.time() < deadline - 20:
            best_mask, best_alpha = local_search(best_mask, max_iterations=100)
    
    # Final polish
    if time.time() < deadline - 5:
        best_mask, best_alpha = local_search(best_mask, max_iterations=50)
    
    # Convert mask to list of triples
    result = []
    for i in range(num_triples):
        if (best_mask >> i) & 1:
            result.append(list(all_triples_list[i]))
    
    return result
