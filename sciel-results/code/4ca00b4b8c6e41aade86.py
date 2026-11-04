import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # 150s kill limit. Use 145s as deadline to be safe.
    deadline = start_time + 145.0

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    def block_triplet_ids(block):
        return set(get_id(triplet) for triplet in itertools.combinations(block, t))

    def greedy_construct(v, k, t, seed=0, bias=0.0):
        """
        Greedy construction with weighted scoring.
        """
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        # Weights for points to break symmetry
        point_weights = np.ones(v)
        
        while uncovered:
            # Pick a random uncovered triplet to start
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                # Score = gain / product(weights of points in block + candidate)
                # This encourages using points that are "underused" in the current partial block
                # or globally? Let's stick to marginal gain primarily, with a small random perturbation.
                
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                best_point = -1
                best_score = -1
                second_best_point = -1
                second_best_score = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    
                    # Add a small random component to break ties and explore
                    score = gain + rng.random() * bias * 1.0
                    
                    if score > best_score:
                        second_best_score = best_score
                        second_best_point = best_point
                        best_score = score
                        best_point = p
                    elif score > second_best_score:
                        second_best_score = score
                        second_best_point = p

                if best_point == -1:
                    # No candidates? Should not happen if k <= v
                    break
                    
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks

    # 1. Initial Construction
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Pure Greedy with different seeds
    for seed in range(30):
        if time.time() > deadline - 60:
            break
        blocks = greedy_construct(v, k, t, seed, bias=0.0)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # Phase 2: Randomized Greedy
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        bias = 0.1 + 0.05 * (seed % 5)
        blocks = greedy_construct(v, k, t, seed, bias=bias)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(v, k, t, 0, bias=0.0)

    # 2. Local Search Optimization
    current_blocks = list(best_blocks)
    
    # Precompute triplet IDs for each block
    def get_block_ids(blocks):
        return [block_triplet_ids(b) for b in blocks]

    # We maintain counts of how many times each triplet is covered
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                count[tid] += delta

    update_counts(current_blocks, 1)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def can_cover_critical(critical_ids, num_blocks):
        """
        Check if the set of critical triplet IDs can be covered by num_blocks blocks of size k.
        This is a set cover problem on a small set.
        """
        if not critical_ids:
            return True, []
        
        # If num_blocks == 1
        if num_blocks == 1:
            # Check if all critical triplets are subsets of some block of size k
            # We can try to construct a block that covers all of them.
            # The union of points in critical triplets must be <= k.
            points_needed = set()
            for tid in critical_ids:
                points_needed.update(all_triplets[tid])
            if len(points_needed) <= k:
                return True, [list(points_needed)]
            return False, []
            
        # If num_blocks == 2
        if num_blocks == 2:
            # Try to partition critical_ids into two sets, each coverable by one block
            # This is exponential, but critical_ids should be small.
            # Heuristic: Greedy partition
            # First, find a block that covers as many as possible
            # Then check if the rest can be covered by one block
            
            # Get all points involved
            all_points = set()
            for tid in critical_ids:
                all_points.update(all_triplets[tid])
            
            if len(all_points) > 2 * k:
                return False, []
                
            # Try all subsets of points for the first block? Too expensive if all_points is large.
            # Instead, iterate through a few candidate blocks.
            # Candidate 1: Maximal block covering critical triplets.
            # Let's just try to find two blocks B1, B2 such that B1 U B2 covers all critical triplets.
            # We can iterate through all possible B1 from the existing blocks? No, we need new blocks.
            # We can iterate through all k-subsets of all_points? Too many.
            
            # Better approach:
            # Pick a triplet from critical_ids. It must be in B1 or B2.
            # Assume it's in B1.
            # We can try to construct B1 to cover as many critical triplets as possible.
            # Then check if the remainder can be covered by B2.
            
            # Simple heuristic:
            # Try to cover with one block (B1). Let R be the remaining critical triplets.
            # If R can be covered by one block (B2), success.
            
            # Let's try to find B1 by picking a random critical triplet and expanding.
            # Since this is called frequently, we need it fast.
            # If len(critical_ids) is small (< 20), we can be a bit more thorough.
            
            # For now, a simple check:
            # Can the union of points be split into two sets of size <= k such that each set covers its assigned triplets?
            # This is hard.
            
            # Alternative: Return True if we find ANY pair.
            # We'll use a limited search.
            # Pick a point p from all_points.
            # Try to form B1 containing p.
            # Actually, let's just return False for now and rely on 2-for-1.
            # 3-for-2 is harder. I will implement a simplified version.
            
            # Simplified 3-for-2:
            # Only try if the union of points in critical_ids is small (<= 12).
            if len(all_points) > 12:
                return False, []
                
            # Iterate through all k-subsets of all_points for B1?
            # C(12, 7) = 792. That's manageable.
            # For each B1, check if the remaining critical triplets can be covered by one block B2.
            
            # Precompute which critical triplets are covered by a candidate block
            # This is still slow if done naively.
            
            # Let's just return False for 3-for-2 in this iteration to keep it simple and fast.
            # The 2-for-1 is the main driver for reduction.
            return False, []
            
        return False, []

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = block_triplet_ids(b1)
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = block_triplet_ids(b2)
                
                critical_points = set()
                
                # Check b1's triplets
                for tid in ids_b1:
                    c = count[tid]
                    covered_by_b1 = 1
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])
                        
                # Check b2's triplets (only those not in b1)
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    c = count[tid]
                    if c == 1:
                        critical_points.update(all_triplets[tid])
                
                if not critical_points:
                    # Both blocks are redundant together
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                if len(critical_points) <= k:
                    # We can form a block containing all critical points
                    new_block = list(critical_points)
                    # Fill with random points
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    # Verify validity
                    # The condition for validity is that ALL previously covered triplets remain covered.
                    # We checked that the "critical" ones (those with count dropping to 0) are in new_block.
                    # Triplets with count > critical threshold remain covered by other blocks.
                    # So this swap is valid.
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    # Main Optimization Loop
    progress_made = True
    while progress_made and time.time() < deadline - 2:
        progress_made = False
        
        # 1. Remove redundants
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True
            
        # 2. Swap 2 for 1
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True

    # Final pass to ensure no redundants remain
    current_blocks, _ = try_remove_redundant(current_blocks)

    return current_blocks

if __name__ == "__main__":
    pass
