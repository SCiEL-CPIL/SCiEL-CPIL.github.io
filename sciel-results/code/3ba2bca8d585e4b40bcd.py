import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Precomputation
    # Map triples to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    full_mask = (1 << num_triples) - 1
    
    # Precompute for each point pair (a, b), the list of triples (a, b, c)
    # Actually, we just need to check specific triples on the fly using idx_map.
    
    def greedy_construct(seed):
        random.seed(seed)
        uncovered_mask = full_mask
        blocks = []
        
        while uncovered_mask:
            block = []
            # First two points: arbitrary (or random) to start
            # To ensure diversity, pick random available points
            available = list(range(v))
            random.shuffle(available)
            block.append(available[0])
            block.append(available[1])
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                # Calculate gain for each candidate
                # Candidates are points not in block
                candidates = [x for x in range(v) if x not in block]
                # Optimization: If random tie-breaking, we can just pick one if gains are equal
                # But we need to find max gain first.
                
                # To speed up, we can pre-compute pairs in block
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    for pair in block_pairs:
                        # Create triple
                        # pair is (a, b) if t=3. x is x.
                        # Need sorted triple for lookup
                        t_tuple = tuple(sorted(pair + (x,)))
                        idx = triple_to_idx[t_tuple]
                        if (uncovered_mask >> idx) & 1:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain:
                        # Random tie-breaking
                        if random.random() < 0.5:
                            best_x = x
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                uncovered_mask &= ~(1 << idx)
                
        return blocks

    def local_search(blocks):
        # Convert to set for fast lookup? No, list is fine for small N.
        # Compute coverage counts
        def compute_coverage(blocks):
            counts = [0] * num_triples
            for b in blocks:
                for triple in itertools.combinations(b, t):
                    counts[triple_to_idx[triple]] += 1
            return counts

        def is_removable(b, counts, blocks):
            # Check if all triples in b have count > 1 (or >= 2 if b is in blocks)
            # Actually, if we remove b, count becomes count-1.
            # So we need count >= 2 for all triples in b.
            for triple in itertools.combinations(b, t):
                if counts[triple_to_idx[triple]] < 2:
                    return False
            return True

        def try_swap(b, counts, blocks):
            # Try to swap a point in b with a point outside b
            # such that coverage is maintained for all triples in b and new triples
            # This is complex. Simplified:
            # Just try random swaps and check validity.
            
            # Identify critical triples in b (count == 1)
            critical = []
            for triple in itertools.combinations(b, t):
                if counts[triple_to_idx[triple]] == 1:
                    critical.append(triple)
            
            if not critical:
                return None # Should have been removed
            
            # To keep b "valid" after swap, the new block b' must cover all critical triples.
            # This implies that for any critical triple (a,b,c), if we swap out 'a', we must ensure
            # that 'a' is not in the critical triple OR the new point 'q' allows covering it?
            # No, if (a,b,c) is critical and we swap a->q, the triple (a,b,c) is lost.
            # So we can only swap points that are NOT in any critical triple?
            # Or, if we swap a point, we must ensure the critical triples are still covered by OTHER blocks?
            # But they are critical (count 1), so they are ONLY covered by b.
            # Therefore, we CANNOT lose any critical triple.
            # So, we can only swap points that are not part of any critical triple.
            
            swappable_points = []
            for p in b:
                in_critical = False
                for c_triple in critical:
                    if p in c_triple:
                        in_critical = True
                        break
                if not in_critical:
                    swappable_points.append(p)
            
            if not swappable_points:
                return None
            
            # Pick a random swappable point
            p = random.choice(swappable_points)
            
            # Pick a random outside point q
            outside = [x for x in range(v) if x not in b]
            if not outside:
                return None
            q = random.choice(outside)
            
            # Form new block b'
            b_new = sorted([x for x in b if x != p] + [q])
            
            # Check if b_new is valid in the context of the current set
            # 1. All triples in b_new must be covered by (blocks \ {b}) U {b_new}
            #    This means:
            #    - Triples in b_new that were in b: covered by b_new (ok)
            #    - Triples in b_new that were NOT in b: must be covered by (blocks \ {b})
            #    - Triples in b that were NOT in b_new: must be covered by (blocks \ {b})
            
            # Let's check triples in b_new NOT in b
            b_set = set(b)
            b_new_set = set(b_new)
            
            # Triples to check for external coverage:
            # 1. New triples: in b_new but not in b
            # 2. Lost triples: in b but not in b_new
            
            # We can just check if the union of coverage from other blocks covers the "delta"
            
            # Efficient check:
            # For each triple in b_new:
            #   If it's not in b, it MUST be covered by another block in `blocks` (excluding b).
            # For each triple in b:
            #   If it's not in b_new, it MUST be covered by another block in `blocks` (excluding b).
            
            # Precompute coverage of other blocks?
            # Since we are in a loop, we can just iterate other blocks.
            # N_blocks is ~76. N_triples in block is 35.
            # 76 * 35 checks is small.
            
            other_blocks = [b2 for b2 in blocks if b2 != b]
            
            def is_covered_by_others(triple, other_blocks):
                for b2 in other_blocks:
                    if triple[0] in b2 and triple[1] in b2 and triple[2] in b2:
                        return True
                return False

            # Check new triples
            for triple in itertools.combinations(b_new, t):
                if triple not in b_set: # Rough check, better to use set of triples
                    # Actually, triple is a tuple. b_set is set of points.
                    # We need to check if triple is a subset of b.
                    if not all(pt in b_set for pt in triple):
                        if not is_covered_by_others(triple, other_blocks):
                            return None

            # Check lost triples
            for triple in itertools.combinations(b, t):
                if not all(pt in b_new_set for pt in triple):
                    if not is_covered_by_others(triple, other_blocks):
                        return None

            return (p, q, b_new)

        blocks = list(blocks)
        improved = True
        while improved:
            improved = False
            counts = compute_coverage(blocks)
            
            # Try to remove
            removable_idx = -1
            for i, b in enumerate(blocks):
                if is_removable(b, counts, blocks):
                    removable_idx = i
                    break
            
            if removable_idx != -1:
                blocks.pop(removable_idx)
                improved = True
                continue
            
            # Try to swap
            # Pick a random block to try swapping in
            if len(blocks) > 0:
                idx = random.randrange(len(blocks))
                b = blocks[idx]
                swap_result = try_swap(b, counts, blocks)
                if swap_result:
                    p, q, b_new = swap_result
                    blocks[idx] = b_new
                    improved = True
    
        return blocks

    best_blocks = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        seed += 1
        try:
            blocks = greedy_construct(seed)
            blocks = local_search(blocks)
            
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = blocks
                # If we hit the known bound or close to it, we can stop earlier?
                # 76 is the best known. If we get 76, we can stop.
                if best_size <= 76:
                    break
        except Exception as e:
            continue
            
    if best_blocks is None:
        # Fallback: return a simple greedy result
        blocks = greedy_construct(0)
        return blocks
        
    return best_blocks
