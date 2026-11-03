import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using randomized greedy + simulated annealing
    local search to minimize block count.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_index = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    max_gain_per_block = len(list(itertools.combinations(range(k), t)))

    # Precompute for each triple, the list of triples that share at least one element with it?
    # Actually, we need: for a given block (k-set), which triples does it cover?
    # And for a given triple, which candidate blocks contain it?
    
    # Precompute: for each triple, list of (k-t)-subsets of remaining points
    # This is expensive to precompute fully, so we'll generate on the fly but cache per triple
    triple_remaining_cache = {}
    
    def get_remaining_for_triple(triple):
        """Get the list of points not in the triple."""
        if triple not in triple_remaining_cache:
            triple_set = set(triple)
            triple_remaining_cache[triple] = [p for p in points if p not in triple_set]
        return triple_remaining_cache[triple]

    def generate_candidate_blocks(triple):
        """Generate all k-subsets containing the given t-subset."""
        remaining_points = get_remaining_for_triple(triple)
        triple_set = set(triple)
        for comb in itertools.combinations(remaining_points, k - t):
            yield tuple(sorted(triple + comb))

    def compute_block_triples(block):
        """Get the list of triple indices covered by a block."""
        return [triple_index[sub] for sub in itertools.combinations(block, t)]

    def greedy_cover(random_seed=0):
        rng = random.Random(random_seed)
        coverage = np.zeros(num_triples, dtype=np.int32)
        blocks = []
        block_set = set()
        uncovered_indices = list(range(num_triples))
        
        while uncovered_indices:
            if time.time() > deadline:
                break
            
            idx = rng.choice(uncovered_indices)
            anchor_triple = all_triples[idx]
            
            best_block = None
            best_gain = -1
            
            for block in generate_candidate_blocks(anchor_triple):
                if block in block_set:
                    continue
                
                gain = 0
                for sub in itertools.combinations(block, t):
                    ti = triple_index[sub]
                    if coverage[ti] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if gain == max_gain_per_block:
                        break
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            block_set.add(best_block)
            
            for sub in itertools.combinations(best_block, t):
                ti = triple_index[sub]
                coverage[ti] += 1
                
            # Remove newly covered triples
            still_uncovered = []
            for u_idx in uncovered_indices:
                if coverage[u_idx] == 0:
                    still_uncovered.append(u_idx)
            uncovered_indices = still_uncovered
            
            if not uncovered_indices:
                break
                
        return blocks

    def remove_redundant_blocks(blocks, coverage):
        """Remove blocks that don't uniquely cover any triple."""
        if not blocks:
            return blocks, coverage
        
        block_set = set(blocks)
        current_blocks = list(blocks)
        current_coverage = coverage.copy()
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            i = 0
            while i < len(current_blocks):
                block = current_blocks[i]
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    ti = triple_index[sub]
                    if current_coverage[ti] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for sub in itertools.combinations(block, t):
                        current_coverage[triple_index[sub]] -= 1
                    current_blocks.pop(i)
                    block_set.discard(block)
                    improved = True
                else:
                    i += 1
        
        return current_blocks, current_coverage

    def simulated_annealing(blocks, coverage, rng, max_time=30.0):
        """
        Simulated annealing to minimize block count.
        Moves: 
          1. Remove a redundant block (if exists)
          2. Swap a block with another candidate block
          3. Replace a block with a different one (repair if needed)
        Objective: minimize number of blocks, subject to all triples covered.
        """
        sa_start = time.time()
        T0 = 10.0
        T_final = 0.01
        cooling_rate = 0.995
        
        # Work with copies
        current_blocks = list(blocks)
        current_coverage = coverage.copy()
        current_len = len(current_blocks)
        
        best_blocks = list(current_blocks)
        best_coverage = current_coverage.copy()
        best_len = current_len
        
        T = T0
        step = 0
        
        while T > T_final and time.time() < min(sa_start + max_time, deadline - 5.0):
            step += 1
            
            # Check for redundant blocks first - always accept if we can remove one
            i = 0
            removed = False
            while i < len(current_blocks):
                block = current_blocks[i]
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    ti = triple_index[sub]
                    if current_coverage[ti] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove this block
                    for sub in itertools.combinations(block, t):
                        current_coverage[triple_index[sub]] -= 1
                    current_blocks.pop(i)
                    current_len -= 1
                    removed = True
                    
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(current_blocks)
                        best_coverage = current_coverage.copy()
                    break
                else:
                    i += 1
            
            if removed:
                # Continue looking for more redundant blocks at same temperature
                continue
            
            # Try a swap: pick a random block, remove it, find uncovered triples,
            # try to cover them with a single new block
            if not current_blocks:
                break
                
            block_idx = rng.randrange(len(current_blocks))
            old_block = current_blocks[block_idx]
            
            # Remove old block temporarily
            for sub in itertools.combinations(old_block, t):
                current_coverage[triple_index[sub]] -= 1
            
            # Find uncovered triples
            uncovered = [ti for ti in range(num_triples) if current_coverage[ti] == 0]
            
            if not uncovered:
                # Old block was redundant - this shouldn't happen due to check above,
                # but if it does, we've effectively removed a block
                current_blocks.pop(block_idx)
                current_len -= 1
                if current_len < best_len:
                    best_len = current_len
                    best_blocks = list(current_blocks)
                    best_coverage = current_coverage.copy()
                continue
            
            # Try to find a new block that covers all uncovered triples
            # The new block must contain all points from all uncovered triples
            # Actually, it just needs to cover each uncovered triple (each triple must be a subset of the new block)
            # So the new block must be a superset of the union of all uncovered triples? No.
            # Each uncovered triple must be contained in the new block.
            # So the new block must contain all points that appear in any uncovered triple.
            
            required_points = set()
            for ti in uncovered:
                for p in all_triples[ti]:
                    required_points.add(p)
            
            if len(required_points) > k:
                # Can't cover all uncovered triples with a single block
                # Restore old block
                for sub in itertools.combinations(old_block, t):
                    current_coverage[triple_index[sub]] += 1
                continue
            
            # Generate candidate blocks that contain all required points
            # The block must be a k-subset containing all points in required_points
            if len(required_points) < k:
                remaining = [p for p in points if p not in required_points]
                found_new_block = None
                # Try a few random candidates
                for _ in range(min(20, len(remaining) + 1)):
                    extra = rng.sample(remaining, k - len(required_points))
                    candidate = tuple(sorted(required_points | set(extra)))
                    
                    # Check if this block covers all uncovered triples
                    valid = True
                    for ti in uncovered:
                        triple_set = set(all_triples[ti])
                        if not triple_set.issubset(set(candidate)):
                            valid = False
                            break
                    
                    if valid:
                        found_new_block = candidate
                        break
                
                if found_new_block is None:
                    # Restore old block
                    for sub in itertools.combinations(old_block, t):
                        current_coverage[triple_index[sub]] += 1
                    continue
                
                # Accept the swap
                new_block = found_new_block
                for sub in itertools.combinations(new_block, t):
                    current_coverage[triple_index[sub]] += 1
                
                current_blocks[block_idx] = new_block
                # current_len stays the same
                
                # Check if this enables further removals
                # (handled in next iteration)
                
            else:
                # len(required_points) == k, the block is uniquely determined
                candidate = tuple(sorted(required_points))
                valid = True
                for ti in uncovered:
                    triple_set = set(all_triples[ti])
                    if not triple_set.issubset(set(candidate)):
                        valid = False
                        break
                
                if valid:
                    new_block = candidate
                    for sub in itertools.combinations(new_block, t):
                        current_coverage[triple_index[sub]] += 1
                    current_blocks[block_idx] = new_block
                else:
                    # Restore
                    for sub in itertools.combinations(old_block, t):
                        current_coverage[triple_index[sub]] += 1
            
            # Also try: pick a random block, replace it with a completely different block
            # that covers the same or more triples, potentially making another block redundant
            if step % 3 == 0:
                block_idx2 = rng.randrange(len(current_blocks))
                old_block2 = current_blocks[block_idx2]
                
                # Remove old block2
                for sub in itertools.combinations(old_block2, t):
                    current_coverage[triple_index[sub]] -= 1
                
                uncovered2 = [ti for ti in range(num_triples) if current_coverage[ti] == 0]
                
                if uncovered2:
                    required_points2 = set()
                    for ti in uncovered2:
                        for p in all_triples[ti]:
                            required_points2.add(p)
                    
                    if len(required_points2) <= k:
                        if len(required_points2) < k:
                            remaining2 = [p for p in points if p not in required_points2]
                            for _ in range(min(10, len(remaining2) + 1)):
                                extra2 = rng.sample(remaining2, k - len(required_points2))
                                candidate2 = tuple(sorted(required_points2 | set(extra2)))
                                valid2 = True
                                for ti in uncovered2:
                                    if not set(all_triples[ti]).issubset(set(candidate2)):
                                        valid2 = False
                                        break
                                if valid2:
                                    for sub in itertools.combinations(candidate2, t):
                                        current_coverage[triple_index[sub]] += 1
                                    current_blocks[block_idx2] = candidate2
                                    break
                        else:
                            candidate2 = tuple(sorted(required_points2))
                            valid2 = True
                            for ti in uncovered2:
                                if not set(all_triples[ti]).issubset(set(candidate2)):
                                    valid2 = False
                                    break
                            if valid2:
                                for sub in itertools.combinations(candidate2, t):
                                    current_coverage[triple_index[sub]] += 1
                                current_blocks[block_idx2] = candidate2
                    
                    if time.time() > deadline - 5.0:
                        # Restore if we removed and didn't replace
                        for sub in itertools.combinations(old_block2, t):
                            if current_coverage[triple_index[sub]] == 0:
                                pass  # This is getting complex; just break
                        break
                    
                else:
                    # old_block2 was redundant, we've removed it
                    current_blocks.pop(block_idx2)
                    current_len -= 1
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(current_blocks)
                        best_coverage = current_coverage.copy()
                
                # If we didn't successfully replace, restore
                # Check if any triple is uncovered
                any_uncovered = np.any(current_coverage == 0)
                if any_uncovered:
                    # Restore old_block2
                    for sub in itertools.combinations(old_block2, t):
                        current_coverage[triple_index[sub]] += 1
                    if len(current_blocks) < current_len:
                        current_blocks.append(old_block2)
                        current_len = len(current_blocks)
            
            # Cool down
            T *= cooling_rate
        
        return best_blocks, best_coverage, best_len

    best_blocks = None
    best_coverage = None
    best_len = float('inf')
    
    # Run multiple randomized greedy instances
    t0 = time.time()
    blocks_0 = greedy_cover(0)
    t1 = time.time()
    t_greedy = t1 - t0
    
    if len(blocks_0) < best_len:
        # Compute coverage for blocks_0
        cov_0 = np.zeros(num_triples, dtype=np.int32)
        for block in blocks_0:
            for sub in itertools.combinations(block, t):
                cov_0[triple_index[sub]] += 1
        best_blocks = blocks_0
        best_coverage = cov_0
        best_len = len(blocks_0)
    
    # Run simulated annealing on the best so far
    if best_blocks and time.time() < deadline - 10.0:
        rng = random.Random(42)
        sa_blocks, sa_cov, sa_len = simulated_annealing(best_blocks, best_coverage, rng, max_time=30.0)
        if sa_len < best_len:
            best_blocks = sa_blocks
            best_coverage = sa_cov
            best_len = sa_len
    
    # Estimate how many more greedy runs we can do
    time_left = deadline - time.time()
    if t_greedy > 0 and time_left > 0:
        num_extra_runs = int(time_left / (t_greedy + 0.5))  # +0.5 for SA overhead estimate
    else:
        num_extra_runs = 50
    
    num_extra_runs = min(num_extra_runs, 200)
    
    for i in range(1, num_extra_runs + 1):
        if time.time() > deadline - 15.0:
            break
        
        seed = i
        blocks = greedy_cover(seed)
        
        if len(blocks) < best_len:
            cov = np.zeros(num_triples, dtype=np.int32)
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    cov[triple_index[sub]] += 1
            best_blocks = blocks
            best_coverage = cov
            best_len = len(blocks)
            
            # Try SA on this new best
            if time.time() < deadline - 10.0:
                rng = random.Random(1000 + i)
                sa_blocks, sa_cov, sa_len = simulated_annealing(best_blocks, best_coverage, rng, max_time=15.0)
                if sa_len < best_len:
                    best_blocks = sa_blocks
                    best_coverage = sa_cov
                    best_len = sa_len
    
    # Final SA pass on the overall best
    if best_blocks and time.time() < deadline - 5.0:
        rng = random.Random(9999)
        sa_blocks, sa_cov, sa_len = simulated_annealing(best_blocks, best_coverage, rng, max_time=10.0)
        if sa_len < best_len:
            best_blocks = sa_blocks
            best_coverage = sa_cov
            best_len = sa_len
    
    if best_blocks is None:
        best_blocks = greedy_cover(0)
    
    return [list(b) for b in best_blocks]
