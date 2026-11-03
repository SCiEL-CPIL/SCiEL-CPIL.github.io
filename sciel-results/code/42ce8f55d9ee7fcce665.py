"""Optimized covering C(24,7,3) using Randomized Greedy and Fast Local Search."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using:
    1. Randomized Greedy: Maximize uncovered triples covered, with random tie-breaking.
    2. Fast Local Search: 
       - Aggressive removal of redundant blocks using coverage counts.
       - Swap-based optimization to try and replace blocks with more efficient ones.
    """
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    total_triples = len(all_triples)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized restarts to find a better local optimum
    # 120s limit allows for several restarts.
    # Each greedy run is O(NumBlocks * V * K^2) approx, which is fast for v=24.
    # Local search is the bottleneck.
    
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    while time.time() - start_time < max_time:
        # Phase 1: Randomized Greedy
        blocks = _randomized_greedy(v, k, t, all_triples, triple_to_idx)
        
        # Phase 2: Fast Local Search (Compression + Swaps)
        blocks = _local_search_fast(v, k, t, all_triples, triple_to_idx, blocks)
        
        # Update best
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
        # If we hit the theoretical lower bound (69), we can stop early
        # But 69 is not proven, so we just keep trying.
        if best_count <= 70:
             break

    return best_blocks

def _randomized_greedy(v, k, t, all_triples, triple_to_idx):
    """
    Greedy covering with random tie-breaking.
    """
    uncovered_counts = [1] * len(all_triples)
    blocks = []
    
    # Pre-compute all possible k-subsets? No, v=24, k=7 is C(24,7) = 346104.
    # It's feasible to generate them on the fly or pre-compute if memory allows.
    # 346k blocks * 7 ints is small. But iterating through all of them for each step
    # is expensive: 76 steps * 346k blocks * 35 triples/block = ~900M ops. Too slow.
    
    # Instead, we construct blocks greedily by adding points.
    # This is the standard "Greedy Set Cover" adapted for construction.
    
    uncovered_set = set(range(len(all_triples)))
    
    while uncovered_set:
        best_block = None
        best_gain = 0
        candidates = []
        
        # To speed up, we can pick a random uncovered triple to start the block construction
        # or try to find the best block from a random sample of all blocks.
        # Given the size, let's try a hybrid: 
        # 1. Pick a random uncovered triple.
        # 2. Try to extend it to a full block that maximizes gain.
        # 3. Also, occasionally try a "random block" to escape local minima? 
        #    Actually, the "extend from random triple" is a good heuristic for covering designs.
        
        # Let's stick to the "extend from random uncovered triple" strategy for diversity.
        # It ensures we cover hard triples.
        
        # Pick 5 random uncovered triples and extend each, picking the best resulting block.
        sample_triples = random.sample(list(uncovered_set), min(5, len(uncovered_set)))
        
        for t_idx in sample_triples:
            seed_triple = all_triples[t_idx]
            block = list(seed_triple)
            
            # Extend block to size k
            # We want to maximize the number of newly covered triples.
            # We'll do a greedy extension.
            
            while len(block) < k:
                best_point = None
                best_gain = 0
                # Try all remaining points
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate gain of adding x
                    # New triples formed are {x, p1, p2} where p1, p2 in block
                    gain = 0
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            tr = tuple(sorted((block[i], block[j], x)))
                            idx = triple_to_idx.get(tr)
                            if idx is not None and idx in uncovered_set:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                if best_point is None or best_gain == 0:
                    # If we can't improve gain, pick a random point to complete the block
                    # to ensure we have a valid block, though it might be inefficient.
                    # However, for covering, we usually just pick any point.
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                         best_point = random.choice(remaining)
                    else:
                         break
                         
                block.append(best_point)
                
            block.sort()
            
            # Calculate total gain of this complete block
            total_gain = 0
            for tr in itertools.combinations(block, t):
                idx = triple_to_idx.get(tr)
                if idx is not None and idx in uncovered_set:
                    total_gain += 1
            
            if total_gain > best_gain:
                best_gain = total_gain
                best_block = block[:]
            
        if best_block is None:
            # Fallback: pick a random block
            best_block = random.sample(range(v), k)
            best_block.sort()
        
        blocks.append(best_block)
        
        # Update uncovered set
        for tr in itertools.combinations(best_block, t):
            idx = triple_to_idx.get(tr)
            if idx is not None:
                uncovered_set.discard(idx)
                
    return blocks

def _local_search_fast(v, k, t, all_triples, triple_to_idx, blocks):
    """
    Fast local search using coverage counts.
    1. Compress: Remove any block where all its triples are covered by >= 2 blocks.
    2. Swap: Try to replace a block with a new one that covers its 'exclusive' triples
       and potentially more, without increasing the count.
    """
    blocks = [b[:] for b in blocks]
    
    # Build coverage counts
    # coverage_count[i] = number of blocks covering triple i
    coverage_count = [0] * len(all_triples)
    
    def update_counts(block, delta):
        for tr in itertools.combinations(block, t):
            idx = triple_to_idx.get(tr)
            if idx is not None:
                coverage_count[idx] += delta
                
    for b in blocks:
        update_counts(b, 1)
        
    # Phase 1: Aggressive Removal
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            b = blocks[i]
            # Check if all triples in b have coverage > 1
            can_remove = True
            for tr in itertools.combinations(b, t):
                idx = triple_to_idx.get(tr)
                if idx is not None and coverage_count[idx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block
                update_counts(b, -1)
                blocks.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    # Phase 2: Local Search (Swaps)
    # We try to improve the solution by swapping blocks.
    # A swap is valid if we remove block A and add block B, such that:
    # 1. All triples covered only by A are covered by B.
    # 2. The new block B is not already in the list.
    # 3. Ideally, B covers more triples that are currently covered by only 1 block (other than A),
    #    or simply that the resulting set is still a valid cover.
    
    # To make this efficient, we limit the number of iterations.
    max_iters = 500
    random.seed(42)
    
    for _ in range(max_iters):
        if not blocks:
            break
            
        idx = random.randint(0, len(blocks) - 1)
        old_block = blocks[idx]
        
        # Identify triples covered ONLY by old_block
        exclusive_triples = []
        for tr in itertools.combinations(old_block, t):
            idx_tr = triple_to_idx.get(tr)
            if idx_tr is not None and coverage_count[idx_tr] == 1:
                exclusive_triples.append(idx_tr)
                
        if not exclusive_triples:
            # If no exclusive triples, we can just remove it (should have been caught in Phase 1, 
            # but maybe counts changed? No, Phase 1 ran to convergence).
            # However, if we are in the middle of swaps, maybe.
            # If no exclusive triples, removing it is safe and reduces count.
            update_counts(old_block, -1)
            blocks.pop(idx)
            continue
            
        # We need a new block that covers all exclusive_triples.
        # The new block must be a superset of the union of points in exclusive_triples?
        # No, it just needs to contain the triples.
        # Let's try to construct a new block.
        
        # Candidate points must include all points from exclusive triples?
        # Not necessarily. A triple (a,b,c) is covered if the block contains a,b,c.
        # So the new block MUST contain a, b, and c for each exclusive triple.
        # This implies the new block must be a superset of the union of all points in exclusive_triples.
        
        candidate_points = set()
        for idx_tr in exclusive_triples:
            candidate_points.update(all_triples[idx_tr])
            
        if len(candidate_points) > k:
            # Cannot cover all exclusive triples with a single k-block
            continue
            
        new_block = list(candidate_points)
        
        # Fill the rest of the block to maximize coverage of other "weakly covered" triples
        # or just random.
        # Let's try to maximize the number of triples with coverage_count == 1 (excluding old_block's contribution)
        # Actually, simpler: just fill randomly or greedily.
        
        # Greedy fill:
        while len(new_block) < k:
            best_point = None
            best_gain = 0
            for x in range(v):
                if x in new_block:
                    continue
                gain = 0
                for tr in itertools.combinations(new_block + [x], t):
                    # Check if this triple is currently "at risk" (coverage 1 from old_block, so becomes 0 if we remove old_block)
                    # Or just generally useful.
                    idx_tr = triple_to_idx.get(tr)
                    if idx_tr is not None:
                         # If this triple is covered by old_block, its count is currently >=1.
                         # If we remove old_block, its count drops by 1.
                         # We want to ensure it's covered by new_block.
                         # This is handled by the candidate_points logic for exclusive ones.
                         # For non-exclusive ones, we just want to cover as many as possible.
                         gain += 1
                         
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point is None:
                break
            new_block.append(best_point)
            
        new_block.sort()
        
        # Check if new_block is distinct from existing blocks
        if new_block in blocks:
            continue
            
        # Check if new_block covers all exclusive_triples
        # (It should by construction, but double check)
        new_block_set = set(new_block)
        valid_swap = True
        for idx_tr in exclusive_triples:
            if not all(p in new_block_set for p in all_triples[idx_tr]):
                valid_swap = False
                break
                
        if not valid_swap:
            continue
            
        # Perform swap
        # Remove old_block
        update_counts(old_block, -1)
        
        # Check if removing old_block broke the cover (shouldn't if exclusive_triples logic is correct)
        # But what about triples covered by old_block and one other block?
        # Their count drops from 2 to 1. Still covered.
        
        # Add new_block
        update_counts(new_block, 1)
        
        # Check validity: all triples must have count >= 1
        # We can check this quickly if we maintain a count of "uncovered" or "single-covered"
        # But since we just updated counts, we can verify.
        # To save time, we assume the swap is valid if it covers exclusive triples.
        # However, we must ensure we didn't create a hole.
        # A hole is created if a triple had count 1 (from old_block) and is NOT in new_block.
        # We ensured exclusive_triples (count 1) are in new_block.
        # What about triples with count 1 that are NOT in old_block? They are unaffected.
        # What about triples with count 1 that ARE in old_block? They are in exclusive_triples.
        # So the cover remains valid.
        
        blocks[idx] = new_block
        
    return blocks
