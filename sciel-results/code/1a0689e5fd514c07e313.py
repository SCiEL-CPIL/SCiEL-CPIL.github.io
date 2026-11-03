import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We target 110s to be safe.
    deadline = start_time + 110.0
    
    # Precompute mapping from triple to index for O(1) lookup
    # There are C(24,3) = 2024 triples.
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Helper to get triple indices for a block
    # We will compute this on the fly to avoid memory overhead
    def get_block_triple_indices(block):
        return [triple_to_idx[tup] for tup in itertools.combinations(block, t)]

    # Helper to generate candidate blocks containing a specific triple
    def get_candidates_containing(triple):
        # triple is a tuple of 3 points
        # We need to choose k-t=4 points from the remaining v-t=21 points
        available_points = [p for p in range(v) if p not in triple]
        # Generate all combinations of 4 points
        for extra in itertools.combinations(available_points, k - t):
            yield tuple(sorted(triple + extra))

    def greedy_cover(random_state, time_limit=None):
        rng = random.Random(random_state)
        if time_limit is None:
            time_limit = 10.0
        local_deadline = time.time() + time_limit
        
        # uncovered[i] = 1 if triple i is not covered
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        selected_blocks = []
        
        # To speed up finding an uncovered triple, we can just scan or use a list.
        # Scanning 2024 elements is fast.
        
        while num_uncovered > 0 and time.time() < local_deadline:
            # Find an uncovered triple. 
            # Randomization helps escape local minima in greedy choices.
            # Pick a random index, check if uncovered. If not, scan.
            # Better: Keep a list of uncovered indices? 
            # Updating a list of 2024 items is fast.
            
            # Simple approach: Pick a random triple index, if covered, try again.
            # This is O(1) expected if coverage is high, O(N) if low.
            # Given ~80 blocks, coverage is dense.
            
            target_idx = -1
            while target_idx == -1:
                target_idx = rng.randint(0, num_triples - 1)
                if uncovered[target_idx] == 0:
                    target_idx = -1 # covered, retry
                    # If we loop too much, just scan
                    if rng.random() < 0.1: # Fallback check
                         # Find first uncovered
                         for i in range(num_triples):
                             if uncovered[i]:
                                 target_idx = i
                                 break
                         break
            
            if target_idx == -1:
                 # All covered?
                 if num_uncovered == 0:
                     break
                 # Fallback: scan
                 for i in range(num_triples):
                     if uncovered[i]:
                         target_idx = i
                         break
                 if target_idx == -1:
                     break

            target_triple = triples[target_idx]
            
            # Generate candidates
            best_block = None
            best_gain = -1
            
            # Iterate through candidates containing target_triple
            # There are C(21, 4) = 5985 candidates
            for block in get_candidates_containing(target_triple):
                # Calculate gain: number of currently uncovered triples covered by this block
                gain = 0
                # Block has C(7,3)=21 triples
                for tri_idx in get_block_triple_indices(block):
                    if uncovered[tri_idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if gain == 21: # Max possible
                        break
            
            if best_block is None:
                break
                
            # Add block
            selected_blocks.append(best_block)
            
            # Update uncovered
            for tri_idx in get_block_triple_indices(best_block):
                if uncovered[tri_idx]:
                    uncovered[tri_idx] = 0
                    num_uncovered -= 1
        
        return selected_blocks

    def trim_and_optimize(blocks, random_state, time_limit=5.0):
        """
        1. Remove redundant blocks.
        2. Try to replace blocks to cover more? (Swap local search)
        """
        rng = random.Random(random_state)
        local_deadline = time.time() + time_limit
        
        if not blocks:
            return []
        
        # Map block tuple to its index in the list for quick lookup?
        # We work with a list of block tuples.
        
        # Initialize coverage counts
        # counts[tri_idx] = number of blocks covering triple tri_idx
        counts = np.zeros(num_triples, dtype=np.int32)
        
        # Precompute triple indices for each block in the current set
        # This is O(N_blocks * 21)
        block_triple_indices = []
        for b in blocks:
            indices = get_block_triple_indices(b)
            block_triple_indices.append(indices)
            for idx in indices:
                counts[idx] += 1
        
        active = [True] * len(blocks)
        
        # Phase 1: Greedy Removal
        # Repeatedly remove a block if it doesn't cover any triple uniquely.
        changed = True
        while changed and time.time() < local_deadline:
            changed = False
            # Shuffle to make it random-ish
            indices_to_check = list(range(len(blocks)))
            rng.shuffle(indices_to_check)
            
            for i in indices_to_check:
                if not active[i]:
                    continue
                # Check if block i is essential
                essential = False
                for tri_idx in block_triple_indices[i]:
                    if counts[tri_idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove block i
                    active[i] = False
                    changed = True
                    for tri_idx in block_triple_indices[i]:
                        counts[tri_idx] -= 1
        
        # Phase 2: Local Search (Swap)
        # Try to replace an active block with another block that covers the same "essential" triples
        # and possibly more, or allows removal of another block.
        # Simplified Swap:
        # For each active block, check if it can be replaced by a block that covers all its 
        # currently uniquely covered triples (if any) and maintains validity.
        
        while time.time() < local_deadline:
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
            
            rng.shuffle(active_indices)
            made_change = False
            
            for i in active_indices:
                if not active[i]:
                    continue
                
                # Identify triples that are ONLY covered by block i
                unique_triples = []
                for tri_idx in block_triple_indices[i]:
                    if counts[tri_idx] == 1:
                        unique_triples.append(tri_idx)
                
                if not unique_triples:
                    # Can just remove it (should have been caught in Phase 1, but maybe counts changed?)
                    # Actually, if counts changed, we might have missed it.
                    active[i] = False
                    made_change = True
                    for tri_idx in block_triple_indices[i]:
                        counts[tri_idx] -= 1
                    continue
                
                # Find a replacement block that covers ALL unique_triples
                # Candidate blocks must contain all points in the union of unique_triples?
                # No, they must cover the triples.
                # Since unique_triples are triples, a replacement block must contain the points of each triple.
                # If unique_triples share points, the block must contain the union of those points.
                
                # Let's find the set of points that MUST be in the replacement block
                required_points = set()
                for tri_idx in unique_triples:
                    required_points.update(triples[tri_idx])
                
                # If required_points size > k, no single block can replace it.
                if len(required_points) > k:
                    continue
                
                # Generate candidates containing required_points
                # And check if they cover all unique_triples
                # Note: If a block contains the points of a triple, it covers that triple.
                # So if it contains required_points, it covers all unique_triples?
                # Yes, because unique_triples are subsets of required_points?
                # Wait. If unique_triples = [(0,1,2), (0,1,3)], required_points = {0,1,2,3}.
                # A block containing {0,1,2,3} covers both triples.
                
                available_extra = [p for p in range(v) if p not in required_points]
                need_extra = k - len(required_points)
                
                best_replacement = None
                best_score = -1
                
                # Sample candidates to avoid full enumeration if too many
                # C(24 - len(required), need_extra)
                # If required is small, this is large.
                # Let's limit candidates.
                candidates = list(itertools.combinations(available_extra, need_extra))
                if len(candidates) > 500:
                    candidates = rng.sample(candidates, 500)
                
                for extra in candidates:
                    cand_block = tuple(sorted(tuple(required_points) + extra))
                    # Verify it covers all unique_triples (should be true by construction)
                    # Check gain: how many NEW triples does it cover compared to current block i?
                    # Or rather, does it allow removing another block?
                    # Simpler metric: Total uncovered triples it covers (including those already covered by others).
                    # Actually, we just want to maintain validity.
                    # Since it covers unique_triples, validity is maintained.
                    # We prefer a replacement that covers *more* total triples (to help future removals).
                    
                    cand_tris = get_block_triple_indices(cand_block)
                    score = 0
                    for tri_idx in cand_tris:
                        # If this triple is covered only by block i, it's still covered.
                        # If it's covered by others, it's redundant.
                        # We want to maximize the "support" for other blocks?
                        # Let's just pick the first valid one or one with max total coverage.
                        score += 1
                    
                    # Actually, we should check if this replacement allows us to remove ANOTHER block.
                    # That's complex. Let's just do simple replacement for now.
                    # If we find a valid replacement, we swap.
                    # To make it useful, we should prefer replacements that cover triples 
                    # currently covered by other blocks (increasing their counts).
                    
                    # Let's calculate "potential": number of triples covered by cand that are 
                    # currently covered by >= 2 blocks (increasing count to >=3) 
                    # vs covered by 1 block (keeping at 1, but shifting source).
                    # Actually, just swapping to a block that covers more total triples is a good heuristic.
                    
                    if score > best_score:
                        best_score = score
                        best_replacement = cand_block
                
                if best_replacement is not None and best_replacement != blocks[i]:
                    # Perform Swap
                    old_block = blocks[i]
                    
                    # Remove old block from counts
                    for tri_idx in block_triple_indices[i]:
                        counts[tri_idx] -= 1
                    
                    # Add new block to counts
                    new_indices = get_block_triple_indices(best_replacement)
                    for tri_idx in new_indices:
                        counts[tri_idx] += 1
                    
                    # Update block list and indices
                    blocks[i] = best_replacement
                    block_triple_indices[i] = new_indices
                    made_change = True
            
            if not made_change:
                break
        
        return [b for i, b in enumerate(blocks) if active[i]]

    def verify_covering(blocks):
        covered = np.zeros(num_triples, dtype=np.uint8)
        for block in blocks:
            for tri_idx in get_block_triple_indices(block):
                covered[tri_idx] = 1
        return np.all(covered == 1)

    best_blocks = None
    best_len = float('inf')
    
    seed_counter = 0
    while time.time() < deadline:
        # Phase 1: Greedy
        blocks = greedy_cover(seed_counter, time_limit=5.0)
        
        if not blocks:
            seed_counter += 1
            continue
            
        # Phase 2: Trim and Local Search
        blocks = trim_and_optimize(blocks, seed_counter, time_limit=5.0)
        
        # Verify
        if verify_covering(blocks):
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
                if best_len <= 70:
                    break
        
        seed_counter += 1
        
        # If we are running out of time, do a final quick trim on best_blocks
        if time.time() > deadline - 10:
            break

    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback
        return greedy_cover(0, time_limit=5.0)
