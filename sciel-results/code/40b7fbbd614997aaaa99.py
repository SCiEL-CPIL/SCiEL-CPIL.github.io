import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    n_tsets = len(all_tsets)
    
    # For each block (k-subset), precompute the list of tset indices it covers
    # This is expensive for large v, but v=24 is small.
    # Number of k-subsets is C(24,7) = 346104.
    # Each covers C(7,3) = 35 triples.
    # Total operations ~ 12 million. This is fine.
    
    # However, we don't need to precompute for ALL k-subsets, only the ones we consider.
    # But in greedy, we consider many.
    # Let's precompute the "signature" of every possible block?
    # Memory: 346104 blocks * 35 ints * 4 bytes ~ 48 MB. Acceptable.
    
    block_to_tsets = {}
    
    def get_block_tsets(block):
        """Return list of tset indices covered by this block."""
        # block is a tuple of k points
        if block in block_to_tsets:
            return block_to_tsets[block]
        
        covered = []
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx[ts]
            covered.append(idx)
        
        block_to_tsets[block] = covered
        return covered
    
    def greedy_construct(seed):
        random.seed(seed)
        coverage = [0] * n_tsets
        blocks = []
        uncovered_count = n_tsets
        
        # To speed up finding uncovered tsets, we can maintain a set of uncovered indices
        # But updating it is O(35) per block addition.
        # Checking min(coverage) is O(2024).
        # Let's just scan coverage.
        
        while uncovered_count > 0 and time.time() < deadline:
            # Find an uncovered tset
            # We can pick the one with lowest coverage (0)
            # To add variety, pick randomly among those with 0
            candidates = [i for i in range(n_tsets) if coverage[i] == 0]
            if not candidates:
                break
            target_idx = random.choice(candidates)
            target_tset = all_tsets[target_idx]
            
            # Build a block greedily around target_tset
            # Start with the points in target_tset
            block_points = list(target_tset)
            block_set = set(block_points)
            
            # We need to add k - t points
            # At each step, pick the point that maximizes gain in coverage of uncovered tsets
            # Gain of adding x to partial block P:
            # Number of tsets S such that S is uncovered, x in S, and S \ {x} subset of P
            
            while len(block_points) < k and time.time() < deadline:
                best_x = -1
                best_gain = -1
                
                # Precompute (t-1)-subsets of current partial block
                partial_combos = list(itertools.combinations(block_points, t - 1))
                
                # Candidates are all points not in block_points
                # v=24, so at most 24 candidates
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # A tset S is covered by P U {x} if x in S and S \ {x} subset of P
                    # S must have size t.
                    # So S = {x} U R, where R is a (t-1)-subset of P.
                    # We need S to be uncovered (coverage[S] == 0).
                    
                    for R in partial_combos:
                        # S = R U {x}
                        # Construct the tuple for lookup
                        # Optimization: Since R is sorted and x is not in R, we can merge efficiently
                        # But tuple(sorted(...)) is fine for small sizes
                        s_tuple = tuple(sorted(list(R) + [x]))
                        idx = tset_to_idx[s_tuple]
                        if coverage[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1 or best_gain == 0:
                    # No point improves coverage of *uncovered* sets
                    # Pick a random point to fill the block
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                block_points.append(best_x)
                block_set.add(best_x)
            
            block = tuple(sorted(block_points))
            
            # Add block
            blocks.append(block)
            covered_tsets = get_block_tsets(block)
            for idx in covered_tsets:
                if coverage[idx] == 0:
                    uncovered_count -= 1
                coverage[idx] += 1
        
        return blocks, coverage

    def local_search(blocks, coverage, seed):
        random.seed(seed)
        
        # Precompute tsets for each block
        block_tsets = [get_block_tsets(b) for b in blocks]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Remove redundant blocks
            # A block is redundant if all its tsets are covered by other blocks
            # i.e., coverage[idx] >= 2 for all idx in block_tsets[i]
            
            i = 0
            while i < len(blocks):
                redundant = True
                for idx in block_tsets[i]:
                    if coverage[idx] < 2:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for idx in block_tsets[i]:
                        coverage[idx] -= 1
                    blocks.pop(i)
                    block_tsets.pop(i)
                    improved = True
                    # Do not increment i, check same index again (now next block)
                else:
                    i += 1
            
            # Pass 2: Try to swap points to enable removal
            # This is heuristic. We look for a block and a point swap that doesn't break coverage
            # and ideally increases redundancy.
            if not improved:
                # Try random swaps
                # Pick a random block
                if blocks:
                    i = random.randint(0, len(blocks) - 1)
                    b = blocks[i]
                    b_set = set(b)
                    
                    # Try swapping each point in b with a point outside b
                    # We want the new block to cover all tsets that were uniquely covered by b
                    unique_tsets = [idx for idx in block_tsets[i] if coverage[idx] == 1]
                    
                    if unique_tsets:
                        # We must keep these tsets covered
                        # If we swap p with q, the new block must cover all unique_tsets
                        # Actually, if a unique tset contains p, it will be lost unless q is in it
                        # If a unique tset does not contain p, it is safe
                        
                        # Let's try all swaps
                        best_swap = None
                        best_score = -1
                        
                        for p in b:
                            for q in range(v):
                                if q in b_set:
                                    continue
                                
                                # Check if new block covers all unique_tsets
                                new_b_list = list(b)
                                new_b_list.remove(p)
                                new_b_list.append(q)
                                new_b = tuple(sorted(new_b_list))
                                
                                # Get tsets of new block
                                new_tsets = get_block_tsets(new_b)
                                new_tset_set = set(new_tsets)
                                
                                # Check if all unique_tsets are in new_tset_set
                                # Note: unique_tsets are indices.
                                valid = True
                                for idx in unique_tsets:
                                    if idx not in new_tset_set:
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                # Score: number of *additional* tsets covered by new block
                                # that are currently covered by < 2 blocks (i.e. 0 or 1)
                                # But we can't have 0 because they are covered by b, so they are 1.
                                # We want to increase coverage of tsets that are 1, to make them 2.
                                # Or cover tsets that are 0? No, they are covered by b.
                                
                                # Let's count how many tsets in new_b have coverage == 1
                                # And how many in old b had coverage == 1
                                # We want to increase the count of tsets with coverage >= 2
                                
                                # Simple score: count of tsets in new_b with coverage >= 1
                                # This is always 35.
                                
                                # Better score: count of tsets in new_b with coverage >= 2
                                # Minus count of tsets in old_b with coverage >= 2
                                
                                old_score = sum(1 for idx in block_tsets[i] if coverage[idx] >= 2)
                                new_score = sum(1 for idx in new_tsets if coverage[idx] >= 2)
                                
                                # Also, we must ensure we don't uncover anything.
                                # We already checked unique_tsets.
                                
                                if new_score > old_score:
                                    if new_score > best_score:
                                        best_score = new_score
                                        best_swap = (p, q, new_b, new_tsets)
                        
                        if best_swap:
                            p, q, new_b, new_tsets = best_swap
                            # Perform swap
                            # Remove old block coverage
                            for idx in block_tsets[i]:
                                coverage[idx] -= 1
                            # Add new block coverage
                            for idx in new_tsets:
                                coverage[idx] += 1
                            
                            blocks[i] = new_b
                            block_tsets[i] = new_tsets
                            improved = True
    
    # Main loop: Try multiple seeds
    best_blocks = None
    best_len = float('inf')
    
    # Run greedy with different seeds
    num_attempts = 0
    start_time = time.time()
    
    while time.time() < deadline - 5:
        seed = num_attempts
        blocks, coverage = greedy_construct(seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
        
        # Local search
        blocks, coverage = local_search(blocks, coverage, seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
        
        num_attempts += 1
        # If we hit the lower bound or a very good number, stop?
        # Best known is 76.
        if best_len <= 76:
            # Try a few more to see if we can get 75? Unlikely.
            # But let's give it a few more seconds.
            if time.time() > start_time + 30:
                break

    if best_blocks is None:
        # Fallback: just return the first greedy result
        blocks, _ = greedy_construct(0)
        return blocks
        
    return best_blocks
