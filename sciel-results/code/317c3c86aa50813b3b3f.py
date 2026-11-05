import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    # Map all t-subsets to integer IDs
    tsubsets = list(itertools.combinations(range(v), t))
    tsubset_to_id = {ts: i for i, ts in enumerate(tsubsets)}
    n_tsubsets = len(tsubsets)
    
    # Precompute: for each 8-block, which t-subsets it covers
    # Using numpy bitmasks for speed
    # n_tsubsets = C(19,5) = 11628, which fits in an array of int64
    # Use a bytearray or bitarray approach for coverage tracking
    
    uncovered = np.ones(n_tsubsets, dtype=np.uint8)  # 1 = uncovered
    blocks = []
    
    # Precompute all k-blocks and their covered t-subset IDs
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)
    
    # For each k-block, compute the set of t-subset IDs it covers
    # This is expensive to precompute fully, so we compute on-the-fly but cache
    
    # Strategy: greedy with "hardest first" - prioritize uncovered t-subsets 
    # that are covered by fewest possible blocks (structurally isolated)
    
    # Precompute: for each t-subset, how many k-blocks can cover it
    # A t-subset S is covered by k-block B if S ⊆ B
    # Number of k-blocks containing a specific t-subset = C(v-t, k-t)
    # This is uniform for all t-subsets, so "hardest" isn't about count
    
    # Better strategy: greedy that at each step picks the block covering the 
    # most uncovered t-subsets, but with a twist: when starting a new block,
    # seed it with the uncovered t-subset that has the least "future coverage potential"
    
    # Let's use a refined greedy: 
    # 1. Find an uncovered t-subset T
    # 2. Build a block around T by adding points that maximize coverage of uncovered subsets
    # 3. Use a priority on uncovered subsets: prefer those with fewer remaining options
    
    # Precompute coverage masks for each possible extension
    # For a partial block P (list of points), and candidate point x not in P:
    #   gain(x) = number of uncovered t-subsets that contain P∪{x} minus those already covered
    
    # More efficient: for each t-subset, track which points it still needs
    
    # Let's use a different approach: iterated greedy with local search
    
    def greedy_build(seed_tsubset):
        """Build a block starting from seed_tsubset, greedily adding points."""
        block_set = set(seed_tsubset)
        block = list(block_set)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets that would be covered by adding x
                # i.e., t-subsets that contain all of block ∪ {x} and are uncovered
                new_block = frozenset(block_set | {x})
                # t-subsets contained in new_block that are uncovered
                gain = 0
                # We need t-subsets that are subsets of new_block and contain at least one element not in block_set
                # Actually, a t-subset S is covered by block∪{x} if S ⊆ (block∪{x})
                # and was not covered by block alone, so S must contain x and S\{x} ⊆ block
                # S\{x} has t-1 elements, all in block
                for rest in itertools.combinations(block, t-1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in tsubset_to_id:
                        sid = tsubset_to_id[ts]
                        if uncovered[sid]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x >= 0:
                block_set.add(best_x)
                block.append(best_x)
                block.sort()
        return block
    
    # Find the "hardest" uncovered t-subset: one whose extensions are most constrained
    # Heuristic: pick uncovered t-subset T such that the number of k-blocks containing T 
    # that also cover other uncovered subsets is minimized
    
    while np.any(uncovered):
        if time.time() > deadline - 5:
            break
        
        # Find uncovered t-subsets
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
        
        # Pick a seed: the uncovered t-subset that appears in the fewest 
        # "potentially useful" blocks. Approximate by picking randomly from uncovered,
        # but weighted toward those with rarer point combinations.
        
        # Simple heuristic: pick the uncovered t-subset with the lexicographically 
        # "rarest" point set, or just pick the first one and build greedily
        seed_idx = uncov_indices[0]
        seed_tsubset = tsubsets[seed_idx]
        
        block = greedy_build(seed_tsubset)
        blocks.append(block)
        
        # Update uncovered
        block_set = frozenset(block)
        for ts in itertools.combinations(block, t):
            if ts in tsubset_to_id:
                sid = tsubset_to_id[ts]
                uncovered[sid] = 0
    
    # If we ran out of time, we might have incomplete coverage.
    # Check and try to complete if time permits
    remaining = np.where(uncovered)[0]
    while len(remaining) > 0 and time.time() < deadline - 2:
        seed_idx = remaining[0]
        seed_tsubset = tsubsets[seed_idx]
        block = greedy_build(seed_tsubset)
        # Check if this block is new
        if block not in blocks:
            blocks.append(block)
        else:
            # Block already exists, try a different seed
            pass
        
        block_set = frozenset(block)
        for ts in itertools.combinations(block, t):
            if ts in tsubset_to_id:
                sid = tsubset_to_id[ts]
                uncovered[sid] = 0
        
        remaining = np.where(uncovered)[0]
    
    return blocks
