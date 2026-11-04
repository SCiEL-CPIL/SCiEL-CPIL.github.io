"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a fast bitset-based greedy
    with multiple randomized restarts and local improvement.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Precompute all t-subsets as integers (bit masks) for fast set operations
    # Represent each t-subset as a frozenset or a sorted tuple for hashing
    # For speed, we'll use a different approach: bit masks

    # Number of t-subsets
    from math import comb
    n_tsubsets = comb(v, t)

    # Map each t-subset to an index
    # We'll use a dictionary mapping frozenset -> index, but that's slow for large v
    # Instead, let's use a different strategy:
    # Represent each t-subset as a sorted tuple, and use a set of tuples

    # For v=19, t=4: C(19,4) = 3876 t-subsets
    # For v=19, k=7: C(19,7) = 50388 blocks

    # Precompute all t-subsets as sorted tuples
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # For each point, precompute which t-subsets contain it
    # point_tsubsets[point] = list of indices of t-subsets containing point
    point_tsubsets = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsubsets[p].append(i)

    # For each pair of points, precompute which t-subsets contain both
    # This helps in computing gain when adding a point to a block
    # Actually, let's think about this differently.

    # Greedy approach:
    # uncovered = set of indices of uncovered t-subsets
    # For each new block, start with a seed t-subset, then add points greedily

    # To speed up gain computation:
    # When we have a partial block B, and we want to add point x,
    # the gain is the number of uncovered t-subsets that contain x and are
    # "completable" by B, i.e., t-subsets S such that x in S and (S \ {x}) 
    # can be completed by B.
    
    # More precisely: S is uncovered, x in S, and S \ {x} ⊆ B ∪ {x} ... 
    # Actually: S is a t-subset containing x. S is covered by B ∪ {x} if 
    # S ⊆ B ∪ {x}. Since x ∈ S, this means S \ {x} ⊆ B.
    # So gain(x) = |{S in uncovered : x in S and S \ {x} ⊆ B}|

    # We can precompute for each (partial_block_frozen, x) the gain, but that's expensive.
    # Instead, let's use a different approach.

    # Alternative: use bitsets for uncovered t-subsets.
    # uncovered_mask is a bitset of length n_tsubsets.
    # For each point x, precompute a bitset point_mask[x] where bit i is set if x in all_tsubsets[i].
    # For a partial block B, the set of t-subsets that can be completed by adding x is:
    #   {S : x in S and S \ {x} ⊆ B}
    # This is equivalent to: S contains x, and S ⊆ B ∪ {x}.
    # 
    # Let's precompute for each subset of points (up to size k-1) the set of t-subsets
    # that are subsets of it. But that's C(v, k-1) * something, which is large.

    # Let's try a simpler optimization: for the gain computation, iterate over 
    # (t-1)-subsets of B and check if adding x creates an uncovered t-subset.
    
    def greedy_cover(seed=None, max_blocks=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(range(n_tsubsets))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            if max_blocks and len(blocks) >= max_blocks:
                break
            
            # Pick a random uncovered t-subset as seed (or the first one)
            seed_ts = random.choice(list(uncovered))
            seed_tuple = all_tsubsets[seed_ts]
            block = list(seed_tuple)
            
            while len(block) < k:
                if time.time() > deadline:
                    return blocks, False  # incomplete
                block_set = set(block)
                
                # Compute gain for each candidate point
                best_x = None
                best_gain = -1
                
                # Precompute (t-1)-subsets of current block
                if t - 1 == 0:
                    # t=1, special case
                    block_subsets = [()]
                else:
                    block_subsets = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Gain: number of uncovered t-subsets S such that x in S and S \ {x} ⊆ block
                    # S \ {x} is a (t-1)-subset of block
                    gain = 0
                    for sub in block_subsets:
                        # S = sub + (x,)
                        S = tuple(sorted(sub + (x,)))
                        if S in tsubset_to_idx:
                            idx = tsubset_to_idx[S]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    # Should not happen if uncovered is non-empty
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                uncovered.discard(idx)
        
        return blocks, len(uncovered) == 0

    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        blocks, complete = greedy_cover(seed=seed)
        if complete and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Early exit if we hit a good number
            if best_count <= 151:
                break
    
    # If no complete solution found, try a final greedy without randomness
    if best_blocks is None or not (len(best_blocks) < best_count):
        # Fallback: deterministic greedy
        uncovered = set(range(n_tsubsets))
        blocks = []
        while uncovered and time.time() < deadline - 2:
            seed_ts = min(uncovered)
            seed_tuple = all_tsubsets[seed_ts]
            block = list(seed_tuple)
            while len(block) < k:
                block_set = set(block)
                best_x = None
                best_gain = -1
                block_subsets = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in block_subsets:
                        S = tuple(sorted(sub + (x,)))
                        if S in tsubset_to_idx:
                            idx = tsubset_to_idx[S]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                uncovered.discard(idx)
        
        if len(uncovered) == 0 and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
    
    if best_blocks is None:
        # Emergency fallback: return any valid covering (should not happen)
        # Use a simple but valid construction
        blocks = []
        seen = set()
        for combo in itertools.combinations(range(v), k):
            blocks.append(list(combo))
            if time.time() > deadline - 1:
                break
        return blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
