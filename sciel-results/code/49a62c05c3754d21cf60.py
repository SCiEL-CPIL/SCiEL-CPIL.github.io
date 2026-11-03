"""Rewritten program for C(19,7,4) covering design."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

# Precompute all 4-subsets of 19 points
ALL_4SUBSETS = list(itertools.combinations(range(19), 4))
NUM_4SUBSETS = len(ALL_4SUBSETS)

# Precompute for each 4-subset, which 7-blocks contain it
# We'll build this on the fly for speed


def _compute_block_4subsets(block):
    """Return list of 4-subset indices covered by this block."""
    covered = []
    for comb in itertools.combinations(block, 4):
        # We need an index for this 4-subset
        # Use a precomputed mapping
        idx = _4SUBSET_INDEX.get(comb)
        if idx is not None:
            covered.append(idx)
    return covered


# Build index for 4-subsets
_4SUBSET_INDEX = {comb: i for i, comb in enumerate(ALL_4SUBSETS)}


def _greedy_cover(v, k, t, time_limit=115.0):
    """Greedy set cover with randomized restarts."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # All t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_index = {comb: i for i, comb in enumerate(all_tsubsets)}
    
    # Precompute for each point, which t-subsets it can be part of
    # For a block B and point x not in B, the new t-subsets covered are those
    # containing x and t-1 points from B
    
    best_blocks = None
    best_count = num_tsubsets + 1
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        restart_start = time.time()
        
        # Random ordering for tie-breaking
        rng = random.Random(num_restarts * 12345 + 7)
        
        # Uncovered t-subsets as a bitset (using Python int for efficiency)
        uncovered = (1 << num_tsubsets) - 1  # all bits set
        blocks = []
        
        # Precompute: for each point x, and each subset S of size t-1,
        # the t-subsets that contain x and all elements of S
        # Actually, let's use a different approach:
        # For each point x, precompute the list of t-subsets containing x
        point_to_tsubsets = defaultdict(list)
        for i, comb in enumerate(all_tsubsets):
            for x in comb:
                point_to_tsubsets[x].append(i)
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            
            # Pick the first uncovered t-subset (or random for diversity)
            # Find an uncovered t-subset
            # Use a heuristic: pick the uncovered t-subset with fewest candidate blocks
            # For simplicity, pick the lowest-index uncovered t-subset
            # But for diversity, we can randomize
            
            # Find lowest uncovered t-subset
            # This is O(num_tsubsets) in worst case, but we can optimize
            # Actually, let's just find any uncovered t-subset
            # Use a list of uncovered indices
            
            # For efficiency, maintain a list of uncovered t-subsets
            # But that's expensive to update. Let's use the bitset approach.
            
            # Find an uncovered t-subset
            # Get the lowest set bit
            if uncovered == 0:
                break
            
            # Find index of lowest set bit
            # Python: use bit_length and modulo
            # Actually, let's just iterate
            # For v=19, t=4, num_tsubsets = 3876, so this is fine
            
            # Pick a random uncovered t-subset for diversity
            # Convert uncovered to list of indices
            # This is expensive, so let's be smart
            
            # Use a simpler approach: maintain a list
            # Actually, for the first version, let's just use the bitset
            # and find the lowest set bit
            
            # Get index of lowest set bit
            # Python int: (uncovered & -uncovered).bit_length() - 1
            low_bit = uncovered & (-uncovered)
            idx = low_bit.bit_length() - 1
            target_tsubset = all_tsubsets[idx]
            
            # Start a new block with one point from target_tsubset
            # Choose which point to start with
            # For diversity, randomize
            start_point = target_tsubset[rng.randint(0, t - 1)]
            block = [start_point]
            block_set = {start_point}
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break
                
                # For each candidate point x not in block, compute gain
                best_gain = -1
                best_point = None
                candidates = [x for x in range(v) if x not in block_set]
                
                # Shuffle candidates for tie-breaking diversity
                rng.shuffle(candidates)
                
                for x in candidates:
                    # Gain = number of uncovered t-subsets that contain x 
                    # and t-1 points from current block
                    gain = 0
                    # For each (t-1)-subset of block, check if the t-subset
                    # formed by adding x is uncovered
                    for rest in itertools.combinations(block, t - 1):
                        tsubset = tuple(sorted(rest + (x,)))
                        tidx = tsubset_index.get(tsubset)
                        if tidx is not None:
                            # Check if uncovered
                            if (uncovered >> tidx) & 1:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point is not None:
                        # Tie: keep the first one (already shuffled)
                        pass
                
                if best_point is None:
                    break
                if best_gain <= 0:
                    # No gain, but we still need to fill the block
                    # Just pick a random candidate
                    best_point = candidates[0] if candidates else None
                    if best_point is None:
                        break
                
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) < k:
                # Failed to complete block, skip
                continue
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered: remove all t-subsets covered by this block
            for comb in itertools.combinations(block, t):
                tidx = tsubset_index.get(comb)
                if tidx is not None:
                    uncovered &= ~(1 << tidx)
            
            # Check if we're done
            if uncovered == 0:
                break
        
        # Check if we have a valid cover
        if uncovered == 0 and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        
        # If we're getting close to the known best, stop early
        if best_count <= 151:
            break
        
        # Limit restarts per time
        if restart_start and time.time() - restart_start > 10.0:
            # Each restart taking too long, reduce work
            pass
    
    if best_blocks is None:
        # Fallback: return a trivial (but large) cover
        # Just use all possible 7-blocks... that's too many
        # Use a simple greedy without restarts
        uncovered = (1 << num_tsubsets) - 1
        blocks = []
        while uncovered:
            low_bit = uncovered & (-uncovered)
            idx = low_bit.bit_length() - 1
            target_tsubset = all_tsubsets[idx]
            block = list(target_tsubset)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        tsubset = tuple(sorted(rest + (x,)))
                        tidx = tsubset_index.get(tsubset)
                        if tidx is not None and (uncovered >> tidx) & 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)
                block_set.add(best_point)
            if len(block) < k:
                break
            block.sort()
            blocks.append(tuple(block))
            for comb in itertools.combinations(block, t):
                tidx = tsubset_index.get(comb)
                if tidx is not None:
                    uncovered &= ~(1 << tidx)
        best_blocks = blocks
        best_count = len(blocks)
    
    return [list(b) for b in best_blocks]


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if v == 19 and k == 7 and t == 4:
        return _greedy_cover(19, 7, 4, time_limit=110.0)
    else:
        # Generic greedy
        all_tsubsets = list(itertools.combinations(range(v), t))
        num_tsubsets = len(all_tsubsets)
        tsubset_index = {comb: i for i, comb in enumerate(all_tsubsets)}
        
        uncovered = (1 << num_tsubsets) - 1
        blocks = []
        
        while uncovered and time.time() < deadline:
            low_bit = uncovered & (-uncovered)
            idx = low_bit.bit_length() - 1
            target_tsubset = all_tsubsets[idx]
            block = list(target_tsubset)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        tsubset = tuple(sorted(rest + (x,)))
                        tidx = tsubset_index.get(tsubset)
                        if tidx is not None and (uncovered >> tidx) & 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) < k:
                break
            block.sort()
            blocks.append(block)
            for comb in itertools.combinations(block, t):
                tidx = tsubset_index.get(comb)
                if tidx is not None:
                    uncovered &= ~(1 << tidx)
        
        return blocks
# EVOLVE-BLOCK-END
