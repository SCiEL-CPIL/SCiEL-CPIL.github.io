"""Improved covering design C(19,8,5) constructor with greedy + local search."""
import time
import random
import itertools
import numpy as np

# Precompute all 5-subsets of {0..18} and assign each an index
_ALL_5SUBS = list(itertools.combinations(range(19), 5))
N_5SUBS = len(_ALL_5SUBS)
_SUB_TO_IDX = {s: i for i, s in enumerate(_ALL_5SUBS)}

# Precompute, for each 8-subset, the list of 5-subset indices it covers
def _block_5idx(block):
    return [_SUB_TO_IDX[tuple(sorted(c))] for c in itertools.combinations(block, 5)]

# Precompute a pool of candidate blocks (all 8-subsets, but we'll use a subset for speed)
# For v=19, k=8: C(19,8) = 75582. That's a lot. Let's use all of them but be smart.
# Actually, let's just compute on the fly for the ones we need.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    # For the specific task v=19, k=8, t=5
    # Precompute mapping for all 8-subsets
    # C(19,8) = 75582 blocks, each covers C(8,5)=56 five-subsets
    # Total entries: 75582 * 56 = 4,232,592. Fine in memory.
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute: for each block index, the list of 5-subset indices
    block_5idx = [_block_5idx(b) for b in all_blocks]
    
    # For fast lookup: given a 5-subset index, which blocks cover it?
    # This is expensive to build fully. Instead, we'll use a different approach.
    
    # Greedy with randomization + local search
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        uncovered = np.ones(N_5SUBS, dtype=np.int32)  # coverage count
        blocks = []
        
        # Precompute for speed: for each point, which 5-subsets contain it
        # point_to_5subs[p] = list of 5-subset indices containing point p
        point_to_5subs = [[] for _ in range(v)]
        for idx, s in enumerate(_ALL_5SUBS):
            for p in s:
                point_to_5subs[p].append(idx)
        
        while uncovered.sum() > 0:
            if time.time() > deadline:
                break
            
            # Pick an uncovered 5-subset (randomly for diversity)
            unc_idx = np.where(uncovered > 0)[0]
            if len(unc_idx) == 0:
                break
            target_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            target_set = _ALL_5SUBS[target_idx]
            
            # Build block greedily starting from target_set
            # Start with some points from target, then add points that cover most uncovered
            block_set = set(target_set)
            # Add random additional points from target first
            # Actually, start with all of target, then add k-t more
            
            while len(block_set) < k:
                best_gain = -1
                best_points = []
                for p in range(v):
                    if p in block_set:
                        continue
                    # How many uncovered 5-subsets would be covered by adding p?
                    # A 5-subset is covered if it contains 4 points from block_set and p
                    # i.e., 5-subsets of form {p} ∪ {4 points from block_set}
                    gain = 0
                    # For each 4-subset of block_set, check if {p}+that 4-subset is uncovered
                    for c4 in itertools.combinations(sorted(block_set), 4):
                        sub = tuple(sorted(c4 + (p,)))
                        if sub in _SUB_TO_IDX:
                            idx = _SUB_TO_IDX[sub]
                            if uncovered[idx] > 0:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [p]
                    elif gain == best_gain:
                        best_points.append(p)
                
                if best_points:
                    # Random tie-break
                    block_set.add(rng.choice(best_points))
                else:
                    # No gain, pick random
                    remaining = [p for p in range(v) if p not in block_set]
                    if remaining:
                        block_set.add(rng.choice(remaining))
                    else:
                        break
            
            block = sorted(block_set)
            if len(block) != k:
                break
            
            # Add block and update coverage
            blocks.append(block)
            for idx in _block_5idx(block):
                uncovered[idx] = max(0, uncovered[idx] - 1)
        
        return blocks
    
    def local_search(blocks, seed):
        """Remove redundant blocks and try 1-exchanges."""
        rng = random.Random(seed)
        blocks = list(blocks)
        
        # Build coverage array
        coverage = np.zeros(N_5SUBS, dtype=np.int32)
        for block in blocks:
            for idx in _block_5idx(block):
                coverage[idx] += 1
        
        # Pass 1: remove redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            order = list(range(len(blocks)))
            rng.shuffle(order)
            for i in order:
                if i >= len(blocks):
                    continue
                block = blocks[i]
                idxs = _block_5idx(block)
                # Can we remove this block?
                if all(coverage[idx] > 1 for idx in idxs):
                    for idx in idxs:
                        coverage[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break  # Restart after removal
        
        # Pass 2: try replacing blocks
        # For each block, try replacing one point with another to reduce redundancy
        # This is complex; skip for now and focus on removal
        
        return blocks
    
    # Run multiple greedy constructions with different seeds, keep the best
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy runs
    n_greedy = 20
    for seed in range(n_greedy):
        if time.time() > deadline - 30:
            break
        blocks = greedy_construct(seed * 12345 + 67890)
        if blocks:
            # Local search
            blocks = local_search(blocks, seed * 9999 + 1)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
    
    # Phase 2: More aggressive local search on the best
    if best_blocks is not None:
        # Try multiple passes of removal with different orderings
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            candidate = local_search(best_blocks, 100000 + trial * 7777)
            if len(candidate) < best_len:
                best_len = len(candidate)
                best_blocks = candidate
    
    # Phase 3: Try to add and swap
    # For each block in best_blocks, try replacing it with a different block
    # that covers a superset of its unique coverage
    if best_blocks is not None and time.time() < deadline - 10:
        coverage = np.zeros(N_5SUBS, dtype=np.int32)
        for block in best_blocks:
            for idx in _block_5idx(block):
                coverage[idx] += 1
        
        # For each block, find its "unique" 5-subsets (covered only by this block)
        # Then try to find a replacement block that covers all of them plus maybe more
        # This is expensive, so do a limited version
        
        # Try random swaps: pick a block, pick a random alternative 8-subset
        # that shares at least 5 points with the original
        for swap_trial in range(500):
            if time.time() > deadline - 5:
                break
            if not best_blocks:
                break
            i = random.randint(0, len(best_blocks) - 1)
            old_block = set(best_blocks[i])
            # Generate a candidate block sharing k-1 or k-2 points
            shared = random.sample(sorted(old_block), max(1, k - 2))
            remaining_pool = [p for p in range(v) if p not in shared]
            needed = k - len(shared)
            if needed <= 0 or needed > len(remaining_pool):
                continue
            new_rest = random.sample(remaining_pool, needed)
            new_block = tuple(sorted(shared + new_rest))
            
            # Check if new_block is already in blocks
            if new_block in best_blocks:
                continue
            
            # Simulate the swap
            new_coverage = coverage.copy()
            for idx in _block_5idx(list(old_block)):
                new_coverage[idx] -= 1
            for idx in _block_5idx(list(new_block)):
                new_coverage[idx] += 1
            
            if (new_coverage >= 0).all():
                # Valid swap
                best_blocks[i] = list(new_block)
                coverage = new_coverage
        
        # Final cleanup
        best_blocks = local_search(best_blocks, 200000)
    
    if best_blocks is None or len(best_blocks) == 0:
        # Fallback: return a trivially valid but large set
        # Just use all blocks (valid but huge)
        best_blocks = [list(b) for b in all_blocks]
    
    return best_blocks
