"""Improved covering design C(24,7,3) using bitset-accelerated greedy + randomized restarts + redundancy removal."""
import itertools
import time
import random
import sys
from functools import lru_cache

# Precompute all 3-subsets of {0,...,v-1} and map to indices
# For the specific task v=24, k=7, t=3, but we keep it general.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute which t-subsets contain it
    # point_tsubs[p] = list of indices of t-subsets containing p
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        idx = tsub_to_idx[ts]
        for p in ts:
            point_tsubs[p].append(idx)

    # Bitset representation: use Python ints as bitsets
    # uncovered_mask: bit i is 1 if t-subset i is uncovered
    def greedy_cover(v, k, t, all_tsubs, tsub_to_idx, point_tsubs, seed=None, deadline=None):
        """Single greedy run with optional randomization."""
        rng = random.Random(seed)
        uncovered = (1 << len(all_tsubs)) - 1  # all bits set
        blocks = []
        
        while uncovered:
            # Check deadline
            if deadline and time.time() > deadline:
                break
            
            # Find first uncovered t-subset (or random one for randomization)
            if seed is not None and rng.random() < 0.3:
                # Randomly pick an uncovered t-subset
                bits = []
                u = uncovered
                while u:
                    lb = u & (-u)
                    idx = lb.bit_length() - 1
                    bits.append(idx)
                    u ^= lb
                if bits:
                    chosen_idx = rng.choice(bits)
                else:
                    break
            else:
                # Pick the lowest-index uncovered t-subset
                if uncovered == 0:
                    break
                lb = uncovered & (-uncovered)
                chosen_idx = lb.bit_length() - 1
            
            chosen_tsub = all_tsubs[chosen_idx]
            block = list(chosen_tsub)
            block_set = set(block)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                if deadline and time.time() > deadline:
                    break
                
                best_gain = -1
                best_points = []
                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute gain: number of uncovered t-subsets that would be covered by adding x
                    # A t-subset is covered by adding x if it contains x and (t-1) elements from block
                    gain = 0
                    # For each (t-1)-subset of current block, check if (that subset + x) is uncovered
                    # But we need to be efficient. Use precomputed point_tsubs.
                    # Actually, let's compute: for each t-subset containing x, check if the other t-1 elements are in block
                    # That's expensive. Let's use a different approach.
                    
                    # Alternative: for each t-subset that contains x, check if all other elements are in block
                    # We can iterate over point_tsubs[x] and check membership
                    for idx in point_tsubs[x]:
                        if (uncovered >> idx) & 1:
                            ts = all_tsubs[idx]
                            # Check if all elements of ts except x are in block
                            if all(p in block_set for p in ts if p != x):
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if not best_points:
                    break
                
                if len(best_points) == 1:
                    chosen_x = best_points[0]
                else:
                    chosen_x = rng.choice(best_points) if seed is not None else best_points[0]
                
                block.append(chosen_x)
                block_set.add(chosen_x)
            
            if len(block) < k:
                # Can't complete the block, skip
                continue
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered mask
            # For each t-subset of this block, clear its bit
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                uncovered &= ~(1 << idx)
        
        return [list(b) for b in blocks]

    def remove_redundant_blocks(blocks, v, k, t, all_tsubs, tsub_to_idx, deadline=None):
        """Remove blocks that are redundant (their t-subsets are all covered by other blocks)."""
        if not blocks:
            return blocks
        
        # Build coverage: for each t-subset, which blocks cover it
        # For efficiency, only do this if we have time
        if deadline and time.time() > deadline - 5:
            return blocks
        
        n_blocks = len(blocks)
        # For each block, compute the set of t-subset indices it covers
        block_coverage = []
        for b in blocks:
            cov = set()
            for ts in itertools.combinations(b, t):
                cov.add(tsub_to_idx[ts])
            block_coverage.append(cov)
        
        # Greedy removal: try to remove each block if its coverage is subset of union of others
        # Do multiple passes
        removed = [False] * n_blocks
        for _ in range(3):  # up to 3 passes
            if deadline and time.time() > deadline - 2:
                break
            for i in range(n_blocks):
                if removed[i]:
                    continue
                # Check if block i's coverage is fully covered by non-removed blocks
                covered_by_others = set()
                for j in range(n_blocks):
                    if j == i or removed[j]:
                        continue
                    covered_by_others |= block_coverage[j]
                if block_coverage[i] <= covered_by_others:
                    removed[i] = True
        
        return [blocks[i] for i in range(n_blocks) if not removed[i]]

    def verify_covering(blocks, v, t, all_tsubs, tsub_to_idx):
        """Check if all t-subsets are covered."""
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(tsub_to_idx[ts])
        return len(covered) == len(all_tsubs)

    # Strategy: multiple randomized greedy runs, keep the best (fewest blocks)
    # First, do a deterministic run
    best_blocks = None
    best_len = float('inf')
    
    # Deterministic run
    blocks = greedy_cover(v, k, t, all_tsubs, tsub_to_idx, point_tsubs, seed=None, deadline=deadline)
    if blocks and verify_covering(blocks, v, t, all_tsubs, tsub_to_idx):
        blocks = remove_redundant_blocks(blocks, v, k, t, all_tsubs, tsub_to_idx, deadline=deadline)
        if verify_covering(blocks, v, t, all_tsubs, tsub_to_idx):
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
    
    # Randomized runs
    if best_blocks is None:
        # Fallback: if deterministic didn't work, we need a valid covering
        # This shouldn't happen, but just in case
        blocks = greedy_cover(v, k, t, all_tsubs, tsub_to_idx, point_tsubs, seed=0, deadline=deadline)
        if blocks and verify_covering(blocks, v, t, all_tsubs, tsub_to_idx):
            best_blocks = blocks
            best_len = len(blocks)
    
    # Try randomized restarts
    num_tries = 0
    while time.time() < deadline - 5:
        seed = 1000 + num_tries
        blocks = greedy_cover(v, k, t, all_tsubs, tsub_to_idx, point_tsubs, seed=seed, deadline=deadline)
        if blocks:
            # Verify
            if verify_covering(blocks, v, t, all_tsubs, tsub_to_idx):
                blocks = remove_redundant_blocks(blocks, v, k, t, all_tsubs, tsub_to_idx, deadline=deadline)
                if verify_covering(blocks, v, t, all_tsubs, tsub_to_idx):
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = blocks
                        # Print progress to stderr for debugging
                        sys.stderr.write(f"Found covering with {best_len} blocks\n")
        num_tries += 1
        if num_tries > 200:  # safety limit
            break
    
    if best_blocks is None:
        # Emergency fallback: pure greedy without verification shortcuts
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks
    
    return best_blocks
