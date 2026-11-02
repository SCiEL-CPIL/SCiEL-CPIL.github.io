"""Covering design C(19,8,5) via fast greedy + local search with time guard."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(12345)

    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    # Map each t-subset to an integer index for fast lookup
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # For each k-subset (block), precompute which t-subsets it covers
    # But C(19,8) = 75582 blocks, each covers C(8,5)=56 t-subsets.
    # Precompute coverage mask for each possible block? That's 75582 * 56 bits.
    # Instead, we'll compute coverage on the fly using numpy bitmasks.
    # Represent uncovered t-subsets as a numpy boolean array of size num_tsubs.

    def block_covers_tsub_indices(block):
        """Return list of indices of t-subsets covered by this block."""
        return [tsub_to_idx[sub] for sub in itertools.combinations(block, t)]

    def greedy_cover(uncovered_mask, deadline):
        """Greedy: pick block that covers most uncovered t-subsets."""
        blocks = []
        n_uncovered = uncovered_mask.sum()
        while n_uncovered > 0:
            if time.time() > deadline:
                break
            # Find the best block: try all C(v,k) blocks and pick max coverage
            # For speed, use a randomized/greedy approach:
            # Start with a random t-subset that's uncovered, build a block around it.
            # Then try to improve by swapping points.
            
            # Pick a random uncovered t-subset to seed the block
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            seed_tsub = all_tsubs[seed_idx]
            block = list(seed_tsub)
            
            # Greedily add points that cover most uncovered t-subsets
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in tsub_to_idx:
                            idx = tsub_to_idx[ts]
                            if uncovered_mask[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
            
            block.sort()
            
            # Try local improvement: swap points to cover more
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(block)):
                    for x in range(v):
                        if x in block:
                            continue
                        new_block = block[:]
                        new_block[i] = x
                        new_block.sort()
                        # Count coverage of new_block on uncovered
                        new_cover = sum(1 for sub in itertools.combinations(new_block, t)
                                        if uncovered_mask[tsub_to_idx[sub]])
                        old_cover = sum(1 for sub in itertools.combinations(block, t)
                                        if uncovered_mask[tsub_to_idx[sub]])
                        if new_cover > old_cover:
                            block = new_block
                            improved = True
                            break
                    if improved:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            for sub in itertools.combinations(block, t):
                idx = tsub_to_idx[sub]
                uncovered_mask[idx] = False
            n_uncovered = uncovered_mask.sum()
        
        return blocks

    # Initial greedy pass
    uncovered_mask = np.ones(num_tsubs, dtype=bool)
    blocks = greedy_cover(uncovered_mask, deadline)
    
    # Check if we have full coverage
    if uncovered_mask.sum() > 0:
        # Continue greedy until done or timeout
        while uncovered_mask.sum() > 0 and time.time() < deadline:
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            seed_tsub = all_tsubs[seed_idx]
            block = list(seed_tsub)
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in tsub_to_idx:
                            idx = tsub_to_idx[ts]
                            if uncovered_mask[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = tsub_to_idx[sub]
                uncovered_mask[idx] = False
    
    if time.time() > deadline:
        return blocks

    # Local search: remove redundant blocks and do 1-exchange
    def is_block_redundant(block_idx, blocks):
        """Check if removing block_idx still leaves all t-subsets covered."""
        block = blocks[block_idx]
        # For each t-subset covered by this block, check if another block covers it
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            # Check if any other block covers this t-subset
            found = False
            for j, other_block in enumerate(blocks):
                if j == block_idx:
                    continue
                if all(p in other_block for p in sub):
                    found = True
                    break
            if not found:
                return False
        return True

    def try_remove_redundant(blocks, deadline):
        """Remove redundant blocks iteratively."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if is_block_redundant(i, blocks):
                    blocks.pop(i)
                    changed = True
        return blocks

    def try_1_exchange(blocks, deadline):
        """Try replacing one block with another to reduce total count or maintain coverage with fewer blocks."""
        # This is complex; instead, try to find a block that can be replaced by a block covering more uncovered t-subsets
        # For now, just do redundant removal
        return blocks

    # Run redundant removal
    blocks = try_remove_redundant(blocks, deadline)

    # Try more greedy restarts if time permits
    if time.time() < deadline - 10:
        best_blocks = blocks
        best_count = len(blocks)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            uncovered_mask = np.ones(num_tsubs, dtype=bool)
            blocks2 = greedy_cover(uncovered_mask, deadline - 5)
            if uncovered_mask.sum() == 0:
                blocks2 = try_remove_redundant(blocks2, deadline - 2)
                if len(blocks2) < best_count:
                    best_blocks = blocks2
                    best_count = len(blocks2)
        blocks = best_blocks

    return blocks
