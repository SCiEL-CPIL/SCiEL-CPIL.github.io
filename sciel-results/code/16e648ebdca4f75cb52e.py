"""Covering design C(24,7,3) - block-level set cover with local search."""
import itertools
import time
import random
from collections import defaultdict
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 145.0  # safety margin below 150s kill

    # All t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # For each point, list of t-subset indices containing it
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            point_tsubs[x].append(i)

    # Generate candidate blocks: for each uncovered triple during greedy, 
    # we'll pick the best block from a precomputed pool.
    # Precompute all C(24,7) blocks is 346104 - too many to store all at once
    # but we can generate them on demand or use a smart candidate pool.

    # Strategy: block-level greedy set cover.
    # At each step, pick the block that covers the most uncovered triples.
    # To make this fast, we maintain a "gain" array and update incrementally.

    def block_greedy_set_cover(seed, max_blocks=None):
        """Block-level greedy: pick blocks maximizing uncovered triple coverage."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        uncovered_count = n_t
        blocks = []
        
        if max_blocks is None:
            max_blocks = 200  # generous upper bound
        
        while uncovered_count > 0 and len(blocks) < max_blocks:
            if time.time() > deadline - 5.0:
                break
            
            # Find an uncovered triple to start from
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            start_i = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            start_triple = all_t_subsets[start_i]
            
            # Generate candidate blocks containing this triple
            # A block is a k-subset containing the t-subset
            # Number of such blocks: C(v-t, k-t) = C(21, 4) = 5985
            remaining_points = [x for x in range(v) if x not in set(start_triple)]
            candidate_blocks = []
            for comb in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(start_triple + comb))
                candidate_blocks.append(block)
            
            # Score each candidate: count how many uncovered triples it covers
            best_block = None
            best_gain = -1
            best_gain_indices = []
            
            # To speed up, we can sample if candidate pool is large
            # C(21,4) = 5985, manageable
            for block in candidate_blocks:
                gain = 0
                gain_indices = []
                for ts in itertools.combinations(block, t):
                    idx = t_idx[ts]
                    if uncovered[idx]:
                        gain += 1
                        gain_indices.append(idx)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_gain_indices = gain_indices
            
            if best_block is None or best_gain == 0:
                # Fallback: just take the triple + random points
                block = list(start_triple)
                block_set = set(block)
                while len(block) < k:
                    x = rng.choice(remaining_points)
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                best_block = tuple(sorted(block))
                best_gain_indices = []
                for ts in itertools.combinations(best_block, t):
                    idx = t_idx[ts]
                    if uncovered[idx]:
                        best_gain_indices.append(idx)
            
            blocks.append(list(best_block))
            for idx in best_gain_indices:
                uncovered[idx] = False
            uncovered_count -= len(best_gain_indices)
        
        return blocks

    def remove_redundant(blocks):
        """Remove blocks whose triples are all covered by other blocks."""
        if len(blocks) <= 1:
            return blocks
        
        # Build coverage count
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[ts] += 1
        
        new_blocks = []
        for block in blocks:
            can_keep = False
            for ts in itertools.combinations(block, t):
                if cover_count[ts] <= 1:
                    can_keep = True
                    break
            if can_keep:
                new_blocks.append(block)
        
        # If we removed some, recount
        if len(new_blocks) < len(blocks):
            cover_count2 = defaultdict(int)
            for block in new_blocks:
                for ts in itertools.combinations(block, t):
                    cover_count2[ts] += 1
            # Verify all triples still covered
            for ts in all_t_subsets:
                if cover_count2[ts] == 0:
                    # Restore
                    return blocks
            return new_blocks
        
        return blocks

    def local_search_swap(blocks, seed=0, max_iters=50):
        """Try swapping one block for another to reduce total count or improve coverage."""
        rng = random.Random(seed)
        if len(blocks) <= 2:
            return blocks
        
        # Build coverage
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[ts] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            n = len(blocks)
            order = list(range(n))
            rng.shuffle(order)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                block = blocks[idx]
                # Check if removable
                removable = True
                for ts in itertools.combinations(block, t):
                    if cover_count[ts] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove
                    blocks.pop(idx)
                    for ts in itertools.combinations(block, t):
                        cover_count[ts] -= 1
                    improved = True
                    break
        
        return blocks

    def verify_and_fix(blocks):
        """Ensure all triples are covered; if not, add minimal blocks."""
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        
        all_set = set(all_t_subsets)
        uncovered = all_set - covered
        
        if not uncovered:
            return blocks
        
        # Add blocks to cover remaining
        blocks = list(blocks)
        for ts in sorted(uncovered):
            if time.time() > deadline - 3.0:
                break
            # Find a block containing this triple
            block = list(ts)
            block_set = set(block)
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
            block.sort()
            blocks.append(block)
        
        return blocks

    # --- Main search ---
    best_blocks = None
    best_count = float('inf')

    # Phase 1: Multiple block-level greedy set covers with different seeds
    for seed in range(20):
        if time.time() > deadline - 30.0:
            break
        blocks = block_greedy_set_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]

    # Phase 2: Local search on best
    if best_blocks is not None:
        improved = remove_redundant(best_blocks[:])
        if len(improved) < best_count:
            best_count = len(improved)
            best_blocks = improved[:]

        for ls_seed in range(10):
            if time.time() > deadline - 10.0:
                break
            improved = local_search_swap(best_blocks[:], seed=100 + ls_seed, max_iters=30)
            improved = remove_redundant(improved)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved[:]

    # Phase 3: More diverse greedy runs if time allows
    for seed in range(20, 60):
        if time.time() > deadline - 15.0:
            break
        blocks = block_greedy_set_cover(seed)
        blocks = remove_redundant(blocks)
        blocks = local_search_swap(blocks, seed=seed + 200, max_iters=20)
        blocks = remove_redundant(blocks)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]

    # Final verification
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            ts = min(uncovered)
            block = list(ts)
            block_set = set(block)
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
            block.sort()
            blocks.append(block)
            for ts2 in itertools.combinations(block, t):
                uncovered.discard(ts2)
        best_blocks = blocks

    # Final fix
    best_blocks = verify_and_fix(best_blocks)

    return best_blocks
