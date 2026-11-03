import itertools
import random
import time
import sys
import math
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    # Deadline guard
    deadline = time.time() + 145
    
    # 1. Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute masks for all blocks
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Precompute t-subsets within each block for faster gain calculation
    block_tsubs = []
    for b in all_blocks:
        tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs)
        
    # Helper to iterate set bits
    def iter_set_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    # 2. Greedy Initialization - optimized
    def greedy_cover(rng):
        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        
        # Randomize t-subset order
        order = list(range(n_tsubs))
        rng.shuffle(order)
        
        while uncovered_mask:
            if time.time() > deadline - 20:
                return None
            
            # Find first uncovered t-subset in order
            seed_idx = -1
            for idx in order:
                if (uncovered_mask >> idx) & 1:
                    seed_idx = idx
                    break
            if seed_idx == -1:
                break
                
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            
            # Greedily add points - use precomputed t-subsets
            while len(block) < k:
                best_gain = -1
                best_point = -1
                tie_points = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None:
                            if (uncovered_mask >> idx) & 1:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        tie_points = [x]
                    elif gain == best_gain and gain > -1:
                        tie_points.append(x)
                        
                if best_point >= 0:
                    if len(tie_points) > 1:
                        best_point = rng.choice(tie_points)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                        block.append(best_point)
                        block_set.add(best_point)
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            bi = block_to_idx[tuple(block)]
            block_mask = block_masks[bi]
            uncovered_mask &= ~block_mask
            
        if uncovered_mask:
            return None
        return blocks

    # 3. Remove Redundant Blocks
    def remove_redundant(blocks, deadline):
        if not blocks:
            return blocks
        
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        
        for bi in block_indices:
            mask = block_masks[bi]
            for idx in iter_set_bits(mask):
                cover_count[idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                bi = block_indices[i]
                mask = block_masks[bi]
                redundant = True
                for idx in iter_set_bits(mask):
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in iter_set_bits(mask):
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    changed = True
                    break
        return blocks

    # 4. Local Search: try to replace blocks with better ones
    def local_search(blocks, deadline):
        if not blocks:
            return blocks
        
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        
        for bi in block_indices:
            mask = block_masks[bi]
            for idx in iter_set_bits(mask):
                cover_count[idx] += 1
        
        improved = True
        max_iters = 500
        iter_count = 0
        
        while improved and time.time() < deadline - 10 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try removing each block and see if we can cover the gap with a better block
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                    
                bi = block_indices[i]
                mask = block_masks[bi]
                
                # Check if this block is the only cover for any t-subset
                essential = False
                for idx in iter_set_bits(mask):
                    if cover_count[idx] == 1:
                        essential = True
                        break
                
                if essential:
                    continue
                
                # Try to find a block that covers the same t-subsets with fewer total blocks
                # Actually, let's try: remove this block, then greedily add blocks to cover uncovered
                # But that's complex. Instead, try swapping with a block that has more overlap.
                
                # For now, just try to find if there's a block that covers all the same t-subsets
                # and allows removal of another block
                pass
        
        # Simpler local search: try to find redundant blocks after each removal
        return blocks

    # 5. More aggressive redundancy removal with multiple passes
    def aggressive_redundancy(blocks, deadline):
        if not blocks:
            return blocks
        
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        
        for bi in block_indices:
            mask = block_masks[bi]
            for idx in iter_set_bits(mask):
                cover_count[idx] += 1
        
        changed = True
        passes = 0
        max_passes = 20
        
        while changed and time.time() < deadline - 5 and passes < max_passes:
            changed = False
            passes += 1
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                bi = block_indices[i]
                mask = block_masks[bi]
                redundant = True
                for idx in iter_set_bits(mask):
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in iter_set_bits(mask):
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    changed = True
                    break
        
        return blocks

    # Main
    rng = random.Random(42)
    best_blocks = None
    best_count = float('inf')
    
    # Try several greedy starts with different random seeds
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        rng_seed = 42 + restart * 137
        local_rng = random.Random(rng_seed)
        init_blocks = greedy_cover(local_rng)
        if init_blocks:
            # Apply aggressive redundancy removal
            reduced = aggressive_redundancy(init_blocks, deadline)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = [list(b) for b in reduced]
    
    if best_blocks is None:
        # Fallback: Simple greedy without masks to ensure validity
        covered = set()
        blocks = []
        while len(covered) < n_tsubs:
            best_block = None
            best_gain = -1
            for b in all_blocks:
                gain = 0
                for ts in itertools.combinations(b, t):
                    if ts not in covered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            if best_block is None:
                break
            blocks.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                covered.add(ts)
        best_blocks = blocks
        best_count = len(blocks)

    return best_blocks
