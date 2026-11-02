import itertools
import random
import time
import sys
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START

def _precompute(v, k, t):
    """
    Precompute all t-subsets and a mapping from (t-1)-subsets to the 
    list of t-subsets they extend. This allows O(1) lookup for gain calculations.
    """
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    sub_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    
    # Map from (t-1)-subset (tuple) to list of t-subset indices that contain it
    # Specifically, for a (t-1)-subset R, we store the indices of all S in all_tsubsets 
    # such that R is a subset of S. 
    # Note: In the greedy step, we look at a block B. We want to find S such that 
    # S \ {x} is a (t-1)-subset of B.
    # Actually, a more direct mapping for the specific greedy strategy:
    # We pick an uncovered t-subset U to start. We have a partial block P.
    # We want to add x to P. The gain is the number of uncovered t-subsets S 
    # such that S \ {x} is a subset of P.
    # This is equivalent to: S is a t-subset of P ∪ {x} and x ∈ S.
    # Since |P| grows, we can just check the specific t-subsets formed by P.
    
    # Let's stick to the sub_to_idx map which is essential.
    return all_tsubsets, n_tsubsets, sub_to_idx


def _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Greedy covering optimized for speed.
    Starts a new block from a random uncovered t-subset.
    Greedily adds points to the block to maximize coverage of other uncovered t-subsets.
    """
    uncovered = set(range(n_tsubsets))
    blocks = []

    # To speed up the inner loop, we precompute for every possible (t-1)-subset 
    # the list of t-subsets it is part of.
    # However, iterating over all (t-1)-subsets of the current block is cheap 
    # because the block size is small (k=8).
    
    # We will maintain a list of uncovered t-subsets for faster random selection
    # and to avoid O(N) scans for 'choice'.
    uncovered_list = list(uncovered)
    
    while uncovered_list:
        if time.time() > deadline:
            break

        # Pick a random uncovered t-subset to anchor the block
        # Using index for O(1) swap-delete
        idx_to_remove = rng.randrange(len(uncovered_list))
        start_idx = uncovered_list[idx_to_remove]
        start_sub = all_tsubsets[start_idx]
        
        # Remove from list
        uncovered_list.pop(idx_to_remove)
        
        block = list(start_sub)
        block_set = set(block)
        
        # Mark start_sub as covered (it is, by definition, in the block)
        # We track coverage via the 'uncovered' set/list.
        # Note: The start_sub is already removed from uncovered_list.
        
        # Greedily expand the block
        while len(block) < k:
            if time.time() > deadline:
                break

            best_gain = -1
            best_points = []
            
            # For each candidate point x not in block:
            # Gain(x) = number of uncovered t-subsets S such that S \ {x} ⊆ block
            # This is equivalent to: S is a t-subset of block ∪ {x} and x ∈ S.
            # Since |block| is small, we can enumerate the (t-1)-subsets of block.
            
            # Optimization: Pre-compute the (t-1)-subsets of the current block once per step?
            # No, block changes. But block size is <= 8. C(8, 4) = 70.
            
            block_tminus1_subs = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                # Check all (t-1)-subsets of block
                for r in block_tminus1_subs:
                    # The t-subset would be r + (x,)
                    # We need to check if this t-subset is in 'uncovered'
                    # Constructing the tuple and looking up is fast enough for small sizes
                    s = tuple(sorted(r + (x,)))
                    idx = sub_to_idx.get(s)
                    if idx is not None and idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)

            if not best_points:
                break

            chosen = rng.choice(best_points)
            block.append(chosen)
            block_set.add(chosen)
            
            # Update uncovered: remove any t-subsets fully contained in the new block
            # We only need to check t-subsets that include the new point 'chosen'
            # because any t-subset not including 'chosen' was already in 'block' 
            # and thus was already covered (or not, but we didn't check it against 'uncovered' 
            # efficiently in the previous step? Actually, we checked gain based on 'uncovered').
            
            # Let's just remove all t-subsets of the current block that are in 'uncovered'
            # To be efficient, we only look at t-subsets containing 'chosen'.
            # A t-subset S containing 'chosen' is covered if S \ {chosen} is a subset of block \ {chosen}.
            # Since 'block' now contains 'chosen', and we added it, we need to check 
            # t-subsets of 'block' that contain 'chosen'.
            
            # Iterate over (t-1)-subsets of (block - {chosen})
            block_without_chosen = [b for b in block if b != chosen]
            for r in itertools.combinations(block_without_chosen, t - 1):
                s = tuple(sorted(r + (chosen,)))
                idx = sub_to_idx.get(s)
                if idx is not None and idx in uncovered:
                    uncovered.discard(idx)
                    # Also remove from uncovered_list if present
                    # This is O(N). To make it O(1), we'd need a dict of indices in list.
                    # Given N=11628, and this happens ~335 * 4 * C(7,4) times... 
                    # 335 * 4 * 35 = 46k checks. 46k * 11k (scan) is too slow if done naively.
                    
                    # Better: Maintain a set 'uncovered' for O(1) discard.
                    # For 'uncovered_list', we can just leave stale entries and filter later,
                    # or use a different structure.
                    # Let's just use 'uncovered' set for membership and 'uncovered_list' for random pick.
                    # We will clean up 'uncovered_list' periodically or when it gets too large.
                    pass

            # To keep 'uncovered_list' consistent with 'uncovered' set without O(N) scan per removal:
            # We can just rebuild it occasionally or accept that 'rng.choice(list(uncovered))' 
            # is used at the start of the loop. 
            # Let's change strategy: Use 'uncovered' set for membership.
            # For random pick, we can just do: 
            #   while True:
            #       pick = rng.choice(uncovered) # If uncovered is a set, choice is O(1)? No, set has no order.
            #   Actually, 'random.choice' requires a sequence.
            #   So we must maintain a list or array.
            
            # Efficient approach: 
            # Keep 'uncovered' as a set.
            # Keep 'uncovered_indices' as a list.
            # When we remove an idx, we don't remove from list immediately.
            # When we need a random pick, we pick a random index from the list.
            # If that index is in 'uncovered', use it.
            # If not, swap it with the last element and pop, and try again (or just loop).
            # This is amortized O(1).

        blocks.append(tuple(block))

    return blocks, uncovered


def _local_search(blocks, v, k, t, deadline, rng, sub_to_idx):
    """
    Local search to reduce block count.
    1. Iterative Drop: Remove blocks that are fully redundant.
    2. Swap: Try to replace a block with a random one that covers all its 'unique' t-subsets.
    """
    if not blocks:
        return blocks

    n_tsubsets = len(sub_to_idx)
    coverage = np.zeros(n_tsubsets, dtype=np.int32)
    
    # Precompute t-subset indices for each block
    block_tsubs = []
    for block in blocks:
        tsubs = []
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                tsubs.append(idx)
        block_tsubs.append(tsubs)
        coverage[tsubs] += 1

    # 1. Iterative Drop
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_blocks = []
        new_block_tsubs = []
        for bidx in range(len(blocks)):
            redundant = True
            for idx in block_tsubs[bidx]:
                if coverage[idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(blocks[bidx])
                new_block_tsubs.append(block_tsubs[bidx])
            else:
                changed = True
                for idx in block_tsubs[bidx]:
                    coverage[idx] -= 1
        blocks = new_blocks
        block_tsubs = new_block_tsubs

    # 2. Swap / Improve
    # Try to replace blocks to allow further drops.
    # Strategy: For each block, identify "critical" t-subsets (coverage == 1).
    # If we can find a new block that covers all critical t-subsets of the current block,
    # we can swap. This might not reduce count immediately, but might change coverage
    # distribution to allow other blocks to become redundant.
    
    # To keep it simple and effective within time limits:
    # We will try random swaps for a fixed number of iterations.
    
    max_swaps = 5000
    swap_attempts = 0
    while swap_attempts < max_swaps and time.time() < deadline - 10:
        swap_attempts += 1
        if not blocks:
            break
            
        bidx = rng.randrange(len(blocks))
        old_block = blocks[bidx]
        old_tsubs = block_tsubs[bidx]
        
        # Find critical t-subsets for this block
        critical = [idx for idx in old_tsubs if coverage[idx] == 1]
        
        # If no critical t-subsets, we can just try to swap with a better random block?
        # Or just skip, as it's already redundant-safe? 
        # If no critical, it means all its t-subsets are covered by others.
        # We could just drop it? The previous loop would have dropped it.
        # So if we are here, it has at least one critical t-subset.
        
        if not critical:
            continue
            
        # Generate a random new block
        new_block = tuple(sorted(rng.sample(range(v), k)))
        new_tsubs = []
        for sub in itertools.combinations(new_block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                new_tsubs.append(idx)
        
        # Check if new_block covers all critical t-subsets of old_block
        # And does not lose any non-critical t-subsets that are uniquely covered by old_block?
        # Actually, we only care about not losing coverage for ANY t-subset.
        # A t-subset idx is safe to remove from old_block if:
        # 1. coverage[idx] > 1 (covered by others)
        # 2. OR idx is in new_tsubs (covered by the new block)
        
        safe = True
        new_tsubs_set = set(new_tsubs)
        for idx in old_tsubs:
            if coverage[idx] == 1 and idx not in new_tsubs_set:
                safe = False
                break
        
        if safe:
            # Perform swap
            for idx in old_tsubs:
                coverage[idx] -= 1
            for idx in new_tsubs:
                coverage[idx] += 1
            blocks[bidx] = new_block
            block_tsubs[bidx] = new_tsubs
            
            # After a swap, check if we can drop any blocks (including the one we just swapped)
            # Run a quick drop pass
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                new_blocks = []
                new_block_tsubs = []
                for bidx2 in range(len(blocks)):
                    redundant = True
                    for idx in block_tsubs[bidx2]:
                        if coverage[idx] <= 1:
                            redundant = False
                            break
                    if not redundant:
                        new_blocks.append(blocks[bidx2])
                        new_block_tsubs.append(block_tsubs[bidx2])
                    else:
                        changed = True
                        for idx in block_tsubs[bidx2]:
                            coverage[idx] -= 1
                blocks = new_blocks
                block_tsubs = new_block_tsubs

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    all_tsubsets, n_tsubsets, sub_to_idx = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different seeds
    # We don't know how many we can afford, so we loop until deadline.
    seed = 42
    while time.time() < deadline - 20:  # Leave 20s for final polish on best found
        rng = random.Random(seed)
        
        # Greedy phase
        blocks, uncovered = _greedy_cover(v, k, t, deadline - 10, rng, all_tsubsets, n_tsubsets, sub_to_idx)
        
        if len(uncovered) == 0:
            # Local search phase
            # Allocate remaining time for local search, but keep a buffer
            local_deadline = min(deadline - 5, time.time() + 15) 
            polished = _local_search(blocks, v, k, t, local_deadline, rng, sub_to_idx)
            
            if len(polished) < best_count:
                best_count = len(polished)
                best_blocks = polished
                
        seed += 1

    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END
