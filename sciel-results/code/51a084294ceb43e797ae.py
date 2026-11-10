"""Covering design C(v, k, t) – deterministic greedy with local search."""
import itertools
import time
import random
from collections import defaultdict

_DEADLINE_S = 115.0  # Safety margin from 150s limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _count_tsubs_in_block(block, t):
    """Count how many t-subsets of block are covered by other blocks."""
    # This is used in local search, but we need the global context.
    # We will pass the necessary data structures.
    pass


def _greedy_construct(v, k, t):
    """
    Construct a covering using a deterministic greedy strategy.
    At each step, choose the block that covers the most uncovered t-subsets.
    Tie-breaking: lexicographically smallest block.
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    # Pre-compute all possible k-subsets? 
    # C(19, 7) = 50388. This is small enough to iterate over if needed, 
    # but computing gain for all is expensive.
    # Instead, we use a heuristic to find a good block quickly.
    # However, for v=19, k=7, C(19,7)=50388.
    # Number of t-subsets in a block: C(7,4) = 35.
    # Total t-subsets: C(19,4) = 3876.
    
    # A faster greedy: 
    # 1. Find the uncovered t-subset with the "highest demand" (e.g., random or min).
    # 2. Find the best k-subset containing it.
    
    # Let's try a standard greedy set cover:
    # Maintain a list of uncovered t-subsets.
    # For each candidate block, calculate how many uncovered t-subsets it covers.
    # Select the best block, add it, update uncovered.
    
    # Optimization: 
    # Since v=19 is small, we can precompute the list of all k-subsets.
    all_ksubs = list(itertools.combinations(range(v), k))
    
    # For each k-subset, store its t-subsets
    ksub_to_tsubs = {}
    for ksub in all_ksubs:
        ksub_to_tsubs[ksub] = set(itertools.combinations(ksub, t))
    
    # To speed up gain calculation, we can maintain a count of how many times 
    # each t-subset is covered by *uncovered* blocks? No, that's complex.
    
    # Alternative:
    # Keep `uncovered` as a set.
    # For each step, we need to find the k-subset maximizing |ksub_to_tsubs[ksub] & uncovered|.
    # With 50388 ksubs and ~3876 tsubs, this is 50388 * 35 set intersections per step.
    # Steps needed: ~150.
    # 150 * 50388 * 35 ~ 264 million operations. Might be too slow in Python.
    
    # Heuristic approach:
    # Pick an uncovered t-subset `target`.
    # Only consider k-subsets containing `target`.
    # Number of such k-subsets: C(v-t, k-t) = C(19-4, 7-4) = C(15, 3) = 455.
    # For each of these 455 k-subsets, calculate gain.
    # This is much faster.
    
    while uncovered:
        if _time_left() < 1.0:
            break
            
        # Pick a target t-subset. 
        # Strategy: Pick the one with the fewest candidate k-subsets? 
        # All have the same number of candidates (C(15,3)=455).
        # So pick randomly or lexicographically first for determinism?
        # Let's pick the lexicographically smallest for determinism in construction.
        target = min(uncovered)
        
        # Find all k-subsets containing target
        remaining_points = [p for p in range(v) if p not in target]
        candidates = []
        for comb in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(list(target) + list(comb)))
            candidates.append(block)
            
        best_block = None
        best_gain = -1
        
        for block in candidates:
            gain = 0
            for tsub in ksub_to_tsubs[block]:
                if tsub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        for tsub in ksub_to_tsubs[best_block]:
            uncovered.discard(tsub)
            
    return blocks


def _local_search(blocks, v, k, t, max_iterations=500):
    """
    Improve the covering by:
    1. Removing redundant blocks.
    2. Replacing blocks with better ones that cover their unique t-subsets.
    """
    if not blocks:
        return blocks
    
    all_tsubs = list(itertools.combinations(range(v), t))
    
    # Precompute t-subsets for each block
    block_tsubs = [set(itertools.combinations(b, t)) for b in blocks]
    
    # Count coverage of each t-subset
    # Map t-subset -> list of block indices covering it
    tsub_to_blocks = defaultdict(list)
    for i, b_tsubs in enumerate(block_tsubs):
        for tsub in b_tsubs:
            tsub_to_blocks[tsub].append(i)
    
    # 1. Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            # Check if block i is redundant
            # i.e., for every t-subset in block i, there is another block covering it
            redundant = True
            for tsub in block_tsubs[i]:
                # Check if any other block covers tsub
                # tsub_to_blocks[tsub] contains i. If len > 1, it's covered by another.
                if len(tsub_to_blocks[tsub]) == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for tsub in block_tsubs[i]:
                    tsub_to_blocks[tsub].remove(i)
                # Adjust indices in tsub_to_blocks for blocks after i
                for tsub in all_tsubs:
                    if i in tsub_to_blocks[tsub]:
                        # This shouldn't happen if we removed correctly, 
                        # but we need to update indices of blocks > i
                        pass
                
                # Update indices in tsub_to_blocks
                # This is O(N * T). N=150, T=3876. ~570k ops. Acceptable.
                for tsub in all_tsubs:
                    new_list = []
                    for j in tsub_to_blocks[tsub]:
                        if j < i:
                            new_list.append(j)
                        elif j > i:
                            new_list.append(j - 1)
                        # j == i is removed
                    tsub_to_blocks[tsub] = new_list
                    
                blocks.pop(i)
                block_tsubs.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    # 2. Block Replacement
    # Try to replace each block with a different one that covers the same "unique" t-subsets
    # and potentially more.
    
    # Identify unique t-subsets for each block
    # A t-subset is "unique" to block i if only block i covers it.
    
    for _ in range(max_iterations):
        if _time_left() < 2.0:
            break
            
        improved = False
        
        for i in range(len(blocks)):
            # Find t-subsets only covered by block i
            unique_tsubs = set()
            for tsub in block_tsubs[i]:
                if len(tsub_to_blocks[tsub]) == 1:
                    unique_tsubs.add(tsub)
                    
            if not unique_tsubs:
                continue # Block is redundant, should have been removed. But maybe not yet.
                
            # Try to find a replacement block that covers all unique_tsubs
            # and has a higher total gain (covers more non-unique tsubs as well, or just same).
            # Actually, if it covers all unique_tsubs, it's a valid replacement.
            # We want to minimize block count, so replacement doesn't reduce count directly.
            # But it might make other blocks redundant.
            
            # Heuristic: Try to find a block that covers unique_tsubs and maximizes 
            # the number of *other* t-subsets it covers (to potentially make others redundant).
            
            # Candidate generation:
            # Pick one unique_tsub, say `target`.
            # Any replacement must contain `target`.
            # Generate candidates containing `target`.
            
            target = next(iter(unique_tsubs))
            remaining_points = [p for p in range(v) if p not in target]
            candidates = []
            for comb in itertools.combinations(remaining_points, k - t):
                new_block = tuple(sorted(list(target) + list(comb)))
                if new_block != tuple(blocks[i]):
                    candidates.append(new_block)
                    
            best_new_block = None
            best_score = -1
            
            for new_block in candidates:
                # Check if it covers all unique_tsubs
                new_block_set = set(new_block)
                covers_all = True
                for ut in unique_tsubs:
                    if not all(p in new_block_set for p in ut):
                        covers_all = False
                        break
                if not covers_all:
                    continue
                    
                # Calculate score: number of t-subsets covered by new_block 
                # that are NOT covered by any other block (except i, which is being replaced).
                # Actually, simpler: count how many t-subsets of new_block are covered ONLY by block i (currently).
                # If we replace i with new_block, these t-subsets will still be covered.
                # T-subsets of new_block covered by other blocks: they are already covered.
                # T-subsets of new_block NOT covered by other blocks: they become covered by new_block.
                
                # We want to maximize the "new" coverage? No, we want to maintain coverage.
                # The goal is to enable redundancy removal.
                # A good replacement is one that overlaps heavily with other blocks,
                # so that unique_tsubs of *other* blocks might become non-unique?
                # No, that's complex.
                
                # Simple heuristic: Replace with the block that covers the most t-subsets
                # that are currently covered by exactly one block (block i).
                # This ensures we don't lose coverage.
                
                # Let's just pick the lexicographically first valid replacement for now.
                # Or the one that maximizes overlap with existing blocks?
                
                # Let's try: maximize number of t-subsets in new_block that are covered by >= 1 other block.
                # This minimizes the "unique" burden of the new block.
                
                overlap_count = 0
                for tsub in itertools.combinations(new_block, t):
                    # Check if covered by any block j != i
                    # tsub_to_blocks[tsub] contains i.
                    # If len(tsub_to_blocks[tsub]) > 1, it's covered by another.
                    if len(tsub_to_blocks[tsub]) > 1:
                        overlap_count += 1
                        
                if overlap_count > best_score:
                    best_score = overlap_count
                    best_new_block = new_block
                    
            if best_new_block is not None:
                # Perform replacement
                old_block = blocks[i]
                blocks[i] = list(best_new_block)
                
                # Update tsub_to_blocks
                # Remove i from all tsubs of old_block
                for tsub in block_tsubs[i]:
                    if i in tsub_to_blocks[tsub]:
                        tsub_to_blocks[tsub].remove(i)
                        
                # Add i to all tsubs of new_block
                new_block_tsubs = set(itertools.combinations(best_new_block, t))
                for tsub in new_block_tsubs:
                    tsub_to_blocks[tsub].append(i)
                    
                block_tsubs[i] = new_block_tsubs
                improved = True
                
        if not improved:
            break
            
    return blocks


def construct(v, k, t):
    random.seed(42)
    
    # Phase 1: Greedy Construction
    blocks = _greedy_construct(v, k, t)
    
    # Phase 2: Local Search
    blocks = _local_search(blocks, v, k, t)
    
    # Verify validity (optional, but good for debugging)
    # all_tsubs = list(itertools.combinations(range(v), t))
    # covered = set()
    # for b in blocks:
    #     covered.update(itertools.combinations(b, t))
    # if covered != set(all_tsubs):
    #     # Fallback: Add blocks for uncovered tsubs
    #     uncovered = set(all_tsubs) - covered
    #     for tsub in uncovered:
    #         # Find a block containing tsub
    #         # Just pick the first one? Or construct one.
    #         # For now, just append a block containing tsub
    #         rem = [p for p in range(v) if p not in tsub]
    #         fill = list(itertools.islice(rem, k - t))
    #         new_block = sorted(list(tsub) + fill)
    #         blocks.append(new_block)
    
    return blocks
