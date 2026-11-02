"""Covering design C(19,7,4) via bitmask-vectorized greedy + local search."""
import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 135  # safety margin before 150s kill
    best_blocks = None
    best_count = float('inf')

    # Precompute all t-subsets and their bitmask representation
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    tsub_masks = [0] * n_tsubs
    for i, ts in enumerate(all_tsubs):
        m = 0
        for x in ts:
            m |= (1 << x)
        tsub_masks[i] = m

    # Precompute all k-subsets of v points, sorted by index
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, precompute the bitmask of which t-subsets it covers
    # Represent as a set of t-subset indices for fast intersection
    ksub_tsub_sets = []
    for ks in all_ksubs:
        covered = set()
        for ts in itertools.combinations(ks, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                covered.add(idx)
        ksub_tsub_sets.append(covered)

    def greedy_build(seed, randomize=True):
        """Bitmask-vectorized greedy: pick k-subset with max uncovered coverage."""
        rng = random.Random(seed)
        uncovered = set(range(n_tsubs))
        blocks = []
        
        # Precompute for each k-subset its t-subset set (already done)
        # We'll track uncovered as a set for fast intersection
        
        while uncovered:
            best_gain = -1
            best_ks_idx = -1
            best_ks = None
            
            # Candidate order: randomize if needed
            if randomize:
                candidate_order = list(range(n_ksubs))
                rng.shuffle(candidate_order)
            else:
                candidate_order = list(range(n_ksubs))
            
            for ks_idx in candidate_order:
                covered_set = ksub_tsub_sets[ks_idx]
                # Gain = number of t-subsets in covered_set that are still uncovered
                gain = len(covered_set & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_ks_idx = ks_idx
                    best_ks = all_ksubs[ks_idx]
                    if best_gain == len(uncovered):
                        # Perfect: covers all remaining
                        break
                elif gain == best_gain and gain > 0:
                    # Tiebreak: random
                    if rng.random() < 0.3:
                        best_gain = gain
                        best_ks_idx = ks_idx
                        best_ks = all_ksubs[ks_idx]
            
            if best_ks is None or best_gain <= 0:
                # Fallback: pick any uncovered t-subset and build a block around it
                first_uncovered = next(iter(uncovered))
                base_ts = all_tsubs[first_uncovered]
                # Build a k-subset containing base_ts
                remaining = [x for x in range(v) if x not in base_ts]
                if randomize:
                    rng.shuffle(remaining)
                block = list(base_ts) + remaining[:k-t]
                block.sort()
                best_ks = tuple(block)
                best_ks_idx = all_ksubs.index(best_ks) if best_ks in all_ksubs else -1
                # Update uncovered
                for ts in itertools.combinations(block, t):
                    idx = tsub_index.get(ts)
                    if idx is not None:
                        uncovered.discard(idx)
                blocks.append(list(block))
                continue
            
            blocks.append(list(best_ks))
            # Update uncovered
            covered_set = ksub_tsub_sets[best_ks_idx]
            uncovered -= covered_set
        
        return blocks

    def local_search_optimize(blocks, deadline):
        """Try to remove redundant blocks and swap blocks for better ones."""
        if not blocks:
            return blocks
        
        blocks_list = list(blocks)
        
        # Build coverage count for each t-subset
        cov_count = [0] * n_tsubs
        for block in blocks_list:
            for ts in itertools.combinations(block, t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    cov_count[idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            n_blocks = len(blocks_list)
            
            # Phase 1: Remove redundant blocks
            for i in range(n_blocks):
                if time.time() > deadline - 10:
                    break
                
                block_i = blocks_list[i]
                # Check if removing block_i leaves all t-subsets covered
                removable = True
                for ts in itertools.combinations(block_i, t):
                    idx = tsub_index.get(ts)
                    if idx is not None and cov_count[idx] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove block i
                    for ts in itertools.combinations(block_i, t):
                        idx = tsub_index.get(ts)
                        if idx is not None:
                            cov_count[idx] -= 1
                    blocks_list.pop(i)
                    improved = True
                    break
            
            # Phase 2: Try replacing a block with a better one
            if not improved and time.time() < deadline - 15:
                for i in range(len(blocks_list)):
                    if time.time() > deadline - 15:
                        break
                    
                    # Find t-subsets only covered by block i
                    unique_to_i = []
                    for ts in itertools.combinations(blocks_list[i], t):
                        idx = tsub_index.get(ts)
                        if idx is not None and cov_count[idx] == 1:
                            unique_to_i.append(idx)
                    
                    if not unique_to_i:
                        continue
                    
                    # Try to find a replacement block that covers all unique_to_i
                    # and maximizes new coverage
                    best_replace = None
                    best_new_gain = -1
                    
                    # Sample candidate replacements (not all n_ksubs for speed)
                    n_candidates = min(200, n_ksubs)
                    candidate_indices = random.sample(range(n_ksubs), n_candidates)
                    
                    for ks_idx in candidate_indices:
                        covered_set = ksub_tsub_sets[ks_idx]
                        # Must cover all unique_to_i
                        if not all(idx in covered_set for idx in unique_to_i):
                            continue
                        
                        # Gain = number of currently uncovered t-subsets this new block covers
                        # that weren't covered before (i.e., not in block_i's coverage)
                        old_covered = ksub_tsub_sets[all_ksubs.index(blocks_list[i])] if blocks_list[i] in all_ksubs else set()
                        new_only = covered_set - old_covered
                        gain = len(new_only)
                        
                        if gain > best_new_gain:
                            best_new_gain = gain
                            best_replace = ks_idx
                    
                    if best_replace is not None and best_new_gain > 0:
                        # Replace block i
                        old_block = blocks_list[i]
                        new_block = list(all_ksubs[best_replace])
                        
                        # Update coverage
                        for ts in itertools.combinations(old_block, t):
                            idx = tsub_index.get(ts)
                            if idx is not None:
                                cov_count[idx] -= 1
                        
                        for ts in itertools.combinations(new_block, t):
                            idx = tsub_index.get(ts)
                            if idx is not None:
                                cov_count[idx] += 1
                        
                        blocks_list[i] = new_block
                        improved = True
                        break
        
        return blocks_list

    # Main search loop
    # First: deterministic greedy baseline
    blocks = greedy_build(0, randomize=False)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Then: randomized restarts
    for seed in range(1, 1000):
        if time.time() > deadline - 30:
            break
        
        blocks = greedy_build(seed, randomize=True)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        
        if seed % 20 == 0 and time.time() > deadline - 25:
            break
    
    # Local search on best solution
    if best_blocks and time.time() < deadline - 20:
        best_blocks = local_search_optimize(best_blocks, deadline)
        best_count = len(best_blocks)
    
    # Try local search on a few other good solutions if time permits
    if time.time() < deadline - 25:
        for seed in range(1, 50):
            if time.time() > deadline - 15:
                break
            blocks = greedy_build(seed, randomize=True)
            if len(blocks) <= best_count + 2:
                blocks = local_search_optimize(blocks, deadline)
                if len(blocks) < best_count:
                    best_blocks = blocks
                    best_count = len(blocks)
    
    return best_blocks if best_blocks else greedy_build(0, randomize=False)
