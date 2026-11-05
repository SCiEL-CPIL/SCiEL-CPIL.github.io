"""Probabilistic method with alteration for covering design C(19,7,4)."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    rng = random.Random(42)
    
    # Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Coverage count: how many blocks cover each t-subset
    # We'll use a list of ints for speed
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    
    # Helper: add a block, update cover_count, return list of affected tsub indices
    def block_tsubs(block):
        return list(itertools.combinations(block, t))
    
    def add_block(block):
        affected = []
        for ts in block_tsubs(block):
            idx = tsub_to_idx[ts]
            cover_count[idx] += 1
            affected.append(idx)
        return affected
    
    def remove_block(block):
        affected = []
        for ts in block_tsubs(block):
            idx = tsub_to_idx[ts]
            cover_count[idx] -= 1
            affected.append(idx)
        return affected
    
    def is_redundant(block):
        for ts in block_tsubs(block):
            if cover_count[tsub_to_idx[ts]] <= 1:
                return False
        return True
    
    def uncovered_count():
        return int(np.sum(cover_count == 0))
    
    # Phase 1: Random initialization with bias
    # Start with a reasonable number of random blocks
    # C(19,7) = 50388 total possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Randomly select blocks, biased toward covering uncovered tsubsets
    blocks = []
    # First, add some random blocks to get initial coverage
    n_initial = 300
    for _ in range(n_initial):
        if time.time() > deadline:
            break
        b = tuple(rng.sample(all_blocks, 1)[0])
        if b not in blocks:
            blocks.append(b)
            add_block(b)
    
    # Phase 2: Greedy completion - add blocks to cover all uncovered tsubsets
    while uncovered_count() > 0:
        if time.time() > deadline:
            break
        # Find an uncovered tsubset
        uncov_indices = np.where(cover_count == 0)[0]
        if len(uncov_indices) == 0:
            break
        target_tsub = all_tsubs[uncov_indices[0]]
        
        # Find blocks that cover this tsubset and add the one that covers most other uncovered
        best_block = None
        best_gain = -1
        # Sample candidate blocks containing target_tsub
        remaining = [x for x in range(v) if x not in target_tsub]
        # Number of ways to complete: C(12, 3) = 220
        candidates = list(itertools.combinations(remaining, k - t))
        rng.shuffle(candidates)
        # Check up to 50 candidates
        for suffix in candidates[:50]:
            candidate = tuple(sorted(target_tsub + suffix))
            if candidate in blocks:
                continue
            gain = 0
            for ts in block_tsubs(candidate):
                if cover_count[tsub_to_idx[ts]] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = candidate
        if best_block is not None:
            blocks.append(best_block)
            add_block(best_block)
        else:
            # Fallback: just add any block containing target
            candidate = tuple(sorted(target_tsub + tuple(rng.sample(remaining, k - t))))
            blocks.append(candidate)
            add_block(candidate)
    
    # Phase 3: Redundancy removal pass
    def remove_redundant_pass():
        removed = 0
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to remove redundant blocks
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return removed
                if is_redundant(blocks[i]):
                    remove_block(blocks[i])
                    blocks.pop(i)
                    removed += 1
                    changed = True
        return removed
    
    remove_redundant_pass()
    
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    # Phase 4: Local search - try to replace blocks to reduce count
    # Strategy: try removing a block, then see if we can cover the newly uncovered
    # tsubsets with fewer blocks than removed
    search_start = time.time()
    search_budget = 100  # seconds for local search
    
    def try_replace(blocks_list, cover_cnt):
        """Try to reduce block count by removing one block and replacing its coverage."""
        if time.time() > deadline:
            return blocks_list, cover_cnt, False
        
        # Find a block that might be removable
        rng.shuffle(blocks_list)
        for i, b in enumerate(blocks_list):
            if time.time() > deadline:
                return blocks_list, cover_cnt, False
            
            # Temporarily remove block
            remove_block(b)
            newly_uncovered = []
            for ts in block_tsubs(b):
                idx = tsub_to_idx[ts]
                if cover_cnt[idx] == 0:
                    newly_uncovered.append(ts)
            
            if not newly_uncovered:
                # Block was redundant, remove it
                blocks_list.pop(i)
                return blocks_list, cover_cnt, True
            
            # Try to cover newly_uncovered with fewer than 1 block (impossible) 
            # or with 1 block that also helps elsewhere
            # Try to find a single block that covers all newly_uncovered
            if len(newly_uncovered) <= 1:
                target = newly_uncovered[0]
                remaining = [x for x in range(v) if x not in target]
                candidates = list(itertools.combinations(remaining, k - t))
                rng.shuffle(candidates)
                best_gain = -1
                best_candidate = None
                for suffix in candidates[:30]:
                    candidate = tuple(sorted(target + suffix))
                    gain = 0
                    for ts in block_tsubs(candidate):
                        if cover_cnt[tsub_to_idx[ts]] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = candidate
                if best_candidate is not None and best_candidate not in blocks_list:
                    blocks_list.append(best_candidate)
                    add_block(best_candidate)
                    return blocks_list, cover_cnt, False
                else:
                    # Restore
                    add_block(b)
            else:
                # Too many newly uncovered, restore
                add_block(b)
        
        return blocks_list, cover_cnt, False
    
    # Iterative local search
    for iteration in range(200):
        if time.time() > deadline:
            break
        if time.time() - search_start > search_budget:
            break
        
        improved = False
        for _ in range(5):
            if time.time() > deadline:
                break
            blocks, cover_count, improved = try_replace(blocks, cover_count)
            if improved:
                best_len = len(blocks)
                best_blocks = list(blocks)
                break
            if time.time() - search_start > search_budget:
                break
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Final redundancy removal
    if time.time() < deadline:
        remove_redundant_pass()
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Verify coverage
    final_cover = np.zeros(n_tsubs, dtype=np.int32)
    for b in best_blocks:
        for ts in block_tsubs(b):
            final_cover[tsub_to_idx[ts]] += 1
    
    if np.any(final_cover == 0):
        # Coverage not complete, fall back to greedy completion
        for i in range(n_tsubs):
            if final_cover[i] == 0:
                target = all_tsubs[i]
                remaining = [x for x in range(v) if x not in target]
                suffix = tuple(rng.sample(remaining, k - t))
                new_block = tuple(sorted(target + suffix))
                if new_block not in best_blocks:
                    best_blocks.append(new_block)
                    for ts in block_tsubs(new_block):
                        final_cover[tsub_to_idx[ts]] += 1
    
    # Convert to lists and return
    return [list(b) for b in best_blocks]
