import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    # Precompute all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_index = {s: i for i, s in enumerate(all_t_subs)}
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # For each k-subset, compute which t-subsets it covers (as a set of indices)
    # This is expensive but we do it once. Use numpy for speed.
    # Coverage matrix: n_k x n_t boolean (too big for full matrix if n_k is large)
    # Instead, for each k-subset, compute the set of t-subset indices it covers.
    
    # For v=19, k=7, t=4: n_k = C(19,7) = 50388, n_t = C(19,4) = 3876
    # Coverage per k-subset: C(7,4) = 35 t-subsets
    
    # Precompute coverage for each k-subset as a frozenset of t-subset indices
    # This is memory heavy: 50388 * 35 ~ 1.7M entries, manageable
    
    # Actually, let's use a different approach: represent coverage as a bitmask
    # n_t = 3876 bits, so we need ~60 64-bit words or use Python integers (arbitrary precision)
    
    # Use Python integers as bitsets (they're arbitrary precision and fast for bitwise ops)
    # coverage_mask[k_idx] = bitmask of t-subsets covered by k-subset k_idx
    
    coverage_masks = []
    for ks in all_k_subs:
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_sub_index[ts]
            mask |= (1 << idx)
        coverage_masks.append(mask)
    
    # Initial uncovered: all t-subsets
    uncovered = (1 << n_t) - 1
    blocks = []
    used_k_indices = set()
    
    # Greedy set cover: repeatedly pick the k-subset that covers the most uncovered t-subsets
    # To speed up, we can use a heuristic: try random restarts with different orderings
    
    def greedy_cover(uncovered_mask, existing_blocks, deadline):
        """Greedy: add blocks one at a time, each covering max uncovered t-subsets."""
        result = list(existing_blocks)
        unc = uncovered_mask
        used = set()
        for b in existing_blocks:
            used.add(b)
        
        while unc != 0 and time.time() < deadline:
            best_gain = 0
            best_ks = None
            # Sample candidate k-subsets for speed
            # If n_k is manageable, check all; otherwise sample
            if n_k <= 60000:
                candidates = range(n_k)
            else:
                candidates = random.sample(range(n_k), min(20000, n_k))
            
            for ki in candidates:
                if ki in used:
                    continue
                gain = bin(coverage_masks[ki] & unc).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_ks = ki
            
            if best_ks is None or best_gain == 0:
                break
            
            used.add(best_ks)
            result.append(all_k_subs[best_ks])
            unc &= ~coverage_masks[best_ks]
        
        return result, unc
    
    # Run greedy
    blocks, remaining_uncovered = greedy_cover(uncovered, [], deadline)
    
    if time.time() >= deadline:
        return [list(b) for b in blocks]
    
    # Local search: try to improve by swapping points in blocks
    # Strategy: for each block, try replacing one point with another to cover more uncovered t-subsets
    # or to allow removal of another block
    
    def compute_uncovered_mask(block_list):
        """Compute which t-subsets are NOT covered by the given blocks."""
        covered = 0
        for b in block_list:
            for ts in itertools.combinations(b, t):
                idx = t_sub_index[ts]
                covered |= (1 << idx)
        return ((1 << n_t) - 1) & ~covered
    
    # Local search phase
    def local_search(blocks, deadline):
        """Try to improve the covering by local moves."""
        blocks = [list(b) for b in blocks]
        best_blocks = blocks
        best_count = len(blocks)
        
        # Compute coverage for each block
        def block_coverage(b):
            mask = 0
            for ts in itertools.combinations(b, t):
                idx = t_sub_index[ts]
                mask |= (1 << idx)
            return mask
        
        # Try: for each pair of blocks, see if we can merge/replace
        # Simpler: try adding a point to a block (if size < k) - no, blocks are fixed size k
        # Try: swap a point in one block with a point in another
        # Try: remove a block if its t-subsets are all covered by others
        
        # First pass: remove redundant blocks
        for _ in range(3):  # multiple passes
            if time.time() >= deadline:
                break
            new_blocks = []
            for i, b in enumerate(blocks):
                # Check if b's t-subsets are all covered by other blocks
                b_cov = block_coverage(b)
                other_cov = 0
                for j, b2 in enumerate(blocks):
                    if j == i:
                        continue
                    other_cov |= block_coverage(b2)
                if (b_cov & ~other_cov) == 0:
                    # b is redundant
                    continue
                new_blocks.append(b)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                best_blocks = list(blocks)
                best_count = len(blocks)
        
        # Second pass: try swapping points between blocks
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(blocks)):
                    if time.time() >= deadline:
                        break
                    for j in range(i+1, len(blocks)):
                        if time.time() >= deadline:
                            break
                        # Try swapping one point from block i with one from block j
                        bi = blocks[i]
                        bj = blocks[j]
                        # Try each point in bi not in bj, and each point in bj not in bi
                        for pi in range(len(bi)):
                            if time.time() >= deadline:
                                break
                            for pj in range(len(bj)):
                                if time.time() >= deadline:
                                    break
                                x = bi[pi]
                                y = bj[pj]
                                if x == y:
                                    continue
                                # Create new blocks
                                new_bi = list(bi)
                                new_bi[pi] = y
                                new_bj = list(bj)
                                new_bj[pj] = x
                                new_bi.sort()
                                new_bj.sort()
                                # Check validity: no duplicates within a block
                                if len(set(new_bi)) < k or len(set(new_bj)) < k:
                                    continue
                                # Check if this swap improves coverage
                                old_cov = block_coverage(bi) | block_coverage(bj)
                                new_cov = block_coverage(tuple(new_bi)) | block_coverage(tuple(new_bj))
                                if new_cov != old_cov:
                                    # Check if new coverage is a superset or at least not worse
                                    if (new_cov & ~old_cov) != 0 and (old_cov & ~new_cov) == 0:
                                        blocks[i] = new_bi
                                        blocks[j] = new_bj
                                        improved = True
                                        best_blocks = [list(b) for b in blocks]
                                        break
                                # Also check if we can reduce total uncovered
                                # Actually, let's just try and check global uncovered
                        if improved:
                            break
                    if improved:
                        break
        
        return blocks
    
    if time.time() < deadline:
        blocks = local_search(blocks, deadline)
    
    # Final check: ensure all t-subsets are covered
    # If not, add more blocks greedily
    if time.time() < deadline:
        final_unc = compute_uncovered_mask(blocks)
        while final_unc != 0 and time.time() < deadline:
            best_gain = 0
            best_ks = None
            for ki in range(n_k):
                gain = bin(coverage_masks[ki] & final_unc).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_ks = ki
            if best_ks is None or best_gain == 0:
                break
            blocks.append(list(all_k_subs[best_ks]))
            final_unc &= ~coverage_masks[best_ks]
    
    # Deduplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        t = tuple(sorted(b))
        if t not in seen:
            seen.add(t)
            unique_blocks.append(list(t))
    
    return unique_blocks
