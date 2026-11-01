"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # 130s to leave 20s margin

    # Total number of t-subsets
    from math import comb
    total_tsubsets = comb(v, t)

    # Map each t-subset to an integer index
    # We'll use a dictionary for mapping t-tuples to indices
    # But for speed, we'll use a different approach: 
    # Precompute all t-subsets and their indices
    
    # For efficiency, let's use a bitset approach with numpy
    # Each t-subset can be represented as a frozenset or sorted tuple
    
    # First, generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For each k-subset (block), precompute which t-subsets it covers
    # But there are C(v,k) possible blocks, which is too many.
    # Instead, we'll compute on-the-fly but cache.
    
    # Greedy construction
    uncovered = np.ones(total_tsubsets, dtype=bool)
    uncovered_count = total_tsubsets
    
    blocks = []
    
    def get_covered_by_block(block):
        """Get indices of t-subsets covered by a given block (sorted list of points)."""
        covered = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[tuple(ts)]
            covered.append(idx)
        return covered
    
    def gain_of_point(x, block):
        """Compute how many uncovered t-subsets would be covered by adding x to block."""
        count = 0
        for rest in itertools.combinations(block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            if ts in tsub_to_idx:
                idx = tsub_to_idx[ts]
                if uncovered[idx]:
                    count += 1
        return count
    
    # Greedy phase
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find the uncovered t-subset with smallest lex order to start a block
        # Actually, for better performance, pick the uncovered t-subset that appears in the most potential blocks
        # But for simplicity, pick the first uncovered one
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick the first uncovered t-subset
        first_uncovered_idx = uncovered_indices[0]
        first_uncovered_ts = all_tsubsets[first_uncovered_idx]
        
        # Start block with these t points
        block = list(first_uncovered_ts)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x not in block:
                    g = gain_of_point(x, block)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            
            if best_x == -1 or best_gain <= 0:
                # No point adds coverage, just add an arbitrary point
                for x in range(v):
                    if x not in block:
                        best_x = x
                        break
                if best_x == -1:
                    break
            
            block.append(best_x)
        
        block.sort()
        
        # Check if this block is already in our list
        block_tuple = tuple(block)
        if block_tuple not in [tuple(b) for b in blocks]:
            blocks.append(block)
            
            # Update uncovered
            covered = get_covered_by_block(block)
            for idx in covered:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
    
    if uncovered_count == 0:
        return blocks
    
    # Local search: try to remove redundant blocks and improve
    # Phase 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            
            # Temporarily remove block i
            saved_block = blocks[i]
            blocks.pop(i)
            
            # Check if all t-subsets are still covered
            # Build coverage map for remaining blocks
            covered_set = set()
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    covered_set.add(tuple(ts))
            
            # Check if all t-subsets are covered
            still_covered = len(covered_set) == total_tsubsets
            
            if still_covered:
                changed = True
            else:
                blocks.insert(i, saved_block)
    
    # Phase 2: Try to replace blocks with better ones
    # For each block, try removing one point and adding another to cover more
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                
                original_block = list(blocks[i])
                
                # Try swapping one point for another
                for pos in range(len(original_block)):
                    old_point = original_block[pos]
                    for new_point in range(v):
                        if new_point == old_point or new_point in original_block:
                            continue
                        
                        candidate = original_block.copy()
                        candidate[pos] = new_point
                        candidate.sort()
                        
                        # Check if this block covers at least one t-subset not covered by others
                        # And doesn't lose any t-subset that isn't covered by others
                        # This is complex, so let's just try it and see if we can still cover everything
                        
                        blocks[i] = candidate
                        
                        # Quick check: is the covering still valid?
                        # This is expensive, so only do it occasionally
                        if random.random() < 0.1:
                            covered_set = set()
                            for b in blocks:
                                for ts in itertools.combinations(b, t):
                                    covered_set.add(tuple(ts))
                            if len(covered_set) < total_tsubsets:
                                blocks[i] = original_block
                                continue
                        else:
                            # Assume it's fine, but we'll verify at the end
                            pass
                        
                        improved = True
                        break
                    
                    if improved:
                        break
    
    # Final verification and cleanup
    # Verify the covering is valid
    covered_set = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered_set.add(tuple(ts))
    
    if len(covered_set) < total_tsubsets:
        # Covering is invalid, fall back to greedy result
        # Re-run greedy from scratch
        uncovered = np.ones(total_tsubsets, dtype=bool)
        uncovered_count = total_tsubsets
        blocks = []
        
        while uncovered_count > 0 and time.time() < deadline:
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            first_uncovered_idx = uncovered_indices[0]
            first_uncovered_ts = all_tsubsets[first_uncovered_idx]
            
            block = list(first_uncovered_ts)
            
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x not in block:
                        g = gain_of_point(x, block)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                
                if best_x == -1:
                    for x in range(v):
                        if x not in block:
                            best_x = x
                            break
                    if best_x == -1:
                        break
                
                block.append(best_x)
            
            block.sort()
            block_tuple = tuple(block)
            if block_tuple not in [tuple(b) for b in blocks]:
                blocks.append(block)
                covered = get_covered_by_block(block)
                for idx in covered:
                    if uncovered[idx]:
                        uncovered[idx] = False
                        uncovered_count -= 1
    
    # Remove duplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        bt = tuple(b)
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(b)
    
    return unique_blocks
# EVOLVE-BLOCK-END
