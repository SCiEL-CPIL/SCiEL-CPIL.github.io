"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _precompute(v, k, t):
    """Precompute all C(v,k) blocks and their covered t-subsets."""
    all_blocks = list(itertools.combinations(range(v), k))
    # For each block, compute the list of t-subsets it covers (as sorted tuples)
    block_subsets = []
    for b in all_blocks:
        s = tuple(sorted(b))
        subs = [tuple(sorted(x)) for x in itertools.combinations(s, t)]
        block_subsets.append(subs)
    
    # Map from t-subset tuple -> list of block indices that cover it
    subset_to_blocks = {}
    for bi, subs in enumerate(block_subsets):
        for s in subs:
            if s not in subset_to_blocks:
                subset_to_blocks[s] = []
            subset_to_blocks[s].append(bi)
    
    return all_blocks, block_subsets, subset_to_blocks


def _greedy_cover(v, k, t, all_blocks, block_subsets, subset_to_blocks, 
                   uncovered_set, max_time=None):
    """Greedy set cover using precomputed data."""
    start = time.time()
    blocks_chosen = []
    uncovered = set(uncovered_set)
    
    while uncovered:
        if max_time is not None and time.time() - start > max_time:
            break
        
        # Pick a random uncovered subset
        # For determinism in tie-breaking, we could pick the min, but random is fine
        # Actually, for best results, pick the subset with fewest candidate blocks
        # (most constrained) - but that's expensive. Let's just pick random or min.
        # Let's pick the one with fewest candidate blocks to be smart
        min_count = len(all_blocks)
        min_sub = None
        # Sample a few to avoid full scan
        sample_size = min(len(uncovered), 200)
        if len(uncovered) <= sample_size:
            sample = list(uncovered)
        else:
            sample = random.sample(list(uncovered), sample_size)
        for s in sample:
            cnt = len(subset_to_blocks.get(s, []))
            if cnt < min_count:
                min_count = cnt
                min_sub = s
        if min_sub is None:
            min_sub = min(uncovered)
        
        # Get candidate blocks for this subset
        candidates = subset_to_blocks.get(min_sub, [])
        if not candidates:
            # Shouldn't happen if we have all blocks
            break
        
        # Find candidate with max gain
        best_block = -1
        best_gain = -1
        for bi in candidates:
            gain = sum(1 for s in block_subsets[bi] if s in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = bi
            if best_gain == len(uncovered):
                break
        
        if best_block >= 0:
            blocks_chosen.append(best_block)
            uncovered.difference_update(block_subsets[best_block])
    
    return blocks_chosen


def _deletion_pass(v, k, t, all_blocks, block_subsets, blocks_chosen, max_time=None):
    """Remove redundant blocks via a deletion pass."""
    if not blocks_chosen:
        return blocks_chosen
    
    start = time.time()
    # Count how many times each subset is covered
    subset_counts = {}
    for bi in blocks_chosen:
        for s in block_subsets[bi]:
            subset_counts[s] = subset_counts.get(s, 0) + 1
    
    # Try to remove blocks
    changed = True
    while changed:
        if max_time is not None and time.time() - start > max_time:
            break
        changed = False
        new_blocks = []
        for bi in blocks_chosen:
            # Check if this block is the sole cover for any subset
            is_needed = False
            for s in block_subsets[bi]:
                if subset_counts.get(s, 0) == 1:
                    is_needed = True
                    break
            if not is_needed:
                # Safe to remove
                for s in block_subsets[bi]:
                    subset_counts[s] -= 1
                changed = True
            else:
                new_blocks.append(bi)
        blocks_chosen = new_blocks
    
    return blocks_chosen


def _swap_improve(v, k, t, all_blocks, block_subsets, blocks_chosen, max_time=None):
    """Try to reduce block count by replacing subsets of blocks with smaller sets."""
    if len(blocks_chosen) <= 1:
        return blocks_chosen
    
    start = time.time()
    current_blocks = list(blocks_chosen)
    
    # Try removing one block and re-covering its exclusive subsets
    for i in range(len(current_blocks)):
        if max_time is not None and time.time() - start > max_time:
            break
        
        trial = current_blocks[:i] + current_blocks[i+1:]
        
        # Find subsets only covered by removed block
        subset_counts = {}
        for bi in trial:
            for s in block_subsets[bi]:
                subset_counts[s] = subset_counts.get(s, 0) + 1
        
        orphaned = []
        for s in block_subsets[current_blocks[i]]:
            if subset_counts.get(s, 0) == 0:
                orphaned.append(s)
        
        if not orphaned:
            # Block was redundant
            current_blocks = trial
            break
        
        # Try to cover orphaned subsets with one new block
        # Find block that covers most orphaned subsets
        best_bi = -1
        best_cover = 0
        # Only consider blocks that cover at least one orphaned subset
        candidate_bis = set()
        for s in orphaned:
            for bi in []:  # We need subset_to_blocks but don't have it here
                pass
        # This approach is tricky without subset_to_blocks. Let's skip for now.
        # Actually, let's just do the simple deletion pass.
        pass
    
    return current_blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Precompute all blocks and their covered subsets
    all_blocks, block_subsets, subset_to_blocks = _precompute(v, k, t)
    
    # All t-subsets
    all_subsets = set(itertools.combinations(range(v), t))
    
    # Run greedy cover with time limit
    max_greedy_time = min(60.0, deadline - (time.time() - start_time) - 20.0)
    blocks_chosen = _greedy_cover(v, k, t, all_blocks, block_subsets, 
                                   subset_to_blocks, all_subsets, 
                                   max_time=max_greedy_time)
    
    # Deletion pass to remove redundant blocks
    max_del_time = min(30.0, deadline - (time.time() - start_time) - 5.0)
    if max_del_time > 0:
        blocks_chosen = _deletion_pass(v, k, t, all_blocks, block_subsets, 
                                        blocks_chosen, max_time=max_del_time)
    
    # Verify coverage - if not complete, run another greedy pass for remaining
    if time.time() < deadline - 10:
        # Check what's still uncovered
        covered = set()
        for bi in blocks_chosen:
            covered.update(block_subsets[bi])
        uncovered = all_subsets - covered
        
        if uncovered and time.time() < deadline - 10:
            # Run another greedy pass for remaining uncovered
            max_greedy_time2 = min(30.0, deadline - (time.time() - start_time) - 5.0)
            extra_blocks = _greedy_cover(v, k, t, all_blocks, block_subsets,
                                          subset_to_blocks, uncovered,
                                          max_time=max_greedy_time2)
            blocks_chosen.extend(extra_blocks)
            
            # Another deletion pass
            if time.time() < deadline - 5:
                max_del_time2 = min(10.0, deadline - (time.time() - start_time) - 2.0)
                if max_del_time2 > 0:
                    blocks_chosen = _deletion_pass(v, k, t, all_blocks, block_subsets,
                                                    blocks_chosen, max_time=max_del_time2)
    
    # Convert block indices to actual blocks
    result = [list(all_blocks[bi]) for bi in blocks_chosen]
    
    # Ensure all blocks are sorted (they should be since all_blocks are sorted tuples)
    for b in result:
        b.sort()
    
    # Deduplicate just in case
    seen = set()
    unique_result = []
    for b in result:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_result.append(b)
    
    return unique_result
# EVOLVE-BLOCK-END
