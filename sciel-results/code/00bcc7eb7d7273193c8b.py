"""Covering design C(24,7,3) via accelerated greedy with local search polish."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) minimizing the number of blocks.
    Uses accelerated greedy construction followed by local search (block removal
    and block replacement) to reduce block count.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # All t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # For each possible block (k-subset), precompute which t-subsets it covers
    # This is expensive for large v, but for v=24, k=7: C(24,7) = 346104 blocks
    # Each covers C(7,3) = 35 t-subsets
    # Total: 346104 * 35 = ~12M entries - manageable with bitsets
    
    # Use numpy bitsets for coverage tracking
    # n_t = C(24,3) = 2024, so we can use a single uint64 bitset (2024 < 64*32)
    # Actually 2024 bits needs 32 uint64s or we can use Python integers as bitsets
    
    # Precompute for each k-subset the bitset of t-subsets it covers
    # For v=24, k=7: 346104 k-subsets, each with 35 t-subsets
    # We'll compute on the fly to save memory, but cache the t-subset lists
    
    # Precompute t-subsets for each k-subset
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # For each k-subset, store the list of t-subset indices it covers
    block_covers = []
    for ks in all_k_subsets:
        t_indices = [t_sub_index[ts] for ts in itertools.combinations(ks, t)]
        block_covers.append(t_indices)
    
    # Convert to a numpy array for fast operations (ragged, so use list of arrays)
    # Actually, let's use a different approach: for greedy, we just need to know
    # which uncovered t-subsets a candidate point would cover.
    
    # Greedy construction with look-ahead
    uncovered = set(range(n_t))  # indices of uncovered t-subsets
    blocks = []
    
    # For speed, track uncovered as a set of indices
    # For each point x, precompute which t-subsets contain x
    point_t_subsets = {}
    for x in range(v):
        point_t_subsets[x] = set()
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            point_t_subsets[x].add(i)
    
    # Greedy: repeatedly pick the block that covers the most uncovered t-subsets
    # To speed up, use a heuristic: start from an uncovered t-subset, extend greedily
    
    while uncovered and time.time() < deadline - 30:
        # Pick the uncovered t-subset with the "rarest" elements (fewest blocks that could cover it)
        # For simplicity, just pick the first uncovered one, but try a few candidates
        min_uncovered = min(uncovered)
        base_t = all_t_subsets[min_uncovered]
        
        # Greedily build a block containing base_t
        block = set(base_t)
        
        while len(block) < k and time.time() < deadline - 30:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset is covered if it's a subset of block ∪ {x}
                # i.e., it contains x and (t-1) elements from block
                gain = 0
                # t-subsets containing x: need t-1 more from block
                for rest in itertools.combinations(sorted(block), t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in t_sub_index:
                        idx = t_sub_index[ts]
                        if idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.add(best_x)
        
        # Convert to sorted tuple
        block_sorted = tuple(sorted(block))
        
        # Add to blocks and update uncovered
        blocks.append(list(block_sorted))
        for ts in itertools.combinations(block_sorted, t):
            idx = t_sub_index[ts]
            uncovered.discard(idx)
    
    # If we still have uncovered, force-cover them (shouldn't happen if loop completes)
    while uncovered:
        min_uncovered = min(uncovered)
        base_t = all_t_subsets[min_uncovered]
        block = list(base_t)
        # Fill with random points
        remaining = [x for x in range(v) if x not in block]
        random.shuffle(remaining)
        for x in remaining:
            if len(block) >= k:
                break
            block.append(x)
        block.sort()
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            idx = t_sub_index[ts]
            uncovered.discard(idx)
    
    # Local search: try to remove blocks
    # A block can be removed if every t-subset it covers is also covered by another block
    
    # Build coverage count: for each t-subset, how many blocks cover it
    # And for each block, which t-subsets it covers
    
    def build_coverage(blocks):
        """Return (cov_count, block_t_indices) where cov_count[i] = number of blocks covering t-subset i"""
        cov_count = np.zeros(n_t, dtype=np.int32)
        block_t_indices = []
        for b in blocks:
            t_idxs = [t_sub_index[ts] for ts in itertools.combinations(b, t)]
            block_t_indices.append(t_idxs)
            for idx in t_idxs:
                cov_count[idx] += 1
        return cov_count, block_t_indices
    
    def try_remove_blocks(blocks, max_iter=100000):
        """Try to remove redundant blocks. Returns modified blocks list."""
        cov_count, block_t_indices = build_coverage(blocks)
        removed = [False] * len(blocks)
        any_removed = True
        iterations = 0
        while any_removed and iterations < max_iter:
            any_removed = False
            iterations += 1
            for i in range(len(blocks)):
                if removed[i]:
                    continue
                # Check if all t-subsets in block i have cov_count > 1
                removable = True
                for idx in block_t_indices[i]:
                    if cov_count[idx] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove block i
                    removed[i] = True
                    any_removed = True
                    for idx in block_t_indices[i]:
                        cov_count[idx] -= 1
                    # Rebuild for remaining blocks
                    break
            if any_removed:
                # Rebuild coverage for remaining blocks
                new_blocks = [blocks[i] for i in range(len(blocks)) if not removed[i]]
                cov_count, block_t_indices = build_coverage(new_blocks)
                blocks = new_blocks
                removed = [False] * len(blocks)
        return blocks
    
    # Apply block removal
    if time.time() < deadline - 10:
        blocks = try_remove_blocks(blocks)
    
    # Local search: try block replacement (swap one block for another that covers more)
    # This is more complex; for now, do a simple hill-climbing step
    
    # Try to find a better solution by randomly perturbing and re-optimizing
    best_blocks = blocks
    best_count = len(blocks)
    
    # Simple local search: try removing a random block and see if we can fix the gaps
    # by adding a better block
    if time.time() < deadline - 5:
        for _ in range(200):  # limited iterations
            if time.time() > deadline - 5:
                break
            trial = list(best_blocks)
            if len(trial) <= 1:
                break
            # Pick a random block to remove
            idx = random.randint(0, len(trial) - 1)
            removed_block = trial.pop(idx)
            
            # Find uncovered t-subsets
            cov_count = np.zeros(n_t, dtype=np.int32)
            for b in trial:
                for ts in itertools.combinations(b, t):
                    cov_count[t_sub_index[ts]] += 1
            
            uncovered_now = [i for i in range(n_t) if cov_count[i] == 0]
            if not uncovered_now:
                # We removed a redundant block!
                best_blocks = trial
                best_count = len(trial)
                continue
            
            # Try to cover all uncovered t-subsets with one new block
            # Find a k-subset that covers the most uncovered t-subsets
            best_new_block = None
            best_new_gain = 0
            # Sample some candidate blocks
            candidates = random.sample(all_k_subsets, min(500, n_k))
            for ks in candidates:
                gain = 0
                for ts in itertools.combinations(ks, t):
                    idx = t_sub_index[ts]
                    if cov_count[idx] == 0:
                        gain += 1
                if gain > best_new_gain:
                    best_new_gain = gain
                    best_new_block = list(ks)
            
            # Check if the new block covers all uncovered
            if best_new_block:
                still_uncovered = []
                for idx in uncovered_now:
                    ts = all_t_subsets[idx]
                    if all(x in best_new_block for x in ts):
                        continue
                    still_uncovered.append(idx)
                
                if not still_uncovered:
                    # The new block covers all previously uncovered t-subsets
                    trial.append(best_new_block)
                    if len(trial) < best_count:
                        best_blocks = trial
                        best_count = len(trial)
    
    return best_blocks
# EVOLVE-BLOCK-END
