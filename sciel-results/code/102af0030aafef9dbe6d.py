"""Covering C(19,8,5) via maximum-coverage pruning from all blocks."""
import numpy as np
import itertools

def construct(v, k, t):
    # We will work with the complement of coverage: start with all C(v,k) blocks
    # and iteratively remove any block that is not essential (i.e., every t-subset
    # it covers is also covered by at least one other remaining block).
    #
    # For v=19, k=8, t=5: C(19,8)=75582 blocks, C(19,5)=11628 t-subsets.
    # We maintain a coverage count array: for each t-subset, how many remaining
    # blocks contain it. A block is removable iff for every t-subset S it contains,
    # count[S] >= 2.
    
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Precompute all k-blocks and which t-subsets they cover
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # For each block, store the list of t-subset indices it covers
    block_t_list = []
    for block in k_blocks:
        covered = [t_index[ts] for ts in itertools.combinations(block, t)]
        block_t_list.append(covered)
    
    # Coverage counts: how many remaining blocks cover each t-subset
    coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        for idx in covered:
            coverage[idx] += 1
    
    # Active flags for each block
    active = np.ones(n_k, dtype=bool)
    
    # Iteratively remove non-essential blocks
    # We use a queue-based approach: when we remove a block, some t-subsets'
    # counts drop, which may make other blocks non-essential.
    # But actually, the simpler approach: repeatedly scan all active blocks and
    # remove any that are non-essential. Since removal only decreases counts,
    # a block that was essential may become non-essential later.
    
    # More efficient: use a worklist. When we remove block b, for each t-subset s
    # in b, coverage[s] decreases by 1. If coverage[s] becomes 1, then the remaining
    # block covering s must be kept. But other blocks that also covered s can still
    # potentially be removed if all their other t-subsets have count >= 2.
    
    # Let's use a simpler iterative approach with NumPy vectorization where possible.
    # For each active block, check if all its t-subsets have coverage >= 2.
    
    # Precompute for each block the array of t-subset indices
    # We'll store as a list of arrays for simplicity
    
    # Since n_k can be large, let's do this in batches or with a loop.
    # For v=19, k=8, t=5: each block has C(8,5)=56 t-subsets.
    
    # Iterative removal:
    changed = True
    while changed:
        changed = False
        for i in range(n_k):
            if not active[i]:
                continue
            # Check if block i is removable: all its t-subsets have coverage >= 2
            covered = block_t_list[i]
            if all(coverage[idx] >= 2 for idx in covered):
                # Remove this block
                active[i] = False
                changed = True
                for idx in covered:
                    coverage[idx] -= 1
    
    # Collect remaining blocks
    result = [list(k_blocks[i]) for i in range(n_k) if active[i]]
    
    # Verify coverage (optional, for safety)
    # Every t-subset must be covered by at least one block
    if n_t > 0:
        if np.any(coverage < 1):
            # This shouldn't happen if our logic is correct, but as a safeguard:
            # Fall back to greedy on uncovered t-subsets
            uncovered = set()
            for idx in range(n_t):
                if coverage[idx] < 1:
                    uncovered.add(t_subsets[idx])
            
            while uncovered:
                # Greedy: find block covering most uncovered
                best_block = None
                best_gain = -1
                for i in range(n_k):
                    if active[i]:
                        covered = block_t_list[i]
                        gain = sum(1 for idx in covered if t_subsets[idx] in uncovered)
                    else:
                        covered = block_t_list[i]
                        gain = sum(1 for idx in covered if t_subsets[idx] in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                if best_block is None:
                    break
                active[best_block] = True
                result.append(list(k_blocks[best_block]))
                for idx in block_t_list[best_block]:
                    s = t_subsets[idx]
                    if s in uncovered:
                        uncovered.discard(s)
                    coverage[idx] += 1
            result.sort()
    
    return result
