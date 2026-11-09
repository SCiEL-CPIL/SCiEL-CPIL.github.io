import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for safety
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_sub_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Precompute all k-subsets and their covered t-subsets
    # For memory efficiency, we store the list of t-subset indices covered by each k-subset
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Map each k-subset to the indices of the t-subsets it covers
    # This is a list of lists: block_t_indices[block_idx] = [t_idx1, t_idx2, ...]
    block_t_indices = []
    for block in k_subsets:
        covered_t = []
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_idx.get(t_sub)
            if idx is not None:
                covered_t.append(idx)
        block_t_indices.append(covered_t)
        
    # Greedy Set Cover
    # coverage_counts[i] is the number of blocks covering t-subset i
    coverage_counts = [0] * n_t
    blocks = [] # List of k-subsets (tuples)
    block_indices = [] # Indices into k_subsets for quick removal
    
    # To speed up greedy, we can precompute for each t-subset, which blocks cover it?
    # That would be a list of lists of length n_t, each containing block indices.
    # This allows us to quickly find candidates when a t-subset becomes uncovered.
    # However, standard greedy picks the block with max gain.
    # Gain of a block = number of t-subsets in it with coverage_counts == 0.
    
    # Efficient Greedy:
    # We maintain a set of uncovered t-subsets.
    # Initially all are uncovered.
    # We pick the block that covers the most uncovered t-subsets.
    
    uncovered = set(range(n_t))
    
    # To make this fast, we can't iterate all n_k blocks every time if n_k is large.
    # But n_k = 75582, which is small enough.
    # We can iterate all blocks, calculate gain, and pick the max.
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        max_gain = -1
        
        # Optimization: We can skip blocks that are already selected
        # But checking if a block is in `blocks` is O(n_blocks).
        # Better: maintain a boolean array `is_selected` of size n_k.
        
        # For simplicity and speed in Python, let's just iterate.
        # If n_k is ~75k, and we do this ~300 times, that's 22.5 million iterations.
        # Inside, we check gain.
        
        # A faster way:
        # For each uncovered t-subset, look at the blocks that cover it.
        # Increment a "score" for those blocks.
        # The block with the highest score is the best.
        
        # Let's build the inverse mapping: t_sub -> list of block indices
        # This takes memory but speeds up the loop significantly.
        # t_to_blocks = [[] for _ in range(n_t)]
        # for b_idx, t_list in enumerate(block_t_indices):
        #     for t_idx in t_list:
        #         t_to_blocks[t_idx].append(b_idx)
        
        # We only build this once.
        if not hasattr(construct, '_t_to_blocks'):
            construct._t_to_blocks = [[] for _ in range(n_t)]
            for b_idx, t_list in enumerate(block_t_indices):
                for t_idx in t_list:
                    construct._t_to_blocks[t_idx].append(b_idx)
        
        t_to_blocks = construct._t_to_blocks
        
        # Reset scores for all blocks? No, we can use a dictionary or a list with timestamps.
        # Since n_k is 75k, a list of ints is fine.
        scores = [0] * n_k
        # To avoid resetting the whole list every iteration (O(n_k)), we can use a trick
        # or just reset it if n_k is small enough. 75k is small.
        
        # For each uncovered t-subset, add 1 to the score of all blocks covering it
        for t_idx in uncovered:
            for b_idx in t_to_blocks[t_idx]:
                scores[b_idx] += 1
        
        # Find the block with max score
        # We must ensure we don't pick a block already in the covering.
        # We can maintain a set of selected block indices.
        selected_blocks_set = set(block_indices)
        
        max_score = -1
        best_b_idx = -1
        for b_idx in range(n_k):
            if b_idx not in selected_blocks_set:
                if scores[b_idx] > max_score:
                    max_score = scores[b_idx]
                    best_b_idx = b_idx
        
        if best_b_idx == -1:
            break # Should not happen if uncovered is not empty
            
        # Select the block
        blocks.append(k_subsets[best_b_idx])
        block_indices.append(best_b_idx)
        selected_blocks_set.add(best_b_idx)
        
        # Update coverage counts and uncovered set
        for t_idx in block_t_indices[best_b_idx]:
            coverage_counts[t_idx] += 1
            if coverage_counts[t_idx] == 1:
                uncovered.discard(t_idx)
                
    # Local Search: Remove redundant blocks
    # A block is redundant if removing it does not make any t-subset uncovered.
    # i.e., all t-subsets in the block have coverage_counts > 1.
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # Iterate backwards to allow safe removal
        for i in range(len(blocks) - 1, -1, -1):
            b_idx = block_indices[i]
            redundant = True
            for t_idx in block_t_indices[b_idx]:
                if coverage_counts[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block
                for t_idx in block_t_indices[b_idx]:
                    coverage_counts[t_idx] -= 1
                blocks.pop(i)
                block_indices.pop(i)
                changed = True
                if time.time() > deadline:
                    break

    # Convert blocks to lists of integers as required
    final_blocks = [list(block) for block in blocks]
    return final_blocks

# Helper to clear cache if needed, though not strictly necessary for single call
# But good practice if construct is called multiple times with different params
# The prompt implies single call or consistent params.
# The cache `_t_to_blocks` depends on v, k, t.
# We should clear it or key it properly. Since the signature is fixed, 
# and the task is specific, we assume v,k,t are constant during evaluation.
# To be safe, we can store the parameters used for the cache.

_original_construct = construct

def construct(v, k, t):
    # Check if cache is valid for current v, k, t
    if not hasattr(construct, '_cache_params') or construct._cache_params != (v, k, t):
        if hasattr(construct, '_t_to_blocks'):
            del construct._t_to_blocks
        construct._cache_params = (v, k, t)
        # Re-initialize _t_to_blocks inside the function logic
        # The logic inside _original_construct handles the creation if missing.
        # But we deleted it, so it will be recreated.
        
    return _original_construct(v, k, t)
