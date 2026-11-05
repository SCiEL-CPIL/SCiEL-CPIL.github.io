import itertools
import random

def construct(v, k, t):
    points = list(range(v))
    
    # Map t-subsets to indices
    t_subs = list(itertools.combinations(points, t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Initialize uncovered mask with all bits set
    # Using Python integers as bitsets
    uncovered_mask = (1 << num_t_subs) - 1
    
    # Pre-compute all k-blocks and their bitmasks
    # A block covers a t-subset if the t-subset is a subset of the block
    blocks = list(itertools.combinations(points, k))
    block_masks = []
    
    for block in blocks:
        mask = 0
        # For each t-subset of the block, set the corresponding bit
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Greedy Set Cover with Randomization
    # We pick the best block from the top candidates to avoid getting stuck in bad local optima
    result_blocks = []
    
    # To ensure reproducibility in a single run but variation between runs if needed,
    # we can use a fixed seed or just standard random. The evaluator runs the function.
    # Let's use a deterministic tie-breaking first, or random if we want to explore.
    # The directive suggests "Weighted Local Search". Randomization in the top candidates
    # is a form of stochastic local search.
    
    while uncovered_mask:
        best_gain = -1
        best_indices = []
        
        # Calculate gain for all blocks
        # Optimization: Only consider blocks that have some gain?
        # With 50k blocks, iterating all is fast enough in Python with bit ops.
        for i, b_mask in enumerate(block_masks):
            gain = (uncovered_mask & b_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_indices = [i]
            elif gain == best_gain:
                best_indices.append(i)
        
        if best_gain == 0:
            # This shouldn't happen if the problem is solvable, but just in case
            break
            
        # Select a block from the top candidates
        # To improve over pure greedy, we can pick randomly from the top K
        # K=1 is pure greedy. K=5 or 10 adds stochasticity.
        # Let's pick from the top 5 (or all if fewer)
        top_k = 5
        candidates = best_indices[:top_k]
        
        # Weighted selection? The directive mentions "product of marginal gain and inverse degree".
        # Since degree is constant (35), it's just gain.
        # Let's just pick randomly from the top candidates to diversify.
        chosen_idx = random.choice(candidates)
        
        chosen_block = blocks[chosen_idx]
        result_blocks.append(list(chosen_block))
        
        # Update uncovered mask
        uncovered_mask &= ~block_masks[chosen_idx]
        
    return result_blocks
