import itertools
import random

# Precomputation for the specific case v=19, k=7, t=4
# This allows for very fast greedy and local search steps.

def _precompute_covering_data(v, k, t):
    """
    Precomputes bitmasks for blocks and lists of t-subsets for each block.
    Returns:
    - t_subsets: List of all t-subsets (as tuples)
    - t_to_idx: Dict mapping t-subset tuple to index
    - block_indices: List of all k-block indices (combinations of range(v) of size k)
    - block_masks: List of bitmasks (int) where bit j is set if t_subsets[j] is in the block
    - block_tsub_indices: List of lists of indices of t-subsets contained in the block
    """
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    block_combinations = list(itertools.combinations(range(v), k))
    num_blocks = len(block_combinations)
    
    block_masks = [0] * num_blocks
    block_tsub_indices = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(block_combinations):
        # Find all t-subsets within this block
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_tsub_indices[i] = indices
        
    return t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t

# Global cache for precomputation to avoid re-doing it if called multiple times
# Though construct is called once, it's good practice.
_PRECOMP_CACHE = {}

def construct(v, k, t):
    # Check if we have precomputed data for this specific problem size
    key = (v, k, t)
    if key not in _PRECOMP_CACHE:
        if v > 20 or k > 10: # Fallback for larger problems to avoid memory blowup
            # Simple greedy fallback
            uncovered = set(itertools.combinations(range(v), t))
            blocks = []
            while uncovered:
                # Pick a random uncovered t-subset to start a block? 
                # Or pick the t-subset that is in the fewest blocks? 
                # For simplicity, just pick the first one and greedily expand.
                start_tsub = next(iter(uncovered))
                block = list(start_tsub)
                # Greedily add points to cover more uncovered t-subsets
                while len(block) < k:
                    best_point = None
                    best_gain = -1
                    for p in range(v):
                        if p in block:
                            continue
                        # Calculate gain: how many new t-subsets does adding p cover?
                        # A t-subset is covered if it is a subset of block + {p}
                        # It must contain p and t-1 elements from block.
                        gain = 0
                        for rest in itertools.combinations(block, t-1):
                            ts = tuple(sorted(rest + (p,)))
                            if ts in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = p
                    if best_point is None:
                        break
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                # Update uncovered
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
            return blocks
            
        _PRECOMP_CACHE[key] = _precompute_covering_data(v, k, t)
    
    t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t = _PRECOMP_CACHE[key]
    
    # Number of trials for randomized greedy
    num_trials = 20
    best_sol = None
    best_len = float('inf')
    
    full_mask = (1 << num_t) - 1
    
    for trial in range(num_trials):
        random.seed(trial) # Deterministic per trial for reproducibility in testing, but varies across trials
        uncovered = full_mask
        sol = []
        
        # Greedy Phase
        while uncovered:
            best_block_idx = -1
            max_covered = 0
            
            # To speed up, we can iterate through all blocks. 
            # 50388 blocks * ~150 steps = ~7.5M iterations.
            # Each iteration is a bitwise AND and bit_count. This is fast in Python 3.10+.
            # If bit_count is not available, we use bin(x).count('1'), which is slower but still okay for 120s.
            
            # Optimization: Only consider blocks that cover at least one uncovered t-subset.
            # This is hard to filter quickly without precomputing inverse indices.
            # Let's just iterate all.
            
            for i in range(len(block_combinations)):
                covered = block_masks[i] & uncovered
                if covered:
                    # Count bits
                    # Using bit_count if available
                    try:
                        cnt = covered.bit_count()
                    except AttributeError:
                        cnt = bin(covered).count('1')
                    
                    if cnt > max_covered:
                        max_covered = cnt
                        best_block_idx = i
                        # Early exit if we cover all uncovered? Unlikely.
                        # If cnt == popcount(uncovered), we found a perfect block.
            
            if best_block_idx == -1:
                # Should not happen if uncovered is not empty and blocks cover all t-subsets
                break
                
            sol.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        # Local Search: Remove Redundant Blocks
        # Repeat until no more blocks can be removed
        changed = True
        while changed:
            changed = False
            # Calculate frequency of each t-subset coverage
            freq = [0] * num_t
            for b_idx in sol:
                for t_idx in block_tsub_indices[b_idx]:
                    freq[t_idx] += 1
            
            new_sol = []
            for b_idx in sol:
                is_redundant = True
                for t_idx in block_tsub_indices[b_idx]:
                    if freq[t_idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_sol.append(b_idx)
            
            if len(new_sol) < len(sol):
                sol = new_sol
                changed = True
            else:
                changed = False
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol.copy()
            
    if best_sol is None:
        # Fallback: return empty or error? Should not happen.
        return []
        
    # Convert indices back to lists of points
    result_blocks = []
    for b_idx in best_sol:
        result_blocks.append(list(block_combinations[b_idx]))
        
    return result_blocks
