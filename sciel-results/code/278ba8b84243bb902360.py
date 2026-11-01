import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    # Hard wall-clock limit with safety margin. The evaluator kills at 150s;
    # we target returning by 145s to be safe.
    time_limit = 145.0
    
    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # t_count is the number of t-subsets in a k-set
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        # Use integer arithmetic for small numbers to avoid float issues
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
        
    t_count = nCr(k, t)
    
    # block_t_indices[i] is an array of indices into all_ts for the i-th block
    # Shape: (n_ks, t_count)
    # For v=24, k=7, t=3: n_ks=346104, t_count=35. Size ~ 12M int32s ~ 48MB.
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # Also precompute the point sets for each block for faster load checks?
    # all_ks is already a list of tuples of points.
    
    best_solution = None
    best_score = float('inf')
    
    # We will run multiple randomized greedy restarts.
    # Each restart:
    # 1. Greedy construction with random tie-breaking (possibly weighted by point loads or just uniform).
    # 2. Local search: iterative redundancy removal.
    # 3. Optional: local search with swaps? (Too complex for time, stick to removal).
    
    # To diversify, we can vary the tie-breaking strategy.
    # Strategy 1: Random among max gain.
    # Strategy 2: Random among top K gains (e.g., top 10% or top 50).
    # Strategy 3: Weighted random based on gain and point load.
    
    # Let's implement a flexible greedy loop.
    
    def run_greedy_restart(seed, strategy="random_max", top_k_frac=0.1):
        """
        Runs a greedy construction followed by local search.
        Returns (num_blocks, list_of_block_indices)
        """
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        
        # Optimization: We can pre-compute the gain matrix update?
        # No, gains depend on 'covered'.
        # gains = (~covered[block_t_indices]).sum(axis=1)
        
        # To speed up the greedy step, we can maintain an array of "uncovered count" for each block?
        # No, that's what gains is.
        # The bottleneck is the full scan of n_ks every step.
        # n_steps ~ 76. n_ks ~ 346k. 76 * 346k * 35 ops ~ 900M ops.
        # In NumPy, this is vectorized.
        # ~covered is a view? No, ~ creates a new array.
        # covered[block_t_indices] creates a boolean array of size (n_ks, t_count).
        # This is 346104 * 35 ~ 12M booleans.
        # 76 * 12M = ~1B boolean ops. This should take a few seconds in NumPy.
        
        while True:
            # Calculate gains
            # We want the number of uncovered t-subsets in each block.
            # uncovered_flags = ~covered[block_t_indices]
            # gains = uncovered_flags.sum(axis=1)
            
            # To save memory/time, we can do:
            # gains = t_count - covered[block_t_indices].sum(axis=1)
            # This avoids the ~ operator and might be slightly faster?
            # covered[block_t_indices] is a boolean array. .sum() counts Trues.
            
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find candidates
            if strategy == "random_max":
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) == 0:
                    break
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            elif strategy == "top_k_random":
                # Pick from the top K% of gains
                # Sort indices by gain? Expensive.
                # Instead, find threshold.
                # We can use np.partition or just take max and go down?
                # Simpler: take top K elements.
                # But we need to be careful with time.
                # Let's stick to random_max for simplicity and speed,
                # or implement a simple top-k selection if time permits.
                # For now, let's try "random_top" where we pick randomly from blocks with gain >= max_gain - delta
                # Let's try picking randomly from the top 100 blocks by gain.
                # np.argsort is O(N log N). N=346k. ~3M ops. 76 times ~ 200M ops. Acceptable.
                # But let's stick to the proven "random_max" first, maybe add "random_top_100" later.
                
                # Let's implement "random_top_100"
                # Get indices of top 100 gains
                # np.argpartition(gains, -100)[-100:] gives the last 100 (largest)
                # Then random pick from them.
                if n_ks > 100:
                    top_indices = np.argpartition(gains, -100)[-100:]
                else:
                    top_indices = np.arange(n_ks)
                best_idx = top_indices[random.randint(0, len(top_indices) - 1)]
            else:
                candidates = np.where(gains == max_gain)[0]
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
                
            # Add block
            blocks_indices.append(best_idx)
            
            # Update covered
            covered[block_t_indices[best_idx]] = True
            
            # Check if all covered
            # Instead of np.all(covered), we can track remaining count?
            # If we track remaining, we can stop earlier.
            # But np.all is fast.
            # However, if we use the "remaining" count, we can break immediately.
            # Let's add a counter for efficiency.
            
        # Local Search: Remove redundant blocks
        # We do this in a loop until no more removals.
        current_blocks = list(blocks_indices)
        
        # To speed up redundancy check, we can maintain coverage counts.
        # But rebuilding is cheap for ~80 blocks.
        
        improved = True
        while improved:
            improved = False
            new_blocks = []
            
            # Build coverage count
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
                
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                # If all subsets in this block are covered by at least one OTHER block,
                # then coverage_count[subs] > 1 for all subs in block.
                # Note: coverage_count includes this block itself.
                # So we need min(coverage_count[subs_in_block]) > 1.
                
                if np.min(coverage_count[subs_in_block]) > 1:
                    # Redundant
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
                    
            current_blocks = new_blocks
            
        return len(current_blocks), current_blocks

    # Main Loop: Run multiple restarts with different seeds and strategies.
    # We need to track the best solution found so far.
    
    restart_count = 0
    # Strategies to try
    # 1. random_max
    # 2. top_k_random (top 100)
    
    while time.time() - start_time < time_limit - 5.0:
        seed = restart_count * 12345 + 678
        # Alternate strategies
        if restart_count % 2 == 0:
            strategy = "random_max"
        else:
            strategy = "top_k_random"
            
        num_blocks, block_indices = run_greedy_restart(seed, strategy)
        
        if num_blocks < best_score:
            best_score = num_blocks
            best_solution = block_indices
            # If we hit the theoretical lower bound or a very good score, we might stop early?
            # Lower bound is 69. Best known is 76.
            # If we find 70-75, it's a great result.
            # Let's continue searching for a while to see if we can beat it.
            
        restart_count += 1
        
        # If we've done a few restarts and found a very good solution, we can spend more time on local search?
        # For now, just keep restarting.
        
    if best_solution is None:
        # Fallback: just run one greedy if no time for restarts?
        # This shouldn't happen.
        num_blocks, block_indices = run_greedy_restart(0, "random_max")
        best_solution = block_indices
        best_score = num_blocks

    # Convert indices back to blocks
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
