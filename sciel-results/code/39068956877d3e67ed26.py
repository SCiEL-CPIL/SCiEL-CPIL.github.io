import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    initial_uncovered = set(all_t_subsets)
    
    best_blocks = None
    best_len = float('inf')
    
    # Helper to greedily build a block from a seed
    def build_block(seed, uncovered_set):
        block = list(seed)
        # We need to add k - t points
        # To speed up, we can pre-compute candidates? No, just iterate.
        while len(block) < k:
            best_gain = -1
            best_point = None
            # Calculate gain for each candidate point
            # Gain(x) = number of uncovered T such that T \ {x} subset of block and x in T
            # This is equivalent to: T is in uncovered, T contains x, and T \ {x} is subset of block.
            # Let R = T \ {x}. R is a (t-1)-subset of block.
            # So we iterate (t-1)-subsets of block, add x, check if in uncovered.
            
            block_set = set(block)
            # Optimization: Pre-compute the (t-1) subsets of current block?
            # Block size is small (<= 7).
            
            candidates = [x for x in range(v) if x not in block_set]
            
            # If t-1 == 0, gain is just 1 if {x} is uncovered? No, t>=1.
            # For t=5, we look at 4-subsets of block.
            
            # To make this faster, we can just iterate candidates and check subsets.
            # But checking subsets for every candidate is expensive if done naively.
            # Alternative: For each candidate x, count how many uncovered sets T contain x and T\{x} subset of block.
            
            # Let's optimize the gain calculation.
            # We can create a dictionary of uncovered sets keyed by their (t-1) subsets?
            # That's complex.
            
            # Given small sizes, let's just do it directly but efficiently.
            # For a fixed block P, the set of (t-1)-subsets is fixed.
            # Let R_list = list(itertools.combinations(block, t-1))
            # For a candidate x, gain = sum(1 for R in R_list if tuple(sorted(R + (x,))) in uncovered_set)
            
            R_list = list(itertools.combinations(block, t - 1))
            
            for x in candidates:
                gain = 0
                for R in R_list:
                    # Check if R U {x} is in uncovered
                    # R is a tuple of t-1 elements. x is an int.
                    # We need to check if the set R U {x} is in uncovered.
                    # Since uncovered contains sorted tuples, we need to check if sorted(R+(x,)) is in uncovered.
                    # To avoid sorting every time, we can pre-sort R? R is from combinations, so it's sorted.
                    # We just need to insert x into R in sorted order.
                    
                    # Efficient check:
                    # Create a tuple of R + (x,), sort it, check membership.
                    # Sorting a small tuple (size 5) is fast.
                    new_t = tuple(sorted(R + (x,)))
                    if new_t in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                # Should not happen if uncovered is not empty and we can extend
                break
            block.append(best_point)
        
        block.sort()
        return tuple(block)

    # Main loop with restarts
    restart_count = 0
    while time.time() < deadline:
        uncovered = set(initial_uncovered)
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 5:
                break
            
            # Select seed
            # Heuristic: Pick a random uncovered subset
            # To improve, we could pick the one with min "degree" but it's uniform.
            # Let's try picking the one that yields the highest gain block among a few samples.
            
            sample_size = 5
            if len(uncovered) < sample_size:
                candidates_seeds = list(uncovered)
            else:
                candidates_seeds = random.sample(list(uncovered), sample_size)
            
            best_block = None
            best_gain_val = -1
            
            for seed in candidates_seeds:
                b = build_block(seed, uncovered)
                # Calculate gain of this block
                # Gain is number of uncovered t-subsets in b
                gain_val = 0
                for ts in itertools.combinations(b, t):
                    if ts in uncovered:
                        gain_val += 1
                
                if gain_val > best_gain_val:
                    best_gain_val = gain_val
                    best_block = b
            
            if best_block is None:
                # Fallback
                seed = next(iter(uncovered))
                best_block = build_block(seed, uncovered)
            
            blocks.append(best_block)
            
            # Remove covered
            to_remove = []
            for ts in itertools.combinations(best_block, t):
                if ts in uncovered:
                    to_remove.append(ts)
            for ts in to_remove:
                uncovered.discard(ts)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        restart_count += 1
    
    if best_blocks is None:
        # Fallback to simple greedy if no time left or error
        uncovered = set(initial_uncovered)
        blocks = []
        while uncovered:
            seed = next(iter(uncovered))
            b = build_block(seed, uncovered)
            blocks.append(b)
            for ts in itertools.combinations(b, t):
                uncovered.discard(ts)
        best_blocks = blocks
        
    return [list(b) for b in best_blocks]
