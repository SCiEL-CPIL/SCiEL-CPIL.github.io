import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point x, the list of indices of t-subsets containing x
    # This helps in quick updates? Maybe not strictly necessary for the gain calc 
    # derived above, but good for update.
    # Actually, for update, we just iterate over combinations of the block.
    
    def greedy_construct(random_seed=0):
        random.seed(random_seed)
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks = []
        
        # We need to pick the first point of the block.
        # The original code picks min(uncovered). 
        # Let's pick a random point to start diversity, or the one that appears 
        # in the most uncovered subsets? 
        # Simple strategy: pick random point from v.
        
        while np.any(uncovered):
            # Start a new block
            # Pick a random starting point
            start_point = random.randint(0, v - 1)
            block = [start_point]
            block_set = {start_point}
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block_set]
                
                # Calculate gain for each candidate
                # Gain(x) = count of uncovered S such that S subseteq block U {x}
                # This implies S = T U {x} where T subseteq block, |T|=t-1
                # We iterate over all (t-1)-subsets of the current block
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    for T in t_minus_1_subsets:
                        # Form the t-subset
                        S = tuple(sorted(T + (x,)))
                        if S in subset_to_idx:
                            idx = subset_to_idx[S]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if gain == len(t_minus_1_subsets): # Max possible gain for this step?
                             # Actually max gain is limited by number of uncovered subsets
                             # But if we hit all possible T, we can't do better for this x
                             # But other x might cover different sets.
                             pass
                            
                if best_x == -1:
                    break # Should not happen if uncovered exists and k>0
                    
                block.append(best_x)
                block_set.add(best_x)
                
            # Block is full
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for S in itertools.combinations(block, t):
                idx = subset_to_idx[S]
                uncovered[idx] = False
                
        return blocks

    def polish(blocks, time_limit):
        # Maintain coverage counts
        coverage_counts = np.zeros(num_t_subsets, dtype=int)
        for block in blocks:
            for S in itertools.combinations(block, t):
                coverage_counts[subset_to_idx[S]] += 1
        
        # Try to remove blocks
        # We iterate multiple times or until time runs out
        # Random order helps escape local minima? Or just sequential?
        # Let's do multiple passes with random order.
        
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks
            
        # Keep track of active blocks
        active = [True] * n_blocks
        
        # We will try to remove each block.
        # To speed up, we can just iterate and try removal.
        # If removal fails, we restore.
        
        # A more efficient way:
        # For each block i, check if it is "critical".
        # A block is critical if it contains at least one t-subset with count 1.
        # If not critical, we can remove it safely.
        
        # Let's precompute which blocks contain critical subsets.
        # Actually, just trying removal is O(num_t_subsets_in_block) per check.
        
        passes = 0
        while time.time() < time_limit:
            passes += 1
            # Shuffle indices for random order
            indices = list(range(n_blocks))
            random.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                    
                block = blocks[i]
                # Check if any t-subset in this block has count 1
                is_critical = False
                for S in itertools.combinations(block, t):
                    if coverage_counts[subset_to_idx[S]] == 1:
                        is_critical = True
                        break
                        
                if not is_critical:
                    # Safe to remove
                    active[i] = False
                    for S in itertools.combinations(block, t):
                        coverage_counts[subset_to_idx[S]] -= 1
                    # Optional: print(f"Removed block {i}, remaining: {sum(active)}")
                    
        # Reconstruct list of active blocks
        final_blocks = [blocks[i] for i in range(n_blocks) if active[i]]
        return final_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run greedy with different seeds
    # Estimate how many seeds we can run
    # Greedy is roughly O( (C(v,k)/k) * k * C(k,t-1) * C(v,t) ) ? No, simpler.
    # Let's just run as many as possible.
    
    seed = 0
    while time.time() < deadline - 20.0: # Leave 20s for polish
        # Check time periodically
        if time.time() > deadline - 5.0:
            break
            
        blocks = greedy_construct(seed)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
        seed += 1
        
        # If we find a very good one, maybe stop early?
        # Lower bound is 264. If we get close, we can spend more time polishing.
        if best_len <= 280:
             break

    if best_blocks is None:
        # Fallback: run one simple greedy
        best_blocks = greedy_construct(0)
        
    # Polish phase
    if time.time() < deadline - 2.0:
        polished_blocks = polish(best_blocks, deadline - 2.0)
        if len(polished_blocks) < len(best_blocks):
            best_blocks = polished_blocks

    return best_blocks
