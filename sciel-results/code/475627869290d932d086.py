import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Precompute indices for all t-subsets
    # Map (a, b, c, d) to index
    # Since v=19, t=4, total 3876 subsets.
    # We can use a dictionary or a direct mapping.
    # Direct mapping:
    # idx = C(v, t) ... no, standard combination index.
    
    # For speed, precompute all t-subsets and map them to 0..N-1
    all_t_subs = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute for each point x, and each (t-1)-subset S, the t-subset S U {x}
    # This might be overkill.
    
    # Greedy function
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = np.ones(N_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Find an uncovered subset
            # Randomly pick one to vary the search
            # Or pick the one with lowest "degree" in some sense?
            # For now, random pick among uncovered.
            # To make it faster, we can maintain a list of uncovered indices.
            # But updating it is costly.
            # Just use np.where(uncovered)[0] and pick random.
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0: break
            start_sub_idx = unc_idx[random.randint(0, len(unc_idx)-1)]
            start_sub = all_t_subs[start_sub_idx]
            
            block = list(start_sub)
            
            while len(block) < k:
                # Calculate gain for each candidate x
                best_gain = -1
                best_x = -1
                # Candidates
                candidates = [x for x in range(v) if x not in block]
                
                for x in candidates:
                    # Subsets formed by adding x to block:
                    # {x} U C where C is a (t-1)-subset of block
                    # block size is m. We need (t-1)-subsets of block.
                    # t=4, so we need 3-subsets of block.
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        # rest is a tuple of t-1 elements
                        # new_sub = tuple(sorted(rest + (x,)))
                        # Since block elements are not sorted necessarily? 
                        # We should keep block sorted or just sort the new sub.
                        # For speed, we can assume we insert x and sort.
                        # But sorting is slow.
                        # Better: keep block as a set? No, list is fine.
                        # We can pre-sort block.
                        # Actually, let's just create the tuple and sort.
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in sub_to_idx:
                            idx = sub_to_idx[new_sub]
                            if uncovered[idx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain:
                        # Random tie break
                        if random.random() < 0.5:
                            best_x = x
                            
                block.append(best_x)
                
            # Sort block for consistency (optional, but good for dedup)
            block.sort()
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                uncovered[idx] = False
                
            blocks.append(block)
            
        return blocks

    def remove_redundant(blocks):
        # For each block, check if all its t-subsets are covered by other blocks.
        # This is O(N_blocks^2 * C(k,t)).
        # N_blocks ~ 150. C(7,4)=35. 150^2 * 35 ~ 787,500 ops. Fast.
        
        # Precompute coverage: for each block i, list of t-subset indices it covers.
        block_covs = []
        for b in blocks:
            covs = [sub_to_idx[sub] for sub in itertools.combinations(b, t)]
            block_covs.append(covs)
            
        # Count how many blocks cover each t-subset
        count = np.zeros(N_t, dtype=int)
        for covs in block_covs:
            for idx in covs:
                count[idx] += 1
                
        kept_blocks = []
        for i, b in enumerate(blocks):
            # Check if any t-subset in b is covered ONLY by b
            is_critical = False
            for idx in block_covs[i]:
                if count[idx] == 1:
                    is_critical = True
                    break
            if not is_critical:
                # Block i is redundant.
                # Decrease counts
                for idx in block_covs[i]:
                    count[idx] -= 1
            else:
                kept_blocks.append(b)
                
        return kept_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    seed = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # Early stop if we hit theoretical lower bound? 
            # LB is 131.
            if best_len <= 131:
                break
                
        seed += 1
        
    return best_blocks
