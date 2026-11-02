import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    m = len(all_t_subsets)
    
    # Helper to get t-subsets of a block
    def get_block_t_subsets(block):
        return list(itertools.combinations(block, t))
        
    # Fast Greedy to get an initial upper bound
    def greedy_construct_blocks():
        cov = [0] * m
        blocks = []
        
        while True:
            min_cov_val = min(cov)
            if min_cov_val > 0:
                break
            
            candidates = [i for i in range(m) if cov[i] == min_cov_val]
            idx = random.choice(candidates)
            seed_subset = all_t_subsets[idx]
            
            block = list(seed_subset)
            
            while len(block) < k:
                best_gain = -1
                best_x = -1
                candidates_x = [x for x in range(v) if x not in block]
                block_t_minus_1 = list(itertools.combinations(block, t-1))
                
                for x in candidates_x:
                    gain = 0
                    for s_prime in block_t_minus_1:
                        t_sub = tuple(sorted(s_prime + (x,)))
                        i = subset_to_idx[t_sub]
                        if cov[i] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x != -1:
                        if random.random() < 0.5:
                            best_x = x
                
                if best_x == -1:
                    best_x = random.choice(candidates_x)
                block.append(best_x)
            
            block.sort()
            for t_sub in itertools.combinations(block, t):
                i = subset_to_idx[t_sub]
                cov[i] += 1
            blocks.append(block)
            
        return blocks, cov

    def remove_redundant(blocks, cov):
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                block = blocks[i]
                is_essential = False
                for t_sub in itertools.combinations(block, t):
                    idx = subset_to_idx[t_sub]
                    if cov[idx] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    for t_sub in itertools.combinations(block, t):
                        idx = subset_to_idx[t_sub]
                        cov[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks, cov

    # Simulated Annealing for a fixed number of blocks N
    def sa_find_cover(N, time_budget):
        # Initialize with random blocks
        blocks = [sorted(random.sample(range(v), k)) for _ in range(N)]
        
        # Calculate initial cost (uncovered subsets)
        cov = [0] * m
        for b in blocks:
            for t_sub in itertools.combinations(b, t):
                idx = subset_to_idx[t_sub]
                cov[idx] += 1
        
        current_cost = sum(1 for c in cov if c == 0)
        
        if current_cost == 0:
            return blocks, cov
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99
        sa_start = time.time()
        
        while T > T_min:
            if time.time() - sa_start > time_budget:
                break
            
            # Pick a random block to replace
            i = random.randrange(N)
            old_block = blocks[i]
            
            # Generate new random block
            new_block = sorted(random.sample(range(v), k))
            
            if old_block == new_block:
                T *= cooling_rate
                continue
            
            # Calculate delta cost
            # We need to count how many subsets become covered/uncovered
            # delta = (new_uncovered - old_uncovered)
            
            # Let's compute the change in coverage for each subset
            # This is O(2 * C(k,t))
            
            # Store old cov values for subsets in old_block
            # And new cov values for subsets in new_block
            
            # To be efficient, we can just iterate
            # But we need to revert if we reject.
            
            # Let's compute delta directly
            delta = 0
            
            # Subsets in old_block
            for t_sub in itertools.combinations(old_block, t):
                idx = subset_to_idx[t_sub]
                if cov[idx] == 1:
                    delta += 1 # Will become uncovered (cost increases)
            
            # Subsets in new_block
            for t_sub in itertools.combinations(new_block, t):
                idx = subset_to_idx[t_sub]
                if cov[idx] == 0:
                    delta -= 1 # Will become covered (cost decreases)
            
            # Accept if delta < 0 or with prob exp(-delta/T)
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Apply move
                for t_sub in itertools.combinations(old_block, t):
                    idx = subset_to_idx[t_sub]
                    cov[idx] -= 1
                for t_sub in itertools.combinations(new_block, t):
                    idx = subset_to_idx[t_sub]
                    cov[idx] += 1
                    
                blocks[i] = new_block
                current_cost += delta
                
                if current_cost == 0:
                    return blocks, cov
            
            T *= cooling_rate
            
        return blocks, cov

    # Main Logic
    best_blocks = None
    best_size = float('inf')
    
    # Get initial upper bound
    blocks, cov = greedy_construct_blocks()
    blocks, cov = remove_redundant(blocks, cov)
    best_size = len(blocks)
    best_blocks = blocks[:]
    
    # Now try to improve using SA
    # We try to find a valid cover with N < best_size
    # We iterate N from best_size - 1 down to 1
    
    # We'll use a time-based loop.
    # Each iteration: pick a target N, run SA for a short time.
    
    # To be efficient, we should start with N close to best_size.
    
    # Let's try N from best_size - 1 down to max(1, best_size - 30)
    # If we find a valid cover with N, we update best_size and continue.
    
    # We need to be careful with time.
    
    for N in range(best_size - 1, max(0, best_size - 30), -1):
        if time.time() - start_time > time_limit:
            break
            
        # Run SA to find a valid cover with N blocks
        # We give it a small time budget
        sa_time_budget = 1.0
        blocks_sa, cov_sa = sa_find_cover(N, sa_time_budget)
        
        # Check if valid
        uncovered = 0
        for c in cov_sa:
            if c == 0:
                uncovered += 1
        
        if uncovered == 0:
            # Valid cover found with N blocks
            # Remove redundant
            blocks_sa, cov_sa = remove_redundant(blocks_sa, cov_sa)
            
            if len(blocks_sa) < best_size:
                best_size = len(blocks_sa)
                best_blocks = blocks_sa[:]
                
    return best_blocks
