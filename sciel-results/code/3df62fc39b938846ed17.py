import itertools
import numpy as np
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Covering Density heuristic.
    
    Approach:
    1. Generate a dense pool of blocks from cyclic shifts of structured base blocks.
    2. Use greedy set cover to get initial solution.
    3. Apply local search minimizing sum of inverse coverage counts:
       - For each block, compute marginal redundancy (how much it contributes to over-coverage)
       - Try to remove or swap blocks to reduce total count while maintaining coverage
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Generate candidate blocks using cyclic shifts of base blocks
    random.seed(42)
    
    # Generate diverse base blocks - mix of random and structured
    base_blocks = []
    
    # Add some structured patterns (e.g., consecutive elements, spread out)
    for _ in range(50):
        # Consecutive-like patterns
        start = random.randint(0, v - k)
        base = list(range(start, start + k))
        # Perturb slightly
        for _ in range(random.randint(0, 2)):
            if len(base) > 0:
                idx = random.randint(0, len(base) - 1)
                base[idx] = random.randint(0, v - 1)
                base = sorted(set(base))
                while len(base) < k:
                    new_elem = random.randint(0, v - 1)
                    if new_elem not in base:
                        base.append(new_elem)
                        base = sorted(base)
        if len(base) == k:
            base_blocks.append(tuple(sorted(base)))
    
    # Add random base blocks
    for _ in range(150):
        base = tuple(sorted(random.sample(range(v), k)))
        base_blocks.append(base)
    
    # Generate all cyclic shifts
    candidate_blocks = []
    seen_blocks = set()
    for base in base_blocks:
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in base))
            if block not in seen_blocks:
                seen_blocks.add(block)
                candidate_blocks.append(block)
    
    # For each candidate block, compute the t-subset indices it covers
    block_t_indices = []
    for block in candidate_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        block_t_indices.append(np.array(indices, dtype=np.int32))
    
    # Greedy set cover
    uncovered = np.ones(n_t_subsets, dtype=np.bool_)
    selected_block_indices = []
    
    # Track which candidate blocks are still available
    available = np.ones(len(candidate_blocks), dtype=np.bool_)
    
    while uncovered.any() and time.time() - start_time < time_limit:
        # Find the block with maximum coverage of uncovered t-subsets
        # Vectorized: compute coverage for all available blocks at once
        covers = np.zeros(len(candidate_blocks), dtype=np.int32)
        for idx in np.where(available)[0]:
            covers[idx] = np.sum(uncovered[block_t_indices[idx]])
        
        # Only consider available blocks
        covers[~available] = -1
        best_block_idx = np.argmax(covers)
        max_cover = covers[best_block_idx]
        
        if max_cover <= 0:
            break
        
        selected_block_indices.append(best_block_idx)
        available[best_block_idx] = False
        
        # Update uncovered
        uncovered[block_t_indices[best_block_idx]] = False
    
    # Convert to actual blocks
    blocks = [list(candidate_blocks[i]) for i in selected_block_indices]
    block_t_indices_current = [block_t_indices[i] for i in selected_block_indices]
    
    # Compute initial coverage counts
    coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
    for indices in block_t_indices_current:
        coverage_counts[indices] += 1
    
    # Local search: minimize sum of inverse coverage counts
    # Energy = sum over all t-subsets of 1/count(t_subset)
    # We want to reduce this by removing blocks that have high marginal redundancy
    
    def compute_energy(counts):
        # Avoid division by zero
        positive_counts = counts[counts > 0]
        if len(positive_counts) == 0:
            return float('inf')
        return np.sum(1.0 / positive_counts)
    
    current_energy = compute_energy(coverage_counts)
    best_energy = current_energy
    best_blocks = [list(b) for b in blocks]
    best_block_t_indices = [np.array(idx) for idx in block_t_indices_current]
    best_coverage_counts = coverage_counts.copy()
    
    # Local search iterations
    max_iterations = 1000
    no_improvement_count = 0
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try to remove each block if it's redundant
        for i in range(len(blocks)):
            if time.time() - start_time > time_limit:
                break
            
            # Check if removing block i maintains coverage
            if np.all(coverage_counts[block_t_indices_current[i]] > 1):
                # Safe to remove
                coverage_counts[block_t_indices_current[i]] -= 1
                blocks.pop(i)
                block_t_indices_current.pop(i)
                
                new_energy = compute_energy(coverage_counts)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_blocks = [list(b) for b in blocks]
                    best_block_t_indices = [np.array(idx) for idx in block_t_indices_current]
                    best_coverage_counts = coverage_counts.copy()
                
                improved = True
                no_improvement_count = 0
                break  # Restart from beginning
        
        if improved:
            continue
        
        # Try swap: replace a block with another that reduces energy
        # Find the block with highest marginal redundancy
        # Marginal redundancy of block i = sum over its t-subsets of (1/count - 1/(count-1))
        # This is positive when count > 1, meaning removing it would increase energy
        
        # Compute marginal cost of removing each block
        marginal_costs = []
        for i in range(len(blocks)):
            indices = block_t_indices_current[i]
            counts = coverage_counts[indices]
            # If any count is 1, can't remove
            if np.any(counts == 1):
                marginal_costs.append(float('inf'))
            else:
                # Energy change if we remove this block
                # Old contribution: sum(1/counts)
                # New contribution: sum(1/(counts-1))
                old_contrib = np.sum(1.0 / counts)
                new_contrib = np.sum(1.0 / (counts - 1))
                marginal_costs.append(new_contrib - old_contrib)
        
        marginal_costs = np.array(marginal_costs)
        
        # Find block with smallest marginal cost (most redundant)
        min_cost_idx = np.argmin(marginal_costs)
        
        if marginal_costs[min_cost_idx] < 0:
            # Removing this block decreases energy
            i = min_cost_idx
            coverage_counts[block_t_indices_current[i]] -= 1
            blocks.pop(i)
            block_t_indices_current.pop(i)
            
            new_energy = compute_energy(coverage_counts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_blocks = [list(b) for b in blocks]
                best_block_t_indices = [np.array(idx) for idx in block_t_indices_current]
                best_coverage_counts = coverage_counts.copy()
            
            no_improvement_count = 0
        else:
            # Try to find a swap that improves energy
            # Look for a block that, when removed and replaced, reduces total count
            # This is more complex; for now, just try random swaps
            
            if len(blocks) > 1 and random.random() < 0.3:
                # Try removing a random block and see if we can cover its uncovered t-subsets
                # with fewer new blocks
                i = random.randint(0, len(blocks) - 1)
                indices_to_cover = block_t_indices_current[i]
                currently_covered = coverage_counts[indices_to_cover] > 1
                
                # Find t-subsets that would become uncovered
                would_uncover = indices_to_cover[~currently_covered]
                
                if len(would_uncover) == 0:
                    # All covered by others, safe to remove
                    coverage_counts[indices_to_cover] -= 1
                    blocks.pop(i)
                    block_t_indices_current.pop(i)
                    
                    new_energy = compute_energy(coverage_counts)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_blocks = [list(b) for b in blocks]
                        best_block_t_indices = [np.array(idx) for idx in block_t_indices_current]
                        best_coverage_counts = coverage_counts.copy()
                    
                    no_improvement_count = 0
                elif len(would_uncover) <= 35:
                    # Try to find one block that covers all would_uncover t-subsets
                    # and doesn't add too much over-coverage
                    for j, block in enumerate(candidate_blocks):
                        if j in selected_block_indices:
                            continue
                        block_indices = block_t_indices[j]
                        # Check if this block covers all would_uncover
                        if np.all(np.isin(would_uncover, block_indices)):
                            # Compute energy change
                            old_energy = compute_energy(coverage_counts)
                            
                            # Remove block i
                            coverage_counts[indices_to_cover] -= 1
                            # Add block j
                            coverage_counts[block_indices] += 1
                            
                            new_energy = compute_energy(coverage_counts)
                            
                            if new_energy < best_energy:
                                best_energy = new_energy
                                # Actually perform the swap
                                blocks[i] = list(block)
                                block_t_indices_current[i] = block_indices
                                no_improvement_count = 0
                                improved = True
                            else:
                                # Revert
                                coverage_counts[block_indices] -= 1
                                coverage_counts[indices_to_cover] += 1
                            
                            if improved:
                                break
                    
                    if not improved:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
        
        if no_improvement_count > 50:
            break
    
    # Return the best solution found
    return best_blocks
