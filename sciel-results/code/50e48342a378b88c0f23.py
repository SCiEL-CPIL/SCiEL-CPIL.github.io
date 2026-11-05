import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Use 145s to be safe
    deadline = start_time + 145.0
    
    # 1. Indexing
    subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(subsets)}
    num_subsets = len(subsets)
    
    # 2. Precompute block masks
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = {}
    for b in all_blocks:
        mask = 0
        for s in itertools.combinations(b, t):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
        block_masks[b] = mask
            
    # Helper: Popcount
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    # Helper: Greedy cover for a specific uncovered mask
    def greedy_cover(uncovered_mask):
        blocks = []
        while uncovered_mask:
            if time.time() > deadline:
                break
            best_block = None
            max_gain = -1
            for b in all_blocks:
                gain = popcount(uncovered_mask & block_masks[b])
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
            if best_block is None:
                break
            blocks.append(best_block)
            uncovered_mask &= ~block_masks[best_block]
        return blocks

    # Helper: Remove redundant blocks
    def optimize(blocks):
        if not blocks:
            return []
        
        cover_counts = [0] * num_subsets
        for b in blocks:
            for s in itertools.combinations(b, t):
                cover_counts[subset_to_idx[s]] += 1
                
        i = 0
        while i < len(blocks) and time.time() < deadline:
            b = blocks[i]
            # Check redundancy
            for s in itertools.combinations(b, t):
                cover_counts[subset_to_idx[s]] -= 1
            redundant = True
            for s in itertools.combinations(b, t):
                if cover_counts[subset_to_idx[s]] == 0:
                    redundant = False
                    for s2 in itertools.combinations(b, t):
                        cover_counts[subset_to_idx[s2]] += 1
                    break
            if redundant:
                blocks.pop(i)
            else:
                i += 1
        return blocks

    # 3. Initial Solution
    initial_uncovered = (1 << num_subsets) - 1
    best_blocks = greedy_cover(initial_uncovered)
    best_blocks = optimize(best_blocks)
    best_len = len(best_blocks)
    
    # 4. ILS Loop
    current_blocks = best_blocks
    
    while time.time() < deadline - 5.0:
        # Perturb
        # Remove 1 to 3 random blocks
        num_remove = random.randint(1, 3)
        if len(current_blocks) <= num_remove:
            continue
            
        removed_indices = random.sample(range(len(current_blocks)), num_remove)
        # Create a copy
        perturbed = list(current_blocks)
        # Remove in reverse order to keep indices valid
        for idx in sorted(removed_indices, reverse=True):
            perturbed.pop(idx)
            
        # Calculate uncovered mask after removal
        cover_counts = [0] * num_subsets
        for b in perturbed:
            for s in itertools.combinations(b, t):
                cover_counts[subset_to_idx[s]] += 1
        
        uncovered_mask = 0
        for i in range(num_subsets):
            if cover_counts[i] == 0:
                uncovered_mask |= (1 << i)
                
        # Repair
        new_blocks = greedy_cover(uncovered_mask)
        perturbed.extend(new_blocks)
        
        # Optimize
        perturbed = optimize(perturbed)
        
        if len(perturbed) < best_len:
            best_blocks = perturbed
            best_len = len(perturbed)
            current_blocks = best_blocks
        else:
            # Sometimes accept a slightly worse solution to explore?
            # Let's try: if len(perturbed) <= best_len + 2, accept with 10% chance?
            # No, let's stick to best-first for stability.
            pass

    return [list(b) for b in best_blocks]
