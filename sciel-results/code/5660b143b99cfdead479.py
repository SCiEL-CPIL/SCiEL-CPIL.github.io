import itertools
import random

def construct(v, k, t):
    # Generate all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_subsets)
    blocks = []
    
    while uncovered:
        # 1. Select a random uncovered subset as pivot
        pivot = random.choice(list(uncovered))
        pivot_set = set(pivot)
        
        # 2. Generate all candidate blocks that contain pivot
        remaining_points = [p for p in range(v) if p not in pivot_set]
        
        max_gain = -1
        best_blocks = []
        
        # Generate all (k-t) subsets of remaining_points
        for r_subset in itertools.combinations(remaining_points, k - t):
            # Construct the block
            # Combine pivot and r_subset
            block_elements = list(pivot) + list(r_subset)
            block = tuple(sorted(block_elements))
            
            # Count how many uncovered t-subsets are in this block
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_blocks = [block]
            elif gain == max_gain:
                best_blocks.append(block)
                
        # 3. Randomly select one of the best blocks
        best_block = random.choice(best_blocks)
        blocks.append(list(best_block))
        
        # 4. Update uncovered set
        for sub in itertools.combinations(best_block, t):
            if sub in uncovered:
                uncovered.discard(sub)
                
    return blocks
