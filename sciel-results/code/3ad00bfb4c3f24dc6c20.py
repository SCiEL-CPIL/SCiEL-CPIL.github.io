import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets as bitmasks
    all_t_subsets = []
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_t_subsets.append(mask)
    
    # Precompute mapping from (t-1)-subset mask to list of t-subset masks containing it
    # t-1 = 4
    subs_by_t_minus_1 = {}
    for mask_t in all_t_subsets:
        # Get all (t-1) subsets of this t-subset
        # Iterate through bits set in mask_t
        points_in_t = [i for i in range(v) if (mask_t >> i) & 1]
        for combo_t_minus_1 in itertools.combinations(points_in_t, t - 1):
            mask_t_minus_1 = 0
            for x in combo_t_minus_1:
                mask_t_minus_1 |= (1 << x)
            if mask_t_minus_1 not in subs_by_t_minus_1:
                subs_by_t_minus_1[mask_t_minus_1] = []
            subs_by_t_minus_1[mask_t_minus_1].append(mask_t)

    def greedy_once():
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            # This helps diversify the search
            seed_mask = random.choice(list(uncovered))
            
            # Start block with points from seed_mask
            block_points = [i for i in range(v) if (seed_mask >> i) & 1]
            
            while len(block_points) < k:
                # Calculate gain for each candidate point
                best_gain = -1
                best_points = []
                
                for x in range(v):
                    if x in block_points:
                        continue
                    
                    # Calculate how many uncovered t-subsets are covered by adding x
                    # A t-subset S is covered if S contains x and S \ {x} is a subset of block_points
                    # This implies S \ {x} is a (t-1)-subset of block_points
                    gain = 0
                    # Iterate over all (t-1)-subsets of block_points
                    # Since block_points size < k=8, and t-1=4, this is small
                    for combo_t_minus_1 in itertools.combinations(block_points, t - 1):
                        mask_t_minus_1 = 0
                        for y in combo_t_minus_1:
                            mask_t_minus_1 |= (1 << y)
                        
                        # Check if x is a valid extension for any uncovered t-subset with this (t-1) base
                        # The t-subset would be mask_t_minus_1 | (1 << x)
                        mask_t_candidate = mask_t_minus_1 | (1 << x)
                        
                        # If this candidate t-subset is in uncovered, it gets covered
                        if mask_t_candidate in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                # Pick randomly from best points to break ties
                if best_points:
                    block_points.append(random.choice(best_points))
                else:
                    # Fallback: pick any point not in block
                    for x in range(v):
                        if x not in block_points:
                            block_points.append(x)
                            break
            
            block_points.sort()
            blocks.append(block_points)
            
            # Remove all t-subsets of this block from uncovered
            for combo_t in itertools.combinations(block_points, t):
                mask_t = 0
                for x in combo_t:
                    mask_t |= (1 << x)
                if mask_t in uncovered:
                    uncovered.remove(mask_t)
                    
        return blocks

    # Run multiple random restarts
    best_blocks = None
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    while time.time() < deadline:
        current_blocks = greedy_once()
        if best_blocks is None or len(current_blocks) < len(best_blocks):
            best_blocks = current_blocks
            # Optional: break if we hit a theoretical lower bound? 
            # Lower bound for C(19,8,5) is 264.
            # If we find 264, we can stop.
            if len(best_blocks) <= 264:
                break

    return best_blocks
