import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute popcount for all masks
    popcount = [0] * N
    for mask in range(1, N):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Precompute subset relationships: subset[mask] = list of masks that are strict supersets of mask
    # Actually, we need: for each pair, check if one is subset of other.
    # Let's precompute: for each mask, the list of masks that contain it (are supersets)
    # This is expensive to store fully. Instead, let's use a different approach.
    
    # For scoring: count pairs (i,j) where i<j and (masks[i] & masks[j]) == masks[i] or (masks[i] & masks[j]) == masks[j]
    # i.e., one is subset of the other.
    
    def score_family(masks):
        """Count comparable pairs in the family."""
        count = 0
        m = len(masks)
        for i in range(m):
            mi = masks[i]
            for j in range(i+1, m):
                mj = masks[j]
                if (mi & mj) == mi or (mi & mj) == mj:
                    count += 1
        return count
    
    # Generate a good initial family using chain-based approach
    # A chain is a sequence of sets where each is a strict subset of the next.
    # We want to maximize comparable pairs, so we want many chains and cross-comparability.
    
    # Strategy: Build a family that is a union of several chains, with some overlap.
    # Or better: use local search to optimize.
    
    # Initial solution: create a "dense" family by picking sets of various sizes
    # that are likely to be comparable.
    
    def random_initial():
        """Generate a random initial family biased toward comparable structures."""
        masks = set()
        # Start with empty set and full set
        masks.add(0)
        masks.add((1 << n) - 1)
        
        # Add random sets
        while len(masks) < m:
            size = random.randint(0, n)
            mask = 0
            for _ in range(size):
                bit = random.randint(0, n-1)
                mask |= (1 << bit)
            masks.add(mask)
        
        return list(masks)
    
    def chain_initial():
        """Generate an initial family based on chains."""
        masks = set()
        # Create several chains of varying lengths
        # Chain 1: single elements
        current = 0
        for i in range(n):
            current |= (1 << i)
            masks.add(current)
            if len(masks) >= m:
                break
        
        # Add some random sets to fill up
        while len(masks) < m:
            size = random.randint(0, n)
            mask = 0
            for _ in range(size):
                bit = random.randint(0, n-1)
                mask |= (1 << bit)
            masks.add(mask)
        
        return list(masks)
    
    def local_search(masks, max_iterations=50000):
        """Improve the family using local search (swap one set at a time)."""
        masks = list(masks)
        current_score = score_family(masks)
        best_score = current_score
        best_masks = list(masks)
        
        no_improve_count = 0
        iter_count = 0
        
        while iter_count < max_iterations:
            if time.time() > deadline - 2:
                break
            iter_count += 1
            
            # Pick a random position to modify
            idx = random.randint(0, m-1)
            old_mask = masks[idx]
            
            # Generate a new candidate mask
            # Strategy: pick a random mask, or modify the old one slightly
            if random.random() < 0.5:
                # Random new mask
                size = random.randint(0, n)
                new_mask = 0
                for _ in range(size):
                    bit = random.randint(0, n-1)
                    new_mask |= (1 << bit)
            else:
                # Mutate old mask: flip 1-3 bits
                new_mask = old_mask
                flips = random.randint(1, 3)
                for _ in range(flips):
                    bit = random.randint(0, n-1)
                    new_mask ^= (1 << bit)
            
            # Check if new_mask is already in the family (except at idx)
            if new_mask == old_mask:
                continue
            if new_mask in masks:
                continue
            
            # Evaluate the change
            # Remove old_mask, add new_mask, compute delta
            # To speed up, compute delta incrementally
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                mj = masks[j]
                # Check if old_mask and mj were comparable
                old_comp = 1 if ((old_mask & mj) == old_mask or (old_mask & mj) == mj) else 0
                # Check if new_mask and mj are comparable
                new_comp = 1 if ((new_mask & mj) == new_mask or (new_mask & mj) == mj) else 0
                delta += (new_comp - old_comp)
            
            if delta > 0:
                masks[idx] = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_masks = list(masks)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 1000:
                    # Restart with random perturbation
                    no_improve_count = 0
                    # Keep best so far, but perturb a few positions
                    for _ in range(random.randint(1, 3)):
                        idx2 = random.randint(0, m-1)
                        size = random.randint(0, n)
                        new_mask = 0
                        for _ in range(size):
                            bit = random.randint(0, n-1)
                            new_mask |= (1 << bit)
                        if new_mask not in masks:
                            masks[idx2] = new_mask
                            # Recompute score
                            current_score = score_family(masks)
        
        return best_masks, best_score
    
    # Run multiple restarts
    best_overall = None
    best_overall_score = -1
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            init_masks = chain_initial()
        else:
            init_masks = random_initial()
        
        # Run local search
        result_masks, result_score = local_search(init_masks, max_iterations=20000)
        
        if result_score > best_overall_score:
            best_overall_score = result_score
            best_overall = result_masks
    
    if best_overall is None:
        best_overall = chain_initial()
    
    # Convert masks to lists of integers
    result = []
    for mask in best_overall:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    # Ensure exactly m distinct sets
    # If we have duplicates or wrong count, fix it
    seen = set()
    final_result = []
    for s in result:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            final_result.append(list(s))
    
    # If we don't have enough, add random sets
    while len(final_result) < m:
        size = random.randint(0, n)
        s = sorted(random.sample(range(n), size))
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            final_result.append(s)
    
    # Trim if too many (shouldn't happen)
    final_result = final_result[:m]
    
    # Final type check: ensure all elements are Python int
    for i in range(len(final_result)):
        final_result[i] = [int(x) for x in final_result[i]]
    
    return final_result
