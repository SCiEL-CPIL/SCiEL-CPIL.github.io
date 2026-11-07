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
    
    # Precompute subset relationships using a different approach
    # For each mask, precompute which other masks it's comparable with
    # But that's O(N^2) = 16M which is fine
    
    # Actually, let's precompute a lookup: for each pair of masks, are they comparable?
    # That's N*N = 16M booleans, which is about 16MB - acceptable
    
    # Better: precompute for each mask, the set of masks that are subsets or supersets
    # But we only need to check pairs in our family, so let's keep the current approach
    # but optimize the inner loop
    
    def are_comparable(a, b):
        """Check if a and b are comparable (one is subset of other)."""
        return (a & b) == a or (a & b) == b
    
    def score_family(masks):
        """Count comparable pairs in the family."""
        count = 0
        m = len(masks)
        for i in range(m):
            mi = masks[i]
            for j in range(i+1, m):
                mj = masks[j]
                if are_comparable(mi, mj):
                    count += 1
        return count
    
    # Precompute a fast comparability check table for common operations
    # For each mask, precompute its popcount and maybe some other properties
    
    # Generate a good initial family using chain-based approach
    # Key insight: To maximize comparable pairs, we want many chains and cross-comparability.
    # A family that is a union of k chains of length L has k*L elements and 
    # k*L*(L-1)/2 comparable pairs within chains, plus cross-chain comparabilities.
    
    # Better strategy: Create a "grid" structure where we have several dimensions
    # and pick sets that form a lattice-like structure
    
    def structured_initial():
        """Generate an initial family with strong chain structure."""
        masks = set()
        
        # Create a main chain: 0, {0}, {0,1}, {0,1,2}, ..., {0,1,...,11}
        current = 0
        for i in range(n):
            current |= (1 << i)
            masks.add(current)
        
        # Add parallel chains with different starting points
        # Chain starting from {1}: {1}, {0,1}, {0,1,2}, ...
        # But {0,1} is already in the main chain, so we get cross-comparability
        
        # Add sets that are "close" to the main chain
        for i in range(n):
            # Add set missing element i from the full chain
            mask = (1 << n) - 1
            mask &= ~(1 << i)
            masks.add(mask)
            
            # Add set with only element i
            masks.add(1 << i)
            
            if len(masks) >= m:
                break
        
        # Fill remaining with random sets biased toward comparability
        while len(masks) < m:
            # Pick a random set from the existing family and modify it
            if masks and random.random() < 0.7:
                base = random.choice(list(masks))
                # Add or remove 1-2 elements
                new_mask = base
                ops = random.randint(1, 2)
                for _ in range(ops):
                    bit = random.randint(0, n-1)
                    if random.random() < 0.5:
                        new_mask |= (1 << bit)  # add
                    else:
                        new_mask &= ~(1 << bit)  # remove
                masks.add(new_mask)
            else:
                size = random.randint(0, n)
                mask = 0
                for _ in range(size):
                    bit = random.randint(0, n-1)
                    mask |= (1 << bit)
                masks.add(mask)
        
        return list(masks)
    
    def local_search(masks, max_iterations=50000):
        """Improve the family using local search with smarter mutations."""
        masks = list(masks)
        current_score = score_family(masks)
        best_score = current_score
        best_masks = list(masks)
        
        no_improve_count = 0
        iter_count = 0
        mask_set = set(masks)
        
        while iter_count < max_iterations:
            if time.time() > deadline - 2:
                break
            iter_count += 1
            
            # Pick a random position to modify
            idx = random.randint(0, m-1)
            old_mask = masks[idx]
            
            # Generate a new candidate mask using multiple strategies
            strategy = random.random()
            
            if strategy < 0.3:
                # Strategy 1: Random new mask
                size = random.randint(0, n)
                new_mask = 0
                for _ in range(size):
                    bit = random.randint(0, n-1)
                    new_mask |= (1 << bit)
            elif strategy < 0.5:
                # Strategy 2: Mutate old mask (flip 1-2 bits)
                new_mask = old_mask
                flips = random.randint(1, 2)
                for _ in range(flips):
                    bit = random.randint(0, n-1)
                    new_mask ^= (1 << bit)
            elif strategy < 0.8:
                # Strategy 3: Take a random existing mask and modify it
                other_idx = random.randint(0, m-1)
                if other_idx == idx:
                    other_idx = (other_idx + 1) % m
                base = masks[other_idx]
                new_mask = base
                # Add or remove 1-2 elements
                ops = random.randint(1, 2)
                for _ in range(ops):
                    bit = random.randint(0, n-1)
                    if random.random() < 0.5:
                        new_mask |= (1 << bit)
                    else:
                        new_mask &= ~(1 << bit)
            else:
                # Strategy 4: Create a set that's likely comparable with many existing sets
                # Pick a random existing set and make it a subset or superset
                other_idx = random.randint(0, m-1)
                if other_idx == idx:
                    other_idx = (other_idx + 1) % m
                base = masks[other_idx]
                if random.random() < 0.5:
                    # Make new_mask a subset of base
                    new_mask = base
                    # Remove 1-3 random elements
                    removals = random.randint(1, 3)
                    bits_in_base = [b for b in range(n) if (base >> b) & 1]
                    for _ in range(min(removals, len(bits_in_base))):
                        if not bits_in_base:
                            break
                        bit_idx = random.randint(0, len(bits_in_base)-1)
                        bit = bits_in_base.pop(bit_idx)
                        new_mask &= ~(1 << bit)
                else:
                    # Make new_mask a superset of base
                    new_mask = base
                    # Add 1-3 random elements
                    additions = random.randint(1, 3)
                    bits_not_in_base = [b for b in range(n) if not (base >> b) & 1]
                    for _ in range(min(additions, len(bits_not_in_base))):
                        if not bits_not_in_base:
                            break
                        bit_idx = random.randint(0, len(bits_not_in_base)-1)
                        bit = bits_not_in_base.pop(bit_idx)
                        new_mask |= (1 << bit)
            
            # Check if new_mask is already in the family (except at idx)
            if new_mask == old_mask:
                continue
            if new_mask in mask_set:
                continue
            
            # Evaluate the change using incremental scoring
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                mj = masks[j]
                old_comp = 1 if are_comparable(old_mask, mj) else 0
                new_comp = 1 if are_comparable(new_mask, mj) else 0
                delta += (new_comp - old_comp)
            
            if delta > 0:
                mask_set.remove(old_mask)
                masks[idx] = new_mask
                mask_set.add(new_mask)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_masks = list(masks)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 500:
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
                        if new_mask not in mask_set:
                            mask_set.remove(masks[idx2])
                            masks[idx2] = new_mask
                            mask_set.add(new_mask)
                            current_score = score_family(masks)
        
        return best_masks, best_score
    
    # Run multiple restarts
    best_overall = None
    best_overall_score = -1
    
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            init_masks = structured_initial()
        else:
            # Mix of structured and random
            if r % 2 == 0:
                init_masks = structured_initial()
            else:
                # Random initial
                masks = set()
                masks.add(0)
                masks.add((1 << n) - 1)
                while len(masks) < m:
                    size = random.randint(0, n)
                    mask = 0
                    for _ in range(size):
                        bit = random.randint(0, n-1)
                        mask |= (1 << bit)
                    masks.add(mask)
                init_masks = list(masks)
        
        # Run local search
        result_masks, result_score = local_search(init_masks, max_iterations=15000)
        
        if result_score > best_overall_score:
            best_overall_score = result_score
            best_overall = result_masks
    
    if best_overall is None:
        best_overall = structured_initial()
    
    # Convert masks to lists of integers
    result = []
    for mask in best_overall:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    # Ensure exactly m distinct sets
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
