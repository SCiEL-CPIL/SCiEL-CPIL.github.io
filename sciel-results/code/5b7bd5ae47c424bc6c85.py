import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 88  # 2s safety margin for 90s limit
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute popcount for all masks
    popcount = [0] * N
    for mask in range(1, N):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
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
    
    def greedy_initial():
        """
        Construct a family by greedily adding the mask that maximizes
        the number of comparable pairs with the current family.
        Uses vectorized gain computation for speed.
        """
        masks = []
        mask_set = set()
        
        # Start with empty set (comparable with everything)
        masks.append(0)
        mask_set.add(0)
        
        # Precompute comparability table for all pairs of masks
        # comp[a][b] = 1 if a and b are comparable, else 0
        # This is a 4096x4096 table = 16M entries, too large.
        # Instead, compute gain on the fly but more efficiently.
        
        while len(masks) < m:
            best_mask = -1
            best_score = -1
            
            # For each candidate, compute how many existing masks it's comparable with
            # Vectorize: for each candidate, check against all existing masks
            for candidate in range(N):
                if candidate in mask_set:
                    continue
                
                # Calculate score gain using bit operations
                gain = 0
                for existing in masks:
                    # (a & b) == a means a subset b
                    # (a & b) == b means b subset a
                    if (candidate & existing) == candidate or (candidate & existing) == existing:
                        gain += 1
                
                if gain > best_score:
                    best_score = gain
                    best_mask = candidate
                    
                # Early termination: max possible gain is len(masks)
                if best_score == len(masks):
                    break
            
            if best_mask == -1:
                break
                
            masks.append(best_mask)
            mask_set.add(best_mask)
            
            if time.time() > deadline - 12:
                break

        return masks

    def local_search(masks, max_iterations=20000):
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
            
            # Generate a new candidate mask with multiple strategies
            r = random.random()
            if r < 0.25:
                # Pure random
                new_mask = random.randint(0, N - 1)
            elif r < 0.50:
                # Mutation of old_mask: flip 1-3 bits
                new_mask = old_mask
                flips = random.randint(1, 3)
                bits_to_flip = random.sample(range(n), min(flips, n))
                for bit in bits_to_flip:
                    new_mask ^= (1 << bit)
            elif r < 0.70:
                # Add or remove a single element from old_mask
                if old_mask == 0:
                    # Can only add
                    bit = random.randint(0, n-1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == N - 1:
                    # Can only remove
                    bit = random.randint(0, n-1)
                    new_mask = old_mask & ~(1 << bit)
                else:
                    if random.random() < 0.5:
                        # Add an element not in old_mask
                        missing_bits = [b for b in range(n) if not (old_mask >> b) & 1]
                        if missing_bits:
                            bit = random.choice(missing_bits)
                            new_mask = old_mask | (1 << bit)
                        else:
                            new_mask = old_mask
                    else:
                        # Remove an element from old_mask
                        present_bits = [b for b in range(n) if (old_mask >> b) & 1]
                        if present_bits:
                            bit = random.choice(present_bits)
                            new_mask = old_mask & ~(1 << bit)
                        else:
                            new_mask = old_mask
            elif r < 0.85:
                # Swap: take a random existing mask and modify it
                other_idx = random.randint(0, m-1)
                while other_idx == idx:
                    other_idx = random.randint(0, m-1)
                base = masks[other_idx]
                new_mask = base
                flips = random.randint(1, 2)
                bits_to_flip = random.sample(range(n), min(flips, n))
                for bit in bits_to_flip:
                    new_mask ^= (1 << bit)
            else:
                # Chain-based: find a mask that extends or is extended by old_mask
                # Try to find a mask that is a superset or subset of old_mask
                if random.random() < 0.5:
                    # Try to find a superset
                    new_mask = old_mask
                    # Add 1-2 random bits
                    bits_to_add = random.sample(range(n), random.randint(1, 2))
                    for bit in bits_to_add:
                        new_mask |= (1 << bit)
                else:
                    # Try to find a subset
                    new_mask = old_mask
                    # Remove 1-2 random bits
                    present_bits = [b for b in range(n) if (old_mask >> b) & 1]
                    if present_bits:
                        bits_to_remove = random.sample(present_bits, min(2, len(present_bits)))
                        for bit in bits_to_remove:
                            new_mask &= ~(1 << bit)
            
            if new_mask == old_mask:
                continue
            if new_mask in mask_set:
                continue
            
            # Evaluate delta efficiently
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                mj = masks[j]
                old_comp = (old_mask & mj) == old_mask or (old_mask & mj) == mj
                new_comp = (new_mask & mj) == new_mask or (new_mask & mj) == mj
                delta += (1 if new_comp else 0) - (1 if old_comp else 0)
            
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
                    # Perturb to escape local optimum
                    no_improve_count = 0
                    num_perturb = random.randint(2, 5)
                    for _ in range(num_perturb):
                        idx2 = random.randint(0, m-1)
                        new_mask = random.randint(0, N - 1)
                        if new_mask not in mask_set and new_mask != masks[idx2]:
                            mask_set.remove(masks[idx2])
                            masks[idx2] = new_mask
                            mask_set.add(new_mask)
                    current_score = score_family(masks)
        
        return best_masks, best_score

    best_overall = None
    best_overall_score = -1
    
    num_restarts = 25
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            # First restart: pure greedy
            init_masks = greedy_initial()
        elif r < 5:
            # Next few: greedy with slight randomization
            init_masks = greedy_initial()
            # Randomly swap 1-2 elements
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, m-1)
                new_mask = random.randint(0, N - 1)
                if new_mask not in set(init_masks):
                    init_masks[idx] = new_mask
        elif r < 12:
            # Mix: 50% greedy, 50% chain-based random
            if random.random() < 0.5:
                init_masks = greedy_initial()
            else:
                # Chain-based initialization: start with a chain and add random elements
                init_masks = []
                mask_set = set()
                # Create a random chain
                current = 0
                for _ in range(random.randint(5, 15)):
                    if current not in mask_set:
                        init_masks.append(current)
                        mask_set.add(current)
                    # Add a random bit to extend the chain
                    missing_bits = [b for b in range(n) if not (current >> b) & 1]
                    if missing_bits:
                        bit = random.choice(missing_bits)
                        current |= (1 << bit)
                # Fill remaining with random masks
                while len(init_masks) < m:
                    new_mask = random.randint(0, N - 1)
                    if new_mask not in mask_set:
                        init_masks.append(new_mask)
                        mask_set.add(new_mask)
        else:
            # Later restarts: more diverse random inits
            if random.random() < 0.3:
                init_masks = greedy_initial()
            else:
                # Random init with size distribution biased toward middle sizes
                masks = set()
                while len(masks) < m:
                    # Bias toward sizes 3-8 (middle of the poset)
                    size = random.randint(3, 8)
                    s = random.sample(range(n), size)
                    mask = 0
                    for b in s:
                        mask |= (1 << b)
                    if mask not in masks:
                        masks.add(mask)
                init_masks = list(masks)
        
        # Use more iterations for earlier restarts (they have better starts)
        if r < 5:
            max_iter = 15000
        else:
            max_iter = 10000
            
        result_masks, result_score = local_search(init_masks, max_iterations=max_iter)
        
        if result_score > best_overall_score:
            best_overall_score = result_score
            best_overall = result_masks
    
    if best_overall is None:
        best_overall = greedy_initial()
    
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
