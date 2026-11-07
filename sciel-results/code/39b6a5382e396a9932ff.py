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
        """
        masks = []
        mask_set = set()
        
        # Start with empty set and full set? Or just empty set?
        # Let's start with a random seed to avoid bias, or just 0.
        # Actually, starting with 0 is good because it's comparable with everything.
        # But we need distinct sets.
        
        # Let's pick a random starting point to diversify restarts
        # But for greedy, we want the "best" next element.
        # The best first element doesn't matter much, let's pick 0.
        masks.append(0)
        mask_set.add(0)
        
        while len(masks) < m:
            best_mask = -1
            best_score = -1
            # Iterate through all possible masks not in the set
            # This is O(N * |masks|) per step. 4096 * 40 * 40 steps ~ 6.5M ops. Fast enough.
            for candidate in range(N):
                if candidate in mask_set:
                    continue
                
                # Calculate score gain
                gain = 0
                for existing in masks:
                    if are_comparable(candidate, existing):
                        gain += 1
                
                if gain > best_score:
                    best_score = gain
                    best_mask = candidate
                    
                # Optional: early break if we found a perfect match?
                # Max possible gain is len(masks).
                if best_score == len(masks):
                    break
            
            if best_mask == -1:
                # Fallback if no candidate found (should not happen)
                break
                
            masks.append(best_mask)
            mask_set.add(best_mask)
            
            if time.time() > deadline - 10:
                break

        return masks

    def local_search(masks, max_iterations=20000):
        """Improve the family using local search."""
        masks = list(masks)
        current_score = score_family(masks)
        best_score = current_score
        best_masks = list(masks)
        
        no_improve_count = 0
        iter_count = 0
        mask_set = set(masks)
        
        # Precompute comparability with all masks for speed?
        # Let's just use the fast bit check.
        
        while iter_count < max_iterations:
            if time.time() > deadline - 2:
                break
            iter_count += 1
            
            # Pick a random position to modify
            idx = random.randint(0, m-1)
            old_mask = masks[idx]
            
            # Generate a new candidate mask
            # Strategy: Pick a random mask from the universe
            # Or mutate old_mask
            # Or pick a mask that is comparable with many existing ones
            
            # Let's try a mix:
            # 1. Random mask
            # 2. Mutation of old_mask
            # 3. Mutation of a random other mask
            
            r = random.random()
            if r < 0.3:
                new_mask = random.randint(0, N - 1)
            elif r < 0.6:
                new_mask = old_mask
                # Flip 1-2 bits
                flips = random.randint(1, 2)
                for _ in range(flips):
                    bit = random.randint(0, n-1)
                    new_mask ^= (1 << bit)
            else:
                # Pick a random existing mask and mutate it
                other_idx = random.randint(0, m-1)
                if other_idx == idx:
                    other_idx = (other_idx + 1) % m
                base = masks[other_idx]
                new_mask = base
                flips = random.randint(1, 2)
                for _ in range(flips):
                    bit = random.randint(0, n-1)
                    new_mask ^= (1 << bit)
            
            if new_mask == old_mask:
                continue
            if new_mask in mask_set:
                continue
            
            # Evaluate delta
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
                if no_improve_count > 1000:
                    # Restart or perturb
                    no_improve_count = 0
                    # Perturb
                    for _ in range(random.randint(1, 5)):
                        idx2 = random.randint(0, m-1)
                        new_mask = random.randint(0, N - 1)
                        if new_mask not in mask_set:
                            mask_set.remove(masks[idx2])
                            masks[idx2] = new_mask
                            mask_set.add(new_mask)
                    current_score = score_family(masks)
        
        return best_masks, best_score

    best_overall = None
    best_overall_score = -1
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            init_masks = greedy_initial()
        else:
            # Mix of greedy and random
            if random.random() < 0.5:
                init_masks = greedy_initial()
            else:
                # Random init
                masks = set()
                while len(masks) < m:
                    masks.add(random.randint(0, N - 1))
                init_masks = list(masks)
        
        result_masks, result_score = local_search(init_masks, max_iterations=10000)
        
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
