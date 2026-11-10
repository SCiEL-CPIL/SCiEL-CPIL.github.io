import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    N = 256
    # Perfect squares up to 512
    squares = set()
    i = 1
    while i * i <= 512:
        squares.add(i * i)
        i += 1
    
    # Build conflict graph using bitmasks
    n = N
    conflicts = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(i, n):
            b = j + 1
            if (a + b) in squares:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    def popcount(x):
        """Count bits in integer."""
        return bin(x).count('1')
    
    def greedy_fill(mask):
        """Greedily add any vertex that doesn't conflict."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not ((mask >> i) & 1):
                    if not (mask & conflicts[i]):
                        mask |= (1 << i)
                        changed = True
        return mask
    
    def local_search(mask, max_iters=100):
        """Improve mask via 1-swap and 2-swap moves."""
        current = mask
        current_size = popcount(current)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try 1-remove, 1-add moves that increase size
            for rem in range(n):
                if not ((current >> rem) & 1):
                    continue
                reduced = current & ~(1 << rem)
                for add in range(n):
                    if (reduced >> add) & 1:
                        continue
                    if reduced & conflicts[add]:
                        continue
                    candidate = reduced | (1 << add)
                    if popcount(candidate) > current_size:
                        current = candidate
                        current_size = popcount(candidate)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try 2-remove, 2-add moves
                found = False
                for rem1 in range(n):
                    if not ((current >> rem1) & 1):
                        continue
                    for rem2 in range(rem1 + 1, n):
                        if not ((current >> rem2) & 1):
                            continue
                        reduced = current & ~(1 << rem1) & ~(1 << rem2)
                        for add1 in range(n):
                            if (reduced >> add1) & 1:
                                continue
                            if reduced & conflicts[add1]:
                                continue
                            for add2 in range(add1 + 1, n):
                                if (reduced >> add2) & 1:
                                    continue
                                if (reduced & conflicts[add2]) or ((1 << add1) & conflicts[add2]):
                                    continue
                                candidate = reduced | (1 << add1) | (1 << add2)
                                if popcount(candidate) > current_size:
                                    current = candidate
                                    current_size = popcount(candidate)
                                    found = True
                                    improved = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if not improved:
                    break
        return current, current_size
    
    def perturbed_local_search(seed_mask, perturb_ratio=0.1):
        """Apply perturbation then local search."""
        mask = seed_mask
        size = popcount(mask)
        if size == 0:
            return mask, 0
        
        # Randomly remove some elements
        elements = [i for i in range(n) if (mask >> i) & 1]
        num_remove = max(1, int(size * perturb_ratio))
        to_remove = random.sample(elements, min(num_remove, len(elements)))
        for i in to_remove:
            mask &= ~(1 << i)
        
        # Greedy fill
        mask = greedy_fill(mask)
        
        # Local search
        mask, size = local_search(mask)
        return mask, size
    
    # Initialize with multiple strategies
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Numbers ≡ 1 mod 3
    mask1 = 0
    for i in range(n):
        if (i + 1) % 3 == 1:
            mask1 |= (1 << i)
    mask1 = greedy_fill(mask1)
    mask1, size1 = local_search(mask1)
    if size1 > best_size:
        best_mask = mask1
        best_size = size1
    
    # Strategy 2: Numbers ≡ 2 mod 3
    mask2 = 0
    for i in range(n):
        if (i + 1) % 3 == 2:
            mask2 |= (1 << i)
    mask2 = greedy_fill(mask2)
    mask2, size2 = local_search(mask2)
    if size2 > best_size:
        best_mask = mask2
        best_size = size2
    
    # Strategy 3: Numbers ≡ 0 mod 4
    mask3 = 0
    for i in range(n):
        if (i + 1) % 4 == 0:
            mask3 |= (1 << i)
    mask3 = greedy_fill(mask3)
    mask3, size3 = local_search(mask3)
    if size3 > best_size:
        best_mask = mask3
        best_size = size3
    
    # Strategy 4: Pure greedy from empty
    mask4 = greedy_fill(0)
    mask4, size4 = local_search(mask4)
    if size4 > best_size:
        best_mask = mask4
        best_size = size4
    
    # Iterated local search with perturbations
    random.seed(42)
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Perturb current best
        perturb_ratio = 0.05 + 0.1 * (iteration % 5) / 5
        new_mask, new_size = perturbed_local_search(best_mask, perturb_ratio)
        
        if new_size > best_size:
            best_mask = new_mask
            best_size = new_size
    
    # Final greedy pass
    final_mask = greedy_fill(best_mask)
    final_mask, final_size = local_search(final_mask)
    if final_size > best_size:
        best_mask = final_mask
        best_size = final_size
    
    # Convert back to list
    result = [i + 1 for i in range(n) if (best_mask >> i) & 1]
    
    # Verify
    sq_set = squares
    valid = True
    for idx, a in enumerate(result):
        for b in result[idx:]:
            if (a + b) in sq_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to known good
        result = [i + 1 for i in range(n) if (i + 1) % 3 == 1]
        for idx, a in enumerate(result):
            for b in result[idx:]:
                if (a + b) in sq_set:
                    return [1]  # Minimal valid set
    
    return result
