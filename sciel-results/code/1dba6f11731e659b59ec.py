"""Optimized construction of a 4-AP-free subset of [1,256] maximizing 3-APs."""
import time
import random
import numpy as np

N = 256
DEADLINE_MARGIN = 5.0  # seconds to spare before the 60s limit

def count_3aps(mask):
    """Count distinct 3-term arithmetic progressions in a bitmask of [1,256]."""
    count = 0
    n = N
    # For each pair (a, c) with a < c and same parity, check if b = (a+c)//2 is in set
    # More efficiently: iterate over (a, d) where a+d < n and a+2d < n
    for a in range(1, n + 1):
        if not (mask >> (a - 1)) & 1:
            continue
        for d in range(1, (n - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > n:
                break
            if (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1:
                count += 1
    return count

def has_4ap(mask, x):
    """Check if adding element x (1-indexed) to mask creates a 4-AP."""
    bit = 1 << (x - 1)
    new_mask = mask | bit
    # Check all 4-APs involving x
    # x can be a, a+d, a+2d, or a+3d in a 4-AP (a, a+d, a+2d, a+3d)
    # Case 1: x = a (first term): need a+d, a+2d, a+3d all in new_mask
    for d in range(1, (N - x) // 3 + 1):
        if ((new_mask >> (x + d - 1)) & 1) and ((new_mask >> (x + 2*d - 1)) & 1) and ((new_mask >> (x + 3*d - 1)) & 1):
            return True
    # Case 2: x = a+d (second term): a = x-d, a+2d = x+d, a+3d = x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        a = x - d
        if a < 1:
            continue
        if ((new_mask >> (a - 1)) & 1) and ((new_mask >> (x + d - 1)) & 1) and ((new_mask >> (x + 2*d - 1)) & 1):
            return True
    # Case 3: x = a+2d (third term): a = x-2d, a+d = x-d, a+3d = x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        a = x - 2 * d
        if a < 1:
            continue
        if ((new_mask >> (a - 1)) & 1) and ((new_mask >> (x - d - 1)) & 1) and ((new_mask >> (x + d - 1)) & 1):
            return True
    # Case 4: x = a+3d (fourth term): a = x-3d, a+d = x-2d, a+2d = x-d
    for d in range(1, (x - 1) // 3 + 1):
        a = x - 3 * d
        if a < 1:
            continue
        if ((new_mask >> (a - 1)) & 1) and ((new_mask >> (x - 2*d - 1)) & 1) and ((new_mask >> (x - d - 1)) & 1):
            return True
    return False

def delta_3aps(mask, x):
    """Count how many new 3-APs are created by adding x to mask."""
    new_aps = 0
    # x as first term: (x, x+d, x+2d)
    for d in range(1, (N - x) // 2 + 1):
        if ((mask >> (x + d - 1)) & 1) and ((mask >> (x + 2*d - 1)) & 1):
            new_aps += 1
    # x as middle term: (x-d, x, x+d)
    for d in range(1, min(x - 1, N - x) + 1):
        if ((mask >> (x - d - 1)) & 1) and ((mask >> (x + d - 1)) & 1):
            new_aps += 1
    # x as last term: (x-2d, x-d, x)
    for d in range(1, (x - 1) // 2 + 1):
        if ((mask >> (x - 2*d - 1)) & 1) and ((mask >> (x - d - 1)) & 1):
            new_aps += 1
    return new_aps

def greedy_construct(order):
    """Greedy construction given an ordering of elements."""
    mask = 0
    for x in order:
        if not has_4ap(mask, x):
            mask |= (1 << (x - 1))
    return mask

def local_search(mask, max_iter=5000):
    """Hill climbing: try adding elements, then removing/replacing."""
    best_mask = mask
    best_score = count_3aps(mask)
    
    for iteration in range(max_iter):
        improved = False
        
        # Try adding each element not in mask
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                continue
            if has_4ap(mask, x):
                continue
            gain = delta_3aps(mask, x)
            if gain > 0:
                new_mask = mask | (1 << (x - 1))
                new_score = best_score + gain
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    mask = new_mask
                    improved = True
                    break
        
        if not improved:
            # Try single removals that allow better re-additions
            for x in range(1, N + 1):
                if not (mask >> (x - 1)) & 1:
                    continue
                # Try removing x
                mask_without = mask & ~(1 << (x - 1))
                # See if we can add multiple elements
                added = 0
                new_mask = mask_without
                for y in range(1, N + 1):
                    if (new_mask >> (y - 1)) & 1:
                        continue
                    if y == x:
                        continue
                    if not has_4ap(new_mask, y):
                        new_mask |= (1 << (y - 1))
                        added += 1
                if added > 1:
                    new_score = count_3aps(new_mask)
                    if new_score > best_score:
                        best_mask = new_mask
                        best_score = new_score
                        mask = new_mask
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    
    # Multiple greedy orderings
    best_mask = 0
    best_score = 0
    
    # Ordering 1: natural
    mask1 = greedy_construct(list(range(1, N + 1)))
    s1 = count_3aps(mask1)
    if s1 > best_score:
        best_mask, best_score = mask1, s1
    
    # Ordering 2: reverse
    mask2 = greedy_construct(list(range(N, 0, -1)))
    s2 = count_3aps(mask2)
    if s2 > best_score:
        best_mask, best_score = mask2, s2
    
    # Ordering 3: odd first
    mask3 = greedy_construct(list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)))
    s3 = count_3aps(mask3)
    if s3 > best_score:
        best_mask, best_score = mask3, s3
    
    # Ordering 4: even first
    mask4 = greedy_construct(list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)))
    s4 = count_3aps(mask4)
    if s4 > best_score:
        best_mask, best_score = mask4, s4
    
    # Ordering 5: by residue mod 3
    r0 = list(range(3, N + 1, 3))
    r1 = list(range(1, N + 1, 3))
    r2 = list(range(2, N + 1, 3))
    for perm in [r0+r1+r2, r1+r0+r2, r2+r0+r1, r0+r2+r1, r1+r2+r0, r2+r1+r0]:
        m = greedy_construct(perm)
        s = count_3aps(m)
        if s > best_score:
            best_mask, best_score = m, s
    
    # Local search on best
    if time.time() - start_time < DEADLINE_MARGIN:
        best_mask, best_score = local_search(best_mask, max_iter=3000)
    
    # Try random restarts if time permits
    rng = random.Random(42)
    attempts = 0
    while time.time() - start_time < DEADLINE_MARGIN:
        # Random perturbation of best_mask
        perturbed = best_mask
        # Remove a random element
        bits = [i + 1 for i in range(N) if (perturbed >> i) & 1]
        if bits:
            x = rng.choice(bits)
            perturbed &= ~(1 << (x - 1))
        
        # Re-add greedily
        for x in range(1, N + 1):
            if (perturbed >> (x - 1)) & 1:
                continue
            if not has_4ap(perturbed, x):
                perturbed |= (1 << (x - 1))
        
        s = count_3aps(perturbed)
        if s > best_score:
            best_mask, best_score = perturbed, s
        
        # Local search on improved
        if s >= best_score - 2:
            lm, ls = local_search(perturbed, max_iter=1000)
            if ls > best_score:
                best_mask, best_score = lm, ls
        
        attempts += 1
        if attempts >= 30:
            break
    
    # Convert mask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return sorted(result)
