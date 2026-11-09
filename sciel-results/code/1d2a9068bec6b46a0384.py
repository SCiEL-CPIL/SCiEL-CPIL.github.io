import time
import random

def _bitmask_to_list(mask):
    """Convert bitmask to list of integers."""
    res = []
    i = 0
    while mask:
        if mask & 1:
            res.append(i + 1)
        mask >>= 1
        i += 1
    return res

def _can_add(mask, x):
    """
    Check if x can be added to the set represented by mask.
    x is 1-based.
    Returns True if adding x does not create a forbidden triple.
    """
    # Forbidden triple {a, b, c} if a+b, a+c, b+c are all in S.
    # We add x. New triples must include x.
    # So we check if there exist distinct a, b in S such that 
    # {x, a, b} is a forbidden triple.
    # Sums: x+a, x+b, a+b.
    
    # We can iterate over a in S.
    # If x+a is in S, let c = x+a.
    # Then we need b in S such that x+b in S and a+b in S.
    # And a, b, c must be distinct.
    
    # Optimization: 
    # If x > 64 (x in H):
    #   a, b must be in L (since if a,b in H, a+b > 128, so a+b not in S, safe).
    #   Actually, if a,b in H, a+b > 128, so a+b not in S. So no conflict.
    #   So we only need to check a, b in L.
    # If x <= 64 (x in L):
    #   a, b can be in L or H.
    #   If a, b in H, a+b > 128, safe.
    #   So we only need to check pairs (a,b) where at least one is in L.
    #   i.e., (L,L) and (L,H).

    # Let's iterate over all a in S.
    # To be fast, we can precompute lists of L and H elements in S?
    # Or just iterate bits.
    
    # For a given a in S:
    # 1. Check if x+a is in S. If not, continue.
    # 2. Let c = x+a.
    # 3. We need to find b in S, b != a, b != c, such that:
    #    - x+b is in S
    #    - a+b is in S
    
    # Instead of iterating all b for each a, we can do:
    # For each a in S:
    #   if (mask >> (x+a-1)) & 1:
    #      # c = x+a is in S
    #      # We need b in S such that b != a, b != c
    #      # and (mask >> (x+b-1)) & 1 and (mask >> (a+b-1)) & 1
    #      # This is still O(N^2) in worst case if we check all b.
    #      # But N is small (<=128).
    #      # Let's just iterate b in S.
    
    # To speed up, we can note that if x > 64, a and b MUST be <= 64.
    # If x <= 64, a and b can be anything, but if both > 64, safe.
    
    # Let's implement a direct O(N^2) check with bit operations.
    # N is at most 128. 128^2 = 16384 checks. Very fast.
    
    # Iterate a from 1 to 128
    for a in range(1, 129):
        if not (mask & (1 << (a-1))):
            continue
        # a is in S
        # Check if x+a is in S
        sa = x + a
        if sa > 128:
            continue
        if not (mask & (1 << (sa-1))):
            continue
        # c = sa is in S.
        # Now we need b in S, b != a, b != sa
        # such that x+b in S and a+b in S.
        for b in range(1, 129):
            if b == a or b == sa:
                continue
            if not (mask & (1 << (b-1))):
                continue
            # b is in S
            sb = x + b
            if sb > 128:
                continue
            if not (mask & (1 << (sb-1))):
                continue
            sab = a + b
            if sab > 128:
                continue
            if not (mask & (1 << (sab-1))):
                continue
            # Found a conflict: {x, a, b} with sums x+a, x+b, a+b all in S.
            # Note: we must ensure x, a, b are distinct.
            # We checked b != a and b != sa (which is x+a, not x).
            # We need b != x and a != x.
            # a is from S, x is new. So a != x is true.
            # b is from S, so b != x is true.
            # So distinctness is guaranteed.
            return False
            
    return True

def construct():
    deadline = time.time() + 80
    best_mask = 0
    best_size = 0
    
    # Initial heuristic: Start with all H elements
    # H = {65, ..., 128}
    init_mask = 0
    for i in range(65, 129):
        init_mask |= (1 << (i-1))
        
    # Try to add L elements greedily
    current_mask = init_mask
    for x in range(1, 65):
        if time.time() > deadline - 10:
            break
        if _can_add(current_mask, x):
            current_mask |= (1 << (x-1))
            
    if current_mask.bit_count() > best_size:
        best_size = current_mask.bit_count()
        best_mask = current_mask

    # Local Search: Try to remove H elements to allow more L elements
    # Repeat several times to find local optima
    rng = random.Random(42)
    
    for trial in range(1000):
        if time.time() > deadline - 5:
            break
            
        # Start from best_mask or a perturbed version
        # Let's start from best_mask
        cur_mask = best_mask
        
        # Get current L and H
        L = [i for i in range(1, 65) if (cur_mask & (1 << (i-1)))]
        H = [i for i in range(65, 129) if (cur_mask & (1 << (i-1)))]
        
        # Try removing each h in H
        improved = False
        for h in H:
            if time.time() > deadline - 5:
                break
            # Remove h
            new_mask = cur_mask & ~(1 << (h-1))
            
            # Try to add all L elements not in new_mask
            added_count = 0
            for l in range(1, 65):
                if new_mask & (1 << (l-1)):
                    continue
                if _can_add(new_mask, l):
                    new_mask |= (1 << (l-1))
                    added_count += 1
            
            # Net change: -1 (removed h) + added_count
            if added_count > 1:
                # Improvement
                if new_mask.bit_count() > cur_mask.bit_count():
                    cur_mask = new_mask
                    improved = True
        
        # Also try adding L elements if space allows (maybe we missed some)
        for l in range(1, 65):
            if cur_mask & (1 << (l-1)):
                continue
            if _can_add(cur_mask, l):
                cur_mask |= (1 << (l-1))
                improved = True
        
        # Try adding H elements?
        # Removing H to add L is the main move. 
        # But maybe we can add H elements that were previously blocked?
        # If we added more L, some H might still be blocked.
        # If we removed H, we might be able to add other H?
        # H elements don't conflict with each other.
        # So if an H element is not in cur_mask, it must be blocked by L elements.
        # If we just added more L, it's unlikely to unblock H.
        # If we removed H, it doesn't unblock other H.
        
        if cur_mask.bit_count() > best_size:
            best_size = cur_mask.bit_count()
            best_mask = cur_mask
        
        # If no improvement, perturb
        if not improved:
            # Randomly remove a few elements and re-greedy
            if trial > 50:
                # Perturb best_mask
                cur_mask = best_mask
                # Remove 1-3 random elements
                elems = _bitmask_to_list(cur_mask)
                for _ in range(rng.randint(1, 3)):
                    if elems:
                        rem = rng.choice(elems)
                        cur_mask &= ~(1 << (rem-1))
                        elems.remove(rem)
                
                # Re-greedy add
                # Try all elements, add if possible
                # Order: random or sorted?
                # Sorted by value?
                candidates = list(range(1, 129))
                rng.shuffle(candidates)
                for x in candidates:
                    if cur_mask & (1 << (x-1)):
                        continue
                    if _can_add(cur_mask, x):
                        cur_mask |= (1 << (x-1))
                
                if cur_mask.bit_count() > best_size:
                    best_size = cur_mask.bit_count()
                    best_mask = cur_mask

    result = _bitmask_to_list(best_mask)
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
