"""Bitmask-based local search for 3-AP maximization with 4-AP-free constraint."""
import time

N = 256
DEADLINE = time.time() + 85.0  # safety margin before 90s kill

def _bits_to_set(mask):
    """Convert bitmask to sorted list of 1-indexed values."""
    return [i+1 for i in range(N) if (mask >> i) & 1]

def _count_3ap(mask):
    """Count distinct 3-term APs (positive diff, counted once: a, a+d, a+2d)."""
    count = 0
    for a in range(N):
        if not (mask >> a) & 1:
            continue
        for d in range(1, (N - a) // 2 + 1):
            a2 = a + d
            a3 = a + 2*d
            if a3 >= N:
                break
            if (mask >> a2) & 1 and (mask >> a3) & 1:
                count += 1
    return count

def _has_4ap(mask):
    """Check if mask contains any nonconstant 4-term AP."""
    for a in range(N):
        if not (mask >> a) & 1:
            continue
        for d in range(1, (N - a) // 4 + 1):
            a2 = a + d
            a3 = a + 2*d
            a4 = a + 3*d
            if a4 >= N:
                break
            if (mask >> a2) & 1 and (mask >> a3) & 1 and (mask >> a4) & 1:
                return True
    return False

def _add_4ap_violations(mask, pos):
    """Return set of 4-APs that would be created by adding 'pos' (0-indexed).
    Only checks 4-APs where 'pos' is the new element."""
    violations = []
    p = pos
    # pos could be a, a+d, a+2d, or a+3d in a 4-AP
    # Case: pos = a (start)
    for d in range(1, (N - p) // 4 + 1):
        if (mask >> (p+d)) & 1 and (mask >> (p+2*d)) & 1 and (mask >> (p+3*d)) & 1:
            violations.append((p, d))
    # Case: pos = a+d (2nd term)
    for d in range(1, p//1 + 1):
        a = p - d
        if a < 0:
            break
        a2 = p + d
        a3 = p + 2*d
        if a3 >= N:
            break
        if (mask >> a) & 1 and (mask >> a2) & 1 and (mask >> a3) & 1:
            violations.append((a, d))
    # Case: pos = a+2d (3rd term)
    for d in range(1, p//2 + 1):
        a = p - 2*d
        if a < 0:
            break
        a1 = p - d
        a3 = p + d
        if a3 >= N:
            break
        if (mask >> a) & 1 and (mask >> a1) & 1 and (mask >> a3) & 1:
            violations.append((a, d))
    # Case: pos = a+3d (4th term)
    for d in range(1, p//3 + 1):
        a = p - 3*d
        if a < 0:
            break
        a1 = p - 2*d
        a2 = p - d
        if (mask >> a) & 1 and (mask >> a1) & 1 and (mask >> a2) & 1:
            violations.append((a, d))
    return violations

def _remove_4ap_violations(mask, pos):
    """Check if removing 'pos' would break any existing 4-AP (i.e., pos is part of a 4-AP)."""
    p = pos
    # Check all 4-APs containing pos
    # pos = a
    for d in range(1, (N - p) // 4 + 1):
        if (mask >> (p+d)) & 1 and (mask >> (p+2*d)) & 1 and (mask >> (p+3*d)) & 1:
            return True
    # pos = a+d
    for d in range(1, p+1):
        a = p - d
        if a < 0:
            break
        a2 = p + d
        a3 = p + 2*d
        if a3 >= N:
            break
        if (mask >> a) & 1 and (mask >> a2) & 1 and (mask >> a3) & 1:
            return True
    # pos = a+2d
    for d in range(1, p//2 + 1):
        a = p - 2*d
        if a < 0:
            break
        a1 = p - d
        a3 = p + d
        if a3 >= N:
            break
        if (mask >> a) & 1 and (mask >> a1) & 1 and (mask >> a3) & 1:
            return True
    # pos = a+3d
    for d in range(1, p//3 + 1):
        a = p - 3*d
        if a < 0:
            break
        a1 = p - 2*d
        a2 = p - d
        if (mask >> a) & 1 and (mask >> a1) & 1 and (mask >> a2) & 1:
            return True
    return False

def _greedy_order(order):
    """Greedy construction using a specific ordering of elements."""
    mask = 0
    for pos in order:
        # Check if adding pos creates a 4-AP
        if not _add_4ap_violations(mask, pos):
            mask |= (1 << pos)
    return mask

def _local_search(mask, max_iters=5000):
    """Local search: try adding, removing, or swapping elements to improve 3-AP count."""
    best_mask = mask
    best_score = _count_3ap(mask)
    
    for iteration in range(max_iters):
        if time.time() > DEADLINE - 2.0:
            break
        
        improved = False
        current_score = _count_3ap(best_mask)
        
        # Try adding elements
        for pos in range(N):
            if time.time() > DEADLINE - 2.0:
                break
            if (best_mask >> pos) & 1:
                continue
            # Check 4-AP constraint
            if _add_4ap_violations(best_mask, pos):
                continue
            new_mask = best_mask | (1 << pos)
            new_score = _count_3ap(new_mask)
            if new_score > current_score:
                best_mask = new_mask
                current_score = new_score
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing elements that allow adding more
        for pos in range(N):
            if time.time() > DEADLINE - 2.0:
                break
            if not (best_mask >> pos) & 1:
                continue
            # Check if pos is part of a 4-AP (removing it might help)
            if not _remove_4ap_violations(best_mask, pos):
                continue
            new_mask = best_mask & ~(1 << pos)
            # Try to add other elements
            for add_pos in range(N):
                if (new_mask >> add_pos) & 1:
                    continue
                if _add_4ap_violations(new_mask, add_pos):
                    continue
                swap_mask = new_mask | (1 << add_pos)
                swap_score = _count_3ap(swap_mask)
                if swap_score > current_score:
                    best_mask = swap_mask
                    current_score = swap_score
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_mask, _count_3ap(best_mask)

def construct():
    import random
    random.seed(42)
    
    best_mask = 0
    best_score = 0
    
    # Multiple greedy starts with different orderings
    orderings = [
        list(range(N)),          # natural order
        list(range(N-1, -1, -1)), # reverse
        list(range(0, N, 2)) + list(range(1, N, 2)),  # evens then odds
        list(range(1, N, 2)) + list(range(0, N, 2)),  # odds then evens
    ]
    
    # Add some random orderings
    for _ in range(10):
        o = list(range(N))
        random.shuffle(o)
        orderings.append(o)
    
    for order in orderings:
        if time.time() > DEADLINE - 5.0:
            break
        mask = _greedy_order(order)
        score = _count_3ap(mask)
        if score > best_score:
            best_mask = mask
            best_score = score
    
    # Local search from best starting point
    best_mask, best_score = _local_search(best_mask, max_iters=20000)
    
    # Try local search from a few other good starting points
    for order in orderings[:5]:
        if time.time() > DEADLINE - 5.0:
            break
        if order == orderings[0]:  # skip the one we already did
            continue
        mask = _greedy_order(order)
        score = _count_3ap(mask)
        if score >= best_score - 20:  # only if competitive
            mask, score = _local_search(mask, max_iters=10000)
            if score > best_score:
                best_mask = mask
                best_score = score
    
    # Final deadline check
    if time.time() > DEADLINE - 1.0:
        return _bits_to_set(best_mask)
    
    # One more round of local search if time permits
    best_mask, best_score = _local_search(best_mask, max_iters=15000)
    
    return _bits_to_set(best_mask)
