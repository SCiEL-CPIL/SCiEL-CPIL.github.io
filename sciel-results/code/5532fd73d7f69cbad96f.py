"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add_bitmask(S_mask, x):
    """Check if adding x to the set represented by S_mask creates no forbidden triple.
    S_mask is an int bitmask where bit i is set if (i+1) is in the set.
    x is in range 1..128.
    """
    n = 128
    # Precompute: for each pair (a,b) in S, check if a+b, a+x, b+x conditions are met
    # Triple (a, b, x): need a+b in S or a+b==x, AND a+x in S, AND b+x in S
    # Triple (a, b, c) in S where one sum equals x:
    #   a+b == x: need a+c in S and b+c in S for some c in S, c != a, c != b
    
    # Check type 1: triples involving x
    # For each a in S: xa = x+a. If xa in S (xa <= 128), then for each b in S, b != a:
    #   xb = x+b. If xb in S (xb <= 128), then check a+b: if a+b in S or a+b == x -> invalid
    for a in range(1, n+1):
        if not (S_mask >> a) & 1:
            continue
        xa = x + a
        if xa > n:
            continue
        if not (S_mask >> xa) & 1:
            continue
        # xa is in S, so for each b in S, b != a
        for b in range(1, n+1):
            if b == a or not (S_mask >> b) & 1:
                continue
            xb = x + b
            if xb > n:
                continue
            if not (S_mask >> xb) & 1:
                continue
            ab = a + b
            if ab == x:
                return False
            if ab <= n and (S_mask >> ab) & 1:
                return False
    
    # Check type 2: triples in S where x completes a sum
    # Find pairs (a, b) in S with a < b and a + b == x
    for a in range(1, x//2 + 1):
        b = x - a
        if b <= a or b > n:
            continue
        if not ((S_mask >> a) & 1) or not ((S_mask >> b) & 1):
            continue
        # For each c in S, c != a, c != b: check a+c in S and b+c in S
        for c in range(1, n+1):
            if c == a or c == b or not (S_mask >> c) & 1:
                continue
            ac = a + c
            bc = b + c
            if ac > n or bc > n:
                continue
            if (S_mask >> ac) & 1 and (S_mask >> bc) & 1:
                return False
    
    return True

def _is_valid_bitmask(S_mask):
    """Check if the set represented by S_mask has no forbidden triple."""
    n = 128
    for a in range(1, n+1):
        if not (S_mask >> a) & 1:
            continue
        for b in range(a+1, n+1):
            if not (S_mask >> b) & 1:
                continue
            ab = a + b
            if ab > n or not (S_mask >> ab) & 1:
                continue
            for c in range(b+1, n+1):
                if not (S_mask >> c) & 1:
                    continue
                ac = a + c
                bc = b + c
                if ac <= n and (S_mask >> ac) & 1 and bc <= n and (S_mask >> bc) & 1:
                    return False
    return True

def _greedy_add_bitmask(base_mask, candidates, deadline):
    """Greedily add candidates to base_mask, checking validity."""
    current = base_mask
    for x in candidates:
        if time.time() > deadline - 3:
            break
        if not (current >> x) & 1:
            if _can_add_bitmask(current, x):
                current |= (1 << x)
    return current

def _local_search_bitmask(S_mask, deadline, max_iters=5000):
    """Try to improve by removing one element and adding one or two others."""
    current = S_mask
    n = 128
    improved = True
    iter_count = 0
    while improved and time.time() < deadline - 5:
        improved = False
        iter_count += 1
        if iter_count > max_iters:
            break
        # Try all single swaps: remove r, add a
        for r in range(1, n+1):
            if time.time() > deadline - 5:
                break
            if not (current >> r) & 1:
                continue
            without = current & ~(1 << r)
            for a in range(1, n+1):
                if (without >> a) & 1:
                    continue
                if _can_add_bitmask(without, a):
                    new_set = without | (1 << a)
                    if bin(new_set).count('1') > bin(current).count('1'):
                        current = new_set
                        improved = True
                        break
            if improved:
                break
    return current

def _local_search_2bit_swap(S_mask, deadline):
    """Try to improve by removing two elements and adding two others."""
    current = S_mask
    n = 128
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try all pairs (r1, r2) to remove, then try all pairs (a1, a2) to add
        for r1 in range(1, n+1):
            if time.time() > deadline - 5:
                break
            if not (current >> r1) & 1:
                continue
            for r2 in range(r1+1, n+1):
                if not (current >> r2) & 1:
                    continue
                without = current & ~(1 << r1) & ~(1 << r2)
                for a1 in range(1, n+1):
                    if time.time() > deadline - 5:
                        break
                    if (without >> a1) & 1:
                        continue
                    if not _can_add_bitmask(without, a1):
                        continue
                    temp = without | (1 << a1)
                    for a2 in range(a1+1, n+1):
                        if (temp >> a2) & 1:
                            continue
                        if _can_add_bitmask(temp, a2):
                            new_set = temp | (1 << a2)
                            if bin(new_set).count('1') > bin(current).count('1'):
                                current = new_set
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return current

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    n = 128
    
    best_size = 0
    best_mask = 0
    rng = random.Random(42)
    
    # Try various base ranges: from {65..128} down to {45..128}
    for start in range(65, 44, -1):
        if time.time() > deadline - 10:
            break
        base_mask = 0
        for i in range(start, n+1):
            base_mask |= (1 << i)
        candidates = list(range(1, start))
        
        # Greedy add in natural order
        cur = _greedy_add_bitmask(base_mask, candidates, deadline)
        sz = bin(cur).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = cur
        
        # Try several random orderings
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            order = list(range(1, start))
            rng.shuffle(order)
            cur = _greedy_add_bitmask(base_mask, order, deadline)
            sz = bin(cur).count('1')
            if sz > best_size:
                best_size = sz
                best_mask = cur
        
        # Local search on best found for this base
        if time.time() < deadline - 10:
            cur_ls = _local_search_bitmask(best_mask, deadline)
            sz = bin(cur_ls).count('1')
            if sz > best_size:
                best_size = sz
                best_mask = cur_ls
    
    # Also try starting with just high numbers and being more aggressive
    for start in range(70, 85):
        if time.time() > deadline - 5:
            break
        base_mask = 0
        for i in range(start, n+1):
            base_mask |= (1 << i)
        candidates = list(range(1, start))
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            order = list(range(1, start))
            rng.shuffle(order)
            cur = _greedy_add_bitmask(base_mask, order, deadline)
            sz = bin(cur).count('1')
            if sz > best_size:
                best_size = sz
                best_mask = cur
    
    # Final local search
    if time.time() < deadline - 5:
        cur_ls = _local_search_bitmask(best_mask, deadline)
        sz = bin(cur_ls).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = cur_ls
    
    # Try 2-bit swap local search if time permits
    if time.time() < deadline - 10:
        cur_ls2 = _local_search_2bit_swap(best_mask, deadline)
        sz = bin(cur_ls2).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = cur_ls2
    
    # Extract result
    result = [i for i in range(1, n+1) if (best_mask >> i) & 1]
    # Verify final result
    assert _is_valid_bitmask(best_mask), "Final result is invalid!"
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
