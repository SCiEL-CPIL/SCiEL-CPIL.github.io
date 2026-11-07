import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128
    all_elements = list(range(1, N + 1))
    
    best_set = set(range(N//2 + 1, N + 1))
    best_size = len(best_set)
    
    def is_valid_add(S_set, S_mask, x):
        """Check if adding x to S_set keeps validity."""
        # For each pair (a,b) in S_set, check if a+b, a+x, b+x are all in S_set∪{x}
        # Since x is new, a+x and b+x are new (not in S_set unless they equal x, but a,b >= 1 so a+x > x)
        # Actually a+x > x always for a>=1, so a+x and b+x are NOT in S_set
        # So the only way adding x creates a forbidden triple involving x is if
        # for some a,b in S_set: a+b in S_set AND a+x in S_set AND b+x in S_set
        # But a+x > x, so a+x can only be in S_set if it was already there
        # Wait, we're checking if the triple (a,b,x) would be forbidden:
        # need a+b in S, a+x in S, b+x in S
        # a+x > x, b+x > x, so they must be in S_set (already present)
        # a+b could be in S_set
        for a in S_set:
            if a >= x:
                continue
            for b in S_set:
                if b <= a or b >= x:
                    continue
                ab = a + b
                ax = a + x
                bx = b + x
                if ab > N or ax > N or bx > N:
                    continue
                if ((S_mask >> (ab-1)) & 1) and ((S_mask >> (ax-1)) & 1) and ((S_mask >> (bx-1)) & 1):
                    return False
        return True
    
    def greedy_random():
        elems = list(all_elements)
        random.shuffle(elems)
        S = set()
        S_mask = 0
        for x in elems:
            if is_valid_add(S, S_mask, x):
                S.add(x)
                S_mask |= (1 << (x-1))
        return S, S_mask
    
    def local_search_add(S, S_mask):
        """Try to add any element not in S."""
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for x in all_elements:
                if x in S:
                    continue
                if is_valid_add(S, S_mask, x):
                    S.add(x)
                    S_mask |= (1 << (x-1))
                    changed = True
        return S, S_mask
    
    # Multiple restarts
    for r in range(80):
        if time.time() > deadline - 3.0:
            break
        S, S_mask = greedy_random()
        S, S_mask = local_search_add(S, S_mask)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Final: try adding to best_set
    if time.time() < deadline - 1.0:
        S = set(best_set)
        S_mask = 0
        for x in S:
            S_mask |= (1 << (x-1))
        for x in all_elements:
            if x in S:
                continue
            if is_valid_add(S, S_mask, x):
                S.add(x)
                S_mask |= (1 << (x-1))
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    return sorted(best_set)
