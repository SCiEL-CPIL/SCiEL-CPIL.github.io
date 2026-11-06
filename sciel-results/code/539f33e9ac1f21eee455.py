"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    n_max = 256
    
    # Precompute conflicts for 'a' role: list of (b, c) pairs
    # conflict[a] contains (b, c) such that a < b <= c and a | (b+c)
    conflict = [[] for _ in range(n_max + 1)]
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            for c in range(b, n_max + 1):
                if (b + c) % a == 0:
                    conflict[a].append((b, c))
    
    # Precompute M[a][x] for a < x
    # M[a][x] is a bitmask of y such that:
    # (y >= x and a | (x+y)) OR (y < x and a | (y+x))
    # Note: if y=x, condition is a | 2x.
    # We want to check if S intersects M[a][x].
    # If y in S and y in M[a][x], then conflict (a, y, x) or (a, x, y) or (a, x, x) exists.
    # Specifically:
    # If y < x: triple is (a, y, x). Need a < y. 
    #   Wait, M[a][x] definition above didn't check a < y.
    #   The condition for conflict is a < b <= c.
    #   If we add x, and a, y are in S.
    #   Case 1: a < y < x. Triple (a, y, x). Conflict if a | (y+x).
    #   Case 2: a < x <= y. Triple (a, x, y). Conflict if a | (x+y).
    #   Case 3: a < x = y. Triple (a, x, x). Conflict if a | 2x.
    #   
    # So M[a][x] should contain:
    # - y such that a < y < x and a | (x+y)
    # - y such that y > x and a | (x+y)
    # - x itself if a | 2x (since a < x is required, and we assume a < x)
    
    M = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    for a in range(1, n_max + 1):
        for x in range(a + 1, n_max + 1):
            mask = 0
            # Check y > x
            # a | (x+y) => y = k*a - x
            # y > x => k*a > 2x => k > 2x/a
            # y <= n_max
            k_min = (2 * x) // a + 1
            for k in range(k_min, (n_max + x) // a + 1):
                y = k * a - x
                if x < y <= n_max:
                    mask |= (1 << y)
            # Check y < x
            # a | (x+y) => y = k*a - x
            # y < x => k*a < 2x
            # y > a (since a < b=y)
            # Also y >= 1
            k_max = (2 * x - 1) // a
            for k in range(1, k_max + 1):
                y = k * a - x
                if a < y < x:
                    mask |= (1 << y)
            # Check y = x
            if (2 * x) % a == 0:
                mask |= (1 << x)
            M[a][x] = mask

    def can_add(x, S_mask):
        """Check if x can be added to set represented by S_mask."""
        if (S_mask >> x) & 1:
            return False
        
        # 1. x is the smallest element (a=x)
        # Check if any (b,c) in conflict[x] are both in S
        for b, c in conflict[x]:
            if ((S_mask >> b) & 1) and ((S_mask >> c) & 1):
                return False
        
        # 2. x is b or c
        # For each a in S with a < x, check if S intersects M[a][x]
        # We iterate through bits of S_mask that are < x
        # Or just iterate a from 1 to x-1 and check if a is in S
        # Since x <= 256, iterating a is fast.
        for a in range(1, x):
            if (S_mask >> a) & 1:
                if S_mask & M[a][x]:
                    return False
        return True

    def greedy_extend(start_mask, num_passes=10):
        """Extend a set by trying to add elements."""
        current = start_mask
        added_any = True
        for _ in range(num_passes):
            if time.time() > deadline:
                break
            added_any = False
            # Try adding elements in random order? Or ascending?
            # Ascending might block large ones? 
            # Let's try random order to avoid bias.
            candidates = list(range(1, n_max + 1))
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    break
                if not (current >> x) & 1:
                    if can_add(x, current):
                        current |= (1 << x)
                        added_any = True
            if not added_any:
                break
        return current

    def local_search_improve(current_mask):
        """Improve the set via swap operations."""
        current = current_mask
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Get list of elements in current
            S_elems = [i for i in range(1, n_max + 1) if (current >> i) & 1]
            # Get list of elements NOT in current
            not_S_elems = [i for i in range(1, n_max + 1) if not (current >> i) & 1]
            
            random.shuffle(S_elems)
            random.shuffle(not_S_elems)
            
            for u in S_elems:
                if time.time() > deadline:
                    return current
                for v in not_S_elems:
                    if time.time() > deadline:
                        return current
                    # Try swap u with v
                    trial_mask = current ^ (1 << u)
                    if can_add(v, trial_mask):
                        current = trial_mask | (1 << v)
                        improved = True
                        break
                if improved:
                    break
        return current

    best_mask = 0
    best_len = 0
    
    # Strategy 1: Upper tails
    for start in range(128, n_max + 1):
        if time.time() > deadline:
            break
        
        start_mask = 0
        for i in range(start, n_max + 1):
            start_mask |= (1 << i)
        
        # Verify base set is valid (should be, but just in case)
        # Actually, upper tails are always valid?
        # If a < b <= c and all >= start.
        # a >= start. b+c >= 2*start.
        # If a > b+c, no conflict.
        # If start is large enough, a > b+c is not guaranteed?
        # e.g. start=100. a=100, b=101, c=102. b+c=203. 100 does not divide 203.
        # a=100, b=100? No, a<b.
        # The condition is a < b.
        # If we take {100, 101, ..., 256}.
        # Min a=100. Max b+c = 256+255=511.
        # 100 divides 200, 300, 400, 500.
        # Pairs (b,c) summing to 200: (101, 99) - 99 not in set.
        # Pairs summing to 300: (101, 199), (102, 198)...
        # (101, 199) are both in set. 100 < 101 <= 199.
        # 100 | 300. Conflict!
        # So upper tails are NOT always valid.
        
        if not can_add(0, 0): pass # dummy
        
        # Check validity of start_mask
        valid = True
        S_elems = [i for i in range(1, n_max + 1) if (start_mask >> i) & 1]
        for i in range(len(S_elems)):
            if not valid: break
            for j in range(i+1, len(S_elems)):
                # This is slow. Let's use can_add logic or just trust the greedy?
                # No, we need a valid start.
                # Let's just run greedy_extend on it, but first ensure it's valid.
                # We can check validity by trying to add each element one by one?
                # Or just use the full check.
                pass
        
        # Faster validity check for a full set:
        def is_valid(mask):
            S_elems = [i for i in range(1, n_max + 1) if (mask >> i) & 1]
            S_set = set(S_elems)
            for a in S_elems:
                for b, c in conflict[a]:
                    if b in S_set and c in S_set:
                        return False
            return True
        
        if not is_valid(start_mask):
            continue
            
        if bin(start_mask).count('1') > best_len:
            best_mask = start_mask
            best_len = bin(start_mask).count('1')
        
        extended = greedy_extend(start_mask, num_passes=5)
        if bin(extended).count('1') > best_len:
            best_mask = extended
            best_len = bin(extended).count('1')
        
        if time.time() < deadline:
            improved = local_search_improve(extended)
            if bin(improved).count('1') > best_len:
                best_mask = improved
                best_len = bin(improved).count('1')

    # Strategy 2: Random starts
    for _ in range(50):
        if time.time() > deadline:
            break
        start_mask = 0
        # Pick random subset of [64, 256]
        candidates = list(range(64, n_max + 1))
        random.shuffle(candidates)
        for i in range(80): # pick 80 random
            x = candidates[i]
            if can_add(x, start_mask):
                start_mask |= (1 << x)
        
        if bin(start_mask).count('1') > best_len:
            best_mask = start_mask
            best_len = bin(start_mask).count('1')
        
        extended = greedy_extend(start_mask, num_passes=3)
        if bin(extended).count('1') > best_len:
            best_mask = extended
            best_len = bin(extended).count('1')
        
        if time.time() < deadline:
            improved = local_search_improve(extended)
            if bin(improved).count('1') > best_len:
                best_mask = improved
                best_len = bin(improved).count('1')

    # Final polish
    if time.time() < deadline:
        final = best_mask
        improved = local_search_improve(final)
        if bin(improved).count('1') > best_len:
            best_mask = improved
            best_len = bin(improved).count('1')

    if best_mask == 0:
        # Fallback
        best_mask = 1 << 256 # Just one element?
        # Better fallback: largest element
        best_mask = 1 << 256
    
    result = [i for i in range(1, n_max + 1) if (best_mask >> i) & 1]
    return result
# EVOLVE-BLOCK-END
