import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s/90s limit
    n = 256
    
    def is_valid(S):
        """Check if S satisfies the property: for all a<b<=c in S, a does not divide b+c"""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(x, S):
        """Check if adding x to S preserves validity"""
        if x in S:
            return True
        S_list = sorted(S)
        # x must not violate any triple
        for i, a in enumerate(S_list):
            if a >= x:
                # a > x, so x would be before a. Need to check triples where x is the smaller
                # Actually we need to check all triples involving x
                break
        # More careful: check all triples where x participates
        # Case 1: x is the smallest: x < b <= c, check x does not divide b+c for b,c in S with b > x
        # Case 2: x is the middle: a < x <= c, check a does not divide x+c for a in S with a < x, c in S with c >= x
        # Case 3: x is the largest: a < b <= x, check a does not divide b+x for a,b in S with a < b <= x
        
        # Case 1: x < b <= c
        for j, b in enumerate(S_list):
            if b <= x:
                continue
            for k in range(j, len(S_list)):
                c = S_list[k]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: a < x <= c
        for i, a in enumerate(S_list):
            if a >= x:
                break
            for k, c in enumerate(S_list):
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: a < b <= x
        for i, a in enumerate(S_list):
            if a >= x:
                break
            for j, b in enumerate(S_list):
                if b > x:
                    break
                if a < b:
                    if (b + x) % a == 0:
                        return False
        
        return True
    
    # Strategy 1: Greedy descending - larger numbers are safer
    S1 = set()
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S1):
            S1.add(x)
    
    # Strategy 2: Greedy with different orderings
    S2 = set()
    # Try starting from high end but skip some to allow more
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S2):
            S2.add(x)
    
    # Strategy 3: Local search / hill climbing
    # Start from best greedy, try to add elements
    best_S = set(max([S1, S2], key=len))
    
    # Try to add elements that were rejected
    if time.time() < deadline - 5:
        current = set(best_S)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in range(1, 257):
                if x in current:
                    continue
                if can_add(x, current):
                    current.add(x)
                    improved = True
            if len(current) > len(best_S):
                best_S = set(current)
    
    # Try local search: swap elements
    if time.time() < deadline - 5:
        current = set(best_S)
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Try removing a small element and adding a larger one
            smalls = sorted([x for x in current if x < 128])
            if not smalls:
                break
            removed = random.choice(smalls)
            new_current = set(current)
            new_current.discard(removed)
            added = False
            for x in range(256, removed, -1):
                if x not in new_current and can_add(x, new_current):
                    new_current.add(x)
                    added = True
                    break
            if added and len(new_current) > len(current):
                current = new_current
                if len(current) > len(best_S):
                    best_S = set(current)
    
    if not best_S:
        return [256]
    
    return list(best_S)
