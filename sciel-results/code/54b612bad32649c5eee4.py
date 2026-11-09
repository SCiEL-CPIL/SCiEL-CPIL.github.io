import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit out of 90s

    def full_check(S):
        """Check if set S is valid: for all a<b<=c in S, a does not divide b+c."""
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i+1, m):
                b = elems[j]
                for k in range(j, m):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, x):
        """Check if x can be added to S without violating constraints.
        Only checks triples involving x."""
        if x in S:
            return True
        
        # For each pair (u, v) in S with u < v, and for each triple involving x:
        # Case 1: x is the smallest element (x < u < v): check x | (u+v)
        # Case 2: x is the middle element (u < x < v): check u | (x+v)
        # Case 3: x is the largest element (u < v < x): check u | (v+x)
        # Case 4: x = v (u < x, v=x): check u | (x+x) = 2x
        # Case 5: x = u (x < v, but x < v means x is smallest): covered by case 1
        
        S_list = list(S)
        for u in S_list:
            for v in S_list:
                if u >= v:
                    continue
                # Now u < v
                # Triple (x, u, v) with x < u < v: x is smallest
                if x < u:
                    if (u + v) % x == 0:
                        return False
                # Triple (u, x, v) with u < x < v: x is middle
                elif u < x < v:
                    if (x + v) % u == 0:
                        return False
                # Triple (u, v, x) with u < v < x: x is largest
                elif v < x:
                    if (v + x) % u == 0:
                        return False
                # Triple (u, x, x) with u < x: b=c=x
                # This requires x to be in the set twice, but since it's a set,
                # we only have one x. However b<=c allows b=c.
                # If x is in S, then (u, x, x) is a valid triple to check.
                # But we're checking if ADDING x creates violations.
                # If x is added, then (u, x, x) exists for all u < x in S.
                elif x < u:
                    # Already handled above
                    pass
                elif u < x:
                    # Check (u, x, x): u < x, b=x, c=x
                    if (x + x) % u == 0:
                        return False
        return True

    # Base solution: upper tail [171, 256]
    # Valid because for a<b<=c in [171,256]: b+c >= 172+172=344 > 2*171=342
    # and 3*a >= 513 > 512 = max(b+c), so only k=2 possible, but b+c > 2a
    best_set = set(range(171, 257))
    best_size = len(best_set)
    
    # Greedy addition: try adding elements from 170 down to 1
    S = set(range(171, 257))
    for x in range(170, 0, -1):
        if time.time() > deadline - 30:
            break
        if can_add(S, x):
            S.add(x)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    # Multiple greedy passes with different orderings
    for attempt in range(5):
        if time.time() > deadline - 25:
            break
        
        if attempt == 0:
            order = list(range(170, 0, -1))
        elif attempt == 1:
            order = list(range(1, 171))
        elif attempt == 2:
            order = list(range(170, 0, -1))
            random.shuffle(order)
        elif attempt == 3:
            order = list(range(1, 171))
            random.shuffle(order)
        else:
            order = list(range(1, 171))
            random.shuffle(order)
        
        S = set(range(171, 257))
        for x in order:
            if time.time() > deadline - 25:
                break
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search: try swapping elements
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        S_trial = set(best_set)
        removed = []
        
        # Remove 1-2 random elements
        num_remove = random.randint(1, 2)
        S_list = list(S_trial)
        random.shuffle(S_list)
        for i in range(min(num_remove, len(S_list))):
            S_trial.remove(S_list[i])
            removed.append(S_list[i])
        
        # Try to add elements back (including removed ones and new ones)
        candidates = list(set(range(1, 257)) - S_trial)
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if can_add(S_trial, x):
                S_trial.add(x)
        
        if len(S_trial) > best_size:
            best_size = len(S_trial)
            best_set = set(S_trial)
    
    # Final validation
    if full_check(best_set):
        return list(best_set)
    
    # Fallback to known valid solution
    return list(range(171, 257))
