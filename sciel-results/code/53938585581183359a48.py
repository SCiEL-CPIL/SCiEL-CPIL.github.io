import time
import random
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Precompute: for each a, which (b,c) pairs with a<b<=c have a|(b+c)
    # This is expensive, so we'll compute on the fly but cache results

    def is_valid_set(s):
        """Check if a set satisfies the property."""
        elems = sorted(s)
        for i, a in enumerate(elems):
            for j in range(i+1, len(elems)):
                b = elems[j]
                for k in range(j, len(elems)):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(x, s):
        """Check if x can be added to set s without violating the property."""
        if x in s:
            return True
        elems = sorted(s)
        
        # Case 1: x is the smallest element (a=x)
        # Check: for all b,c in s with x<b<=c, x does not divide b+c
        for i, b in enumerate(elems):
            if b <= x:
                continue
            for j in range(i, len(elems)):
                c = elems[j]
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle element (b=x)
        # Check: for all a,c in s with a<x<=c, a does not divide x+c
        for a in elems:
            if a >= x:
                continue
            for c in elems:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest element (c=x)
        # Check: for all a,b in s with a<b<=x, a does not divide b+x
        for i, a in enumerate(elems):
            for j in range(i+1, len(elems)):
                b = elems[j]
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    # Base: upper tail [171, 256] is valid
    base_set = set(range(171, 257))
    best_set = set(base_set)
    best_size = len(best_set)
    
    # Phase 1: Greedy addition - try adding elements from high to low
    for x in range(170, 0, -1):
        if time.time() > deadline - 20:
            break
        if x not in best_set and can_add(x, best_set):
            best_set.add(x)
    
    # Phase 2: Try multiple greedy orderings
    rng = random.Random(42)
    orderings = [
        list(range(170, 0, -1)),  # high to low
        list(range(1, 171)),      # low to high
    ]
    # Add random orderings
    for _ in range(10):
        orderings.append(rng.sample(range(1, 171), 170))
    # Add some structured orderings
    orderings.append(list(range(170, 0, -2)))  # even high to low
    orderings.append(list(range(169, 0, -2)))  # odd high to low
    orderings.append(list(range(1, 171, 2)))   # odd low to high
    orderings.append(list(range(2, 171, 2)))   # even low to high
    
    for ordering in orderings:
        if time.time() > deadline - 15:
            break
        s_trial = set(base_set)
        for x in ordering:
            if x not in s_trial and can_add(x, s_trial):
                s_trial.add(x)
        if len(s_trial) > best_size:
            best_size = len(s_trial)
            best_set = set(s_trial)
    
    # Phase 3: Local search - try to improve by swaps
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        s_trial = set(best_set)
        
        # Try removing one element and adding another
        for rem in sorted(s_trial):
            if time.time() > deadline - 10:
                break
            if rem in s_trial:
                s_trial.discard(rem)
                # Try to add elements not in s_trial
                for add in range(1, 171):
                    if add in s_trial:
                        continue
                    if can_add(add, s_trial):
                        s_trial.add(add)
                        if len(s_trial) > best_size:
                            best_size = len(s_trial)
                            best_set = set(s_trial)
                            improved = True
                            break
                        s_trial.discard(add)
                # Restore
                s_trial.add(rem)
            if improved:
                break
        
        # Try removing two elements and adding up to three
        if not improved and time.time() < deadline - 10:
            small_elems = [x for x in s_trial if x < 171]
            if len(small_elems) >= 2:
                for rem1, rem2 in combinations(small_elems, 2):
                    if time.time() > deadline - 10:
                        break
                    s_trial = set(best_set)
                    s_trial.discard(rem1)
                    s_trial.discard(rem2)
                    added = 0
                    for add in range(1, 171):
                        if add in s_trial or add == rem1 or add == rem2:
                            continue
                        if can_add(add, s_trial):
                            s_trial.add(add)
                            added += 1
                            if added >= 3:
                                break
                    if len(s_trial) > best_size:
                        best_size = len(s_trial)
                        best_set = set(s_trial)
                        improved = True
                        break
    
    # Phase 4: Final polish - try to add any remaining elements
    s_final = set(best_set)
    for x in range(1, 171):
        if time.time() > deadline - 2:
            break
        if x not in s_final and can_add(x, s_final):
            s_final.add(x)
    
    if len(s_final) > best_size:
        best_set = s_final
    
    return sorted(best_set)
