"""Rewritten program for large integer sets with unique triple sums."""
import time
import random
from itertools import combinations_with_replacement

DEADLINE = 80.0  # seconds, safety margin before 90s kill

def _check_addition(S, x, existing_sums):
    """Check if adding x to S keeps all triple sums unique.
    Only need to check triples that include x."""
    S_list = list(S)
    new_sums = set()
    
    # Triple (x, x, x)
    s = 3 * x
    if s in existing_sums or s in new_sums:
        return False
    new_sums.add(s)
    
    # Triples (x, x, y) for y in S
    for y in S_list:
        s = 2 * x + y
        if s in existing_sums or s in new_sums:
            return False
        new_sums.add(s)
    
    # Triples (x, y, y) for y in S
    for y in S_list:
        s = x + 2 * y
        if s in existing_sums or s in new_sums:
            return False
        new_sums.add(s)
    
    # Triples (x, y, z) for y < z in S (y and z from S, both different from x conceptually but we handle all)
    # Actually we need (x, y, z) where y, z in S and y <= z, but we must avoid double-counting
    # The triples involving x are: (x,x,x), (x,x,y), (x,y,y), (x,y,z) where y,z in S
    # For (x,y,z): y and z are from S, y <= z
    for i, y in enumerate(S_list):
        for z in S_list[i:]:
            s = x + y + z
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)
    
    return True

def _build_sums(S):
    """Build the set of all triple sums for set S."""
    S_list = sorted(S)
    sums = set()
    for a, b, c in combinations_with_replacement(S_list, 3):
        sums.add(a + b + c)
    return sums

def _greedy_build(order, max_val=4096, deadline=None):
    """Greedy build using given candidate order."""
    S = []
    existing_sums = set()
    
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        if x > max_val:
            continue
        if x in S:
            continue
        if _check_addition(S, x, existing_sums):
            S.append(x)
            # Update existing_sums with new triples involving x
            S_list = sorted(S)
            for a, b, c in combinations_with_replacement([x] + [y for y in S_list if y != x or S_list.count(y) > S_list.index(y)], 3):
                pass  # This is getting complex, let's just rebuild or add incrementally
            # Simpler: add all new sums
            for a, b, c in combinations_with_replacement(S, 3):
                existing_sums.add(a + b + c)
    
    return S

def _greedy_build_v2(order, max_val=4096, deadline=None):
    """Greedy build with incremental sum tracking."""
    S = []
    existing_sums = set()
    
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        if x > max_val:
            continue
        if x in S:
            continue
        
        # Check if x can be added
        S_list = sorted(S)
        new_sums = set()
        ok = True
        
        # (x,x,x)
        s = 3 * x
        if s in existing_sums:
            ok = False
        else:
            new_sums.add(s)
        
        if ok:
            # (x,x,y) for y in S
            for y in S_list:
                s = 2 * x + y
                if s in existing_sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
        
        if ok:
            # (x,y,y) for y in S
            for y in S_list:
                s = x + 2 * y
                if s in existing_sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
        
        if ok:
            # (x,y,z) for y <= z in S
            n = len(S_list)
            for i in range(n):
                for j in range(i, n):
                    s = x + S_list[i] + S_list[j]
                    if s in existing_sums or s in new_sums:
                        ok = False
                        break
                    new_sums.add(s)
                if not ok:
                    break
        
        if ok:
            S.append(x)
            existing_sums.update(new_sums)
    
    return S

def _geometric_seed(start, ratio, max_val=4096):
    """Generate a geometric sequence."""
    seq = []
    x = start
    while x <= max_val:
        seq.append(x)
        x = int(x * ratio) + 1
    return seq

def _try_backtrack_improve(S, max_val=4096, deadline=None, max_attempts=100):
    """Try to improve S by removing and re-adding elements."""
    S = sorted(S)
    existing_sums = _build_sums(S)
    current_size = len(S)
    
    for _ in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try adding a random number
        x = random.randint(1, max_val)
        if x in S:
            continue
        
        S_list = sorted(S)
        new_sums = set()
        ok = True
        
        s = 3 * x
        if s in existing_sums:
            ok = False
        else:
            new_sums.add(s)
        
        if ok:
            for y in S_list:
                s = 2 * x + y
                if s in existing_sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
        
        if ok:
            for y in S_list:
                s = x + 2 * y
                if s in existing_sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
        
        if ok:
            n = len(S_list)
            for i in range(n):
                for j in range(i, n):
                    s = x + S_list[i] + S_list[j]
                    if s in existing_sums or s in new_sums:
                        ok = False
                        break
                    new_sums.add(s)
                if not ok:
                    break
        
        if ok:
            S.append(x)
            existing_sums.update(new_sums)
            current_size = len(S)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_S = []
    
    # Strategy 1: Greedy with geometric seeds
    for start in [1, 2, 3, 5, 8, 13]:
        for ratio in [1.5, 1.6, 1.7, 1.8, 2.0, 2.2]:
            if time.time() > deadline:
                break
            seed = _geometric_seed(start, ratio)
            # Extend with all numbers for greedy
            order = list(range(1, 4097))
            S = _greedy_build_v2(order, deadline=deadline)
            if len(S) > len(best_S):
                best_S = S[:]
        
        if time.time() > deadline:
            break
    
    # Strategy 2: Random restarts with greedy
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random ordering biased toward spread
        order = list(range(1, 4097))
        random.shuffle(order)
        S = _greedy_build_v2(order, deadline=deadline)
        if len(S) > len(best_S):
            best_S = S[:]
    
    # Strategy 3: Backtrack improvement on best
    if len(best_S) > 0:
        improved = _try_backtrack_improve(best_S, deadline=deadline, max_attempts=500)
        if len(improved) > len(best_S):
            best_S = improved
    
    # Final check: ensure validity
    if best_S:
        S_sorted = sorted(best_S)
        sums = set()
        valid = True
        for a, b, c in combinations_with_replacement(S_sorted, 3):
            s = a + b + c
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            # Fallback to simple greedy
            best_S = _greedy_build_v2(list(range(1, 4097)), deadline=deadline)
    
    return best_S

# EVOLVE-BLOCK-END
