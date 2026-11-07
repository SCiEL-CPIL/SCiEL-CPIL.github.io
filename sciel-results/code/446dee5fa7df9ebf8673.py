"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def _is_valid(S):
    """Check if set S has no three distinct members a,b,c with all pair sums in S."""
    s = set(S)
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j+1, n):
                c = lst[k]
                if (a+b) in s and (a+c) in s and (b+c) in s:
                    return False
    return True

def _greedy_with_order(order, n):
    """Build a set by trying elements in given order, skipping if it creates a violation."""
    S = set()
    for x in order:
        if x in S:
            continue
        S.add(x)
        # Check if adding x creates a violation with any pair already in S
        valid = True
        lst = list(S - {x})
        for i in range(len(lst)):
            a = lst[i]
            for j in range(i+1, len(lst)):
                b = lst[j]
                if (a+x) in S and (b+x) in S and (a+b) in S:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            S.remove(x)
    return S

def construct():
    n = P['n']
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    # Baseline: {63, ..., 128} has size 66
    best_set = set(range(63, n + 1))
    best_size = len(best_set)
    
    # Try randomized greedy approaches
    rng = random.Random(42)
    
    # Strategy 1: Try to extend the upper half by including some lower elements
    # Start from upper half and try to add elements from below
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random ordering biased toward upper half
        upper = list(range(63, n + 1))
        lower = list(range(1, 63))
        rng.shuffle(upper)
        rng.shuffle(lower)
        
        # Mix: put upper half first (to maintain base), then try some lower
        mix_ratio = rng.random()
        num_lower = int(len(lower) * mix_ratio)
        order = upper + lower[:num_lower]
        
        S = _greedy_with_order(order, n)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 2: Pure random orders
    for trial in range(300):
        if time.time() > deadline:
            break
        
        order = list(range(1, n + 1))
        rng.shuffle(order)
        S = _greedy_with_order(order, n)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 3: Start from a random subset of the upper half, then greedily add
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Randomly pick a subset of upper half to start
        upper = list(range(63, n + 1))
        rng.shuffle(upper)
        start_size = rng.randint(30, 60)
        start_set = set(upper[:start_size])
        
        # Greedily add remaining elements in random order
        remaining = [x for x in range(1, n + 1) if x not in start_set]
        rng.shuffle(remaining)
        
        S = set(start_set)
        for x in remaining:
            if x in S:
                continue
            S.add(x)
            valid = True
            lst = list(S - {x})
            for i in range(len(lst)):
                a = lst[i]
                for j in range(i+1, len(lst)):
                    b = lst[j]
                    if (a+x) in S and (b+x) in S and (a+b) in S:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                S.remove(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    if time.time() > deadline - 2.0:
        # Return best found
        return [int(x) for x in sorted(best_set)]
    
    # Final verification
    if _is_valid(best_set):
        return [int(x) for x in sorted(best_set)]
    else:
        # Fallback to known valid set
        return [int(x) for x in range(63, n + 1)]
# EVOLVE-BLOCK-END
