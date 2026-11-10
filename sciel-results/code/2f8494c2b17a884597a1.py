"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 128}

def _is_valid_set(S):
    """Check if set S has no three distinct members a,b,c where a+b, a+c, b+c are all in S."""
    S_set = set(S)
    for a, b, c in combinations(S, 3):
        if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
            return False
    return True

def _greedy_construct(n, order, deadline):
    """Greedy: add elements in given order if no forbidden triple is created."""
    S = []
    S_set = set()
    # For each pair (a,b) in S, track the "dangerous" value d = a+b
    # Adding c creates a forbidden triple if there exist a,b in S such that
    # a+b, a+c, b+c are all in S_set ∪ {c}
    # More precisely: adding c is bad if there exist distinct a,b in S such that
    # (a+b) in S_set AND (a+c) in S_set AND (b+c) in S_set
    # OR (a+b) in S_set AND (a+c) == c (impossible for positive) 
    # Actually: after adding c, the triple {a,b,c} is forbidden if a+b, a+c, b+c all in S∪{c}
    # Since c is new, a+c and b+c are new values. So we need a+b in S_set AND a+c in S_set AND b+c in S_set
    # But a+c and b+c are NOT in S_set yet (they'd be > max(S) typically, but could be in S if S has large values)
    # Wait, a+c could be in S if c is small and a is large. Let me think more carefully.
    # After adding c to S, for triple {a,b,c}: need a+b ∈ S∪{c}, a+c ∈ S∪{c}, b+c ∈ S∪{c}
    # Since c is just added: a+b could be in S or equal to c (if a+b=c, but a,b < c in our order... not necessarily)
    # Let's just check directly for small n=128.
    
    for x in order:
        if time.time() > deadline:
            break
        # Check: does adding x create a forbidden triple?
        # A forbidden triple {a,b,c} with c=x: need a+b, a+x, b+x all in S∪{x}
        # Since x is new, a+x and b+x are in S∪{x} only if they equal x (impossible for positive a,b) or are in S
        # a+b is in S∪{x} if a+b is in S or a+b == x
        bad = False
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a = S[i]
                b = S[j]
                # Check if {a,b,x} would be forbidden
                # Need: a+b in S∪{x}, a+x in S∪{x}, b+x in S∪{x}
                ab = a + b
                ax = a + x
                bx = b + x
                # a+b in S∪{x}
                if ab not in S_set and ab != x:
                    continue
                # a+x in S∪{x}: a+x == x means a==0 (impossible), so need a+x in S_set
                if ax not in S_set:
                    continue
                # b+x in S∪{x}: similarly need b+x in S_set
                if bx not in S_set:
                    continue
                bad = True
                break
            if bad:
                break
        if not bad:
            S.append(x)
            S_set.add(x)
    return S

def _local_improve(S, deadline):
    """Try to add more elements or swap to improve size."""
    S_list = list(S)
    S_set = set(S_list)
    n = P['n']
    
    # Try adding each missing element
    for x in range(1, n+1):
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        bad = False
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                ab = a + b
                ax = a + x
                bx = b + x
                if ab not in S_set and ab != x:
                    continue
                if ax not in S_set:
                    continue
                if bx not in S_set:
                    continue
                bad = True
                break
            if bad:
                break
        if not bad:
            S_list.append(x)
            S_set.add(x)
    
    return S_list

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = P['n']
    
    best = list(range(n//2+1, n+1))  # seed: 64 elements
    best_size = len(best)
    
    # Try multiple random orderings
    random.seed(42)
    
    # Strategy 1: Greedy from high to low
    order = list(range(n, 0, -1))
    S = _greedy_construct(n, order, deadline)
    if len(S) > best_size:
        best = S
        best_size = len(S)
    
    # Strategy 2: Greedy from low to high
    if time.time() < deadline:
        order = list(range(1, n+1))
        S = _greedy_construct(n, order, deadline)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Strategy 3: Random orderings
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        S = _greedy_construct(n, order, deadline)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Local improvement on best
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Verify final result
    if not _is_valid_set(best):
        # Fallback to seed
        return list(range(n//2+1, n+1))
    
    # Ensure types are correct
    result = [int(x) for x in best]
    return result
# EVOLVE-BLOCK-END
