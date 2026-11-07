"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 128
DEADLINE_FRACTION = 0.85  # safety margin

def _is_valid_triple(a, b, c, selected):
    """Check if adding a,b,c to selected would create a forbidden configuration.
    Actually we need: for any three distinct members x,y,z in the set,
    it is NOT the case that x+y, x+z, y+z are all in the set.
    
    The constraint is: there do NOT exist three distinct a,b,c in S such that
    a+b, a+c, b+c are ALL in S.
    
    So when building S, we must ensure no such triple exists.
    """
    pass

def _check_validity(S):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    # For each triple of distinct elements a,b,c in S, check if a+b, a+c, b+c are all in S
    # This is O(n^3 * 1) which for n=64 is ~41664 checks, fine.
    elems = list(S)
    n = len(elems)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = elems[i], elems[j], elems[k]
                if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
                    return False
    return True

def _greedy_build(deadline):
    """Greedy approach: start with empty set, add elements one by one if valid."""
    S = set()
    # Try adding elements in a smart order
    # Heuristic: start with large elements (sums are more likely to exceed 128)
    candidates = list(range(N, 0, -1))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        S.add(x)
        if not _check_validity(S):
            S.discard(x)
    
    return S

def _local_search(S, deadline, max_iters=5000):
    """Local search: try to add/remove elements to improve set size."""
    S = set(S)
    current_size = len(S)
    best_S = set(S)
    best_size = current_size
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try adding a random element not in S
        candidates_to_add = [x for x in range(1, N+1) if x not in S]
        if candidates_to_add:
            x = random.choice(candidates_to_add)
            S.add(x)
            if _check_validity(S):
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = set(S)
            else:
                S.discard(x)
        
        # Occasionally try removing an element to allow more additions
        if random.random() < 0.3 and len(S) > 0:
            x = random.choice(list(S))
            S.discard(x)
            current_size -= 1
    
    return best_S

def _random_restart(deadline):
    """Single random restart with greedy + local search."""
    # Random initial set
    S = set()
    # Start with a random subset
    for x in range(1, N+1):
        if random.random() < 0.5:
            S.add(x)
    # Remove invalid triples by removing elements
    # Simple approach: iteratively remove elements that participate in bad triples
    changed = True
    while changed and time.time() < deadline:
        changed = False
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if (a+b) in S and (a+c) in S and (b+c) in S:
                        # Remove one of them
                        victim = random.choice([a, b, c])
                        S.discard(victim)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    
    # Now S is valid, try to grow it
    S = _local_search(S, deadline, max_iters=2000)
    return S

def _structured_search(deadline):
    """
    Structural approach: exploit the fact that if S is contained in [65, 128],
    then all pair sums are > 128, so no pair sum is in S. Thus any subset of [65,128] is valid.
    Size = 64.
    
    But we can do better by mixing small and large numbers.
    Key insight: if a, b in S and a+b <= 128, then a+b must NOT be in S (unless we break the triple condition).
    Actually the condition is about TRIPLES: a,b,c distinct, and a+b, a+c, b+c all in S.
    
    So if we pick S such that for any three distinct a,b,c in S, at least one of a+b, a+c, b+c is not in S.
    
    A sufficient condition: S is sum-free (no a+b in S for a,b in S). Then no pair sum is in S, so condition holds.
    Max sum-free subset of [1,128] has size 64 (the upper half).
    
    But we can have non-sum-free sets that still satisfy the triple condition.
    E.g., if a+b is in S, we just need to ensure that for any c distinct from a,b, either a+c or b+c is not in S.
    """
    best = set(range(65, N+1))  # size 64
    best_size = len(best)
    
    # Try to improve by adding some small numbers and removing conflicting large ones
    # Strategy: try adding a small number x, see which large numbers must be removed
    for x in range(1, 65):
        if time.time() > deadline:
            break
        new_S = set(best)
        new_S.add(x)
        # Check validity
        if _check_validity(new_S):
            if len(new_S) > best_size:
                best_size = len(new_S)
                best = new_S
        else:
            # Try to fix by removing some elements
            # Find which elements cause the conflict with x
            S_set = set(best)
            # For x to be valid, we need: for any a,b in best (a != b, a != x, b != x),
            # it's not the case that x+a, x+b, a+b are all in new_S
            # Since a+b might be in best, we need to ensure x+a or x+b is not in new_S
            # x+a is in new_S only if x+a is in best (since x+a > x >= 1, and best is [65,128])
            # x+a is in [65,128] iff a >= 65-x
            # Similarly x+b in [65,128] iff b >= 65-x
            # So conflicts arise when a,b >= 65-x and a+b in best
            # This is complex; let's just try local search
            new_S = _local_search(new_S, deadline, max_iters=500)
            if len(new_S) > best_size:
                best_size = len(new_S)
                best = new_S
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 * DEADLINE_FRACTION
    
    random.seed(42)
    
    # Start with the known good solution: [65, 128]
    best = set(range(65, N+1))
    best_size = len(best)
    
    # Try structured search first
    if time.time() < deadline:
        structured = _structured_search(deadline)
        if len(structured) > best_size:
            best = structured
            best_size = len(structured)
    
    # Try random restarts with local search
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        result = _random_restart(deadline)
        if len(result) > best_size:
            best_size = len(result)
            best = result
    
    # Final greedy attempt
    if time.time() < deadline:
        greedy = _greedy_build(deadline)
        if len(greedy) > best_size:
            best = greedy
    
    # Return as list of Python ints
    return list(best)
# EVOLVE-BLOCK-END
