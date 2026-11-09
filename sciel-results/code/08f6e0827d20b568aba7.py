"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def _find_violation(S_set, S_list):
    """Find a violating triple (a,b,c) where a,b,c distinct and a+b,a+c,b+c in S."""
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            for k in range(j+1, n):
                c = S_list[k]
                if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                    return (a, b, c)
    return None

def _count_violations_for_element(x, S_set, S_list):
    """Count how many violating triples include element x."""
    count = 0
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            # Check if x is one of a,b,c or one of the sums
            # We need triples where x participates
            # This is expensive; instead we'll just do a simpler heuristic
    return count

def _is_valid(S_set):
    """Check if S has no violating triple."""
    S_list = list(S_set)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            for k in range(j+1, n):
                c = S_list[k]
                if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                    return False
    return True

def _local_search(start_set, deadline):
    """Iteratively remove elements to eliminate violations."""
    S_set = set(start_set)
    S_list = list(S_set)
    
    while time.time() < deadline:
        viol = _find_violation(S_set, S_list)
        if viol is None:
            break
        a, b, c = viol
        # The sums that are in S: a+b, a+c, b+c
        # One of {a,b,c,a+b,a+c,b+c} must be removed
        # Remove the one whose removal breaks the most potential violations
        # Simple heuristic: try removing each of the 6 and pick best
        candidates = [a, b, c, a+b, a+c, b+c]
        candidates = [x for x in candidates if x in S_set]
        
        best_remove = None
        best_score = -1
        for x in candidates:
            # Estimate: remove x, count remaining violations (approximate)
            S_test = S_set - {x}
            # Quick check: how many violations still exist?
            # Too expensive to fully check; use a proxy
            # Just pick the element that appears in most sum relationships
            score = 0
            # Count how many other elements y in S_set have y+x in S_set or y-x in S_set
            for y in S_set:
                if y != x:
                    if (y + x) in S_set:
                        score += 1
                    if (y - x) in S_set and y > x:
                        score += 1
            if score > best_score:
                best_score = score
                best_remove = x
        
        if best_remove is not None:
            S_set.remove(best_remove)
            S_list = list(S_set)
        else:
            break
    
    return S_set

def _greedy_construct(deadline):
    """Build a set greedily, trying to include as many elements as possible."""
    # Start with all elements, remove violators
    all_set = set(range(1, 129))
    result = _local_search(all_set, deadline)
    return result

def _random_greedy(deadline, seed=42):
    """Try random orderings of addition."""
    rng = random.Random(seed)
    # Try multiple random starting points
    best_size = 0
    best_set = None
    
    # Strategy: start with a large subset and prune
    for attempt in range(20):
        if time.time() > deadline:
            break
        # Random subset of size ~90
        candidates = list(range(1, 129))
        rng.shuffle(candidates)
        start_size = 85 + rng.randint(0, 15)
        start_set = set(candidates[:start_size])
        
        result = _local_search(start_set, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    if best_set is None:
        best_set = set(range(65, 129))
    
    return best_set

def construct():
    deadline = time.time() + 50  # Safety margin
    
    # Try multiple strategies
    best_set = None
    best_size = 0
    
    # Strategy 1: Full set + local search
    s1 = _local_search(set(range(1, 129)), deadline)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1
    
    # Strategy 2: Random starts
    if time.time() < deadline:
        s2 = _random_greedy(deadline, seed=1)
        if len(s2) > best_size:
            best_size = len(s2)
            best_set = s2
    
    if time.time() < deadline:
        s3 = _random_greedy(deadline, seed=2)
        if len(s3) > best_size:
            best_size = len(s3)
            best_set = s3
    
    if time.time() < deadline:
        s4 = _random_greedy(deadline, seed=3)
        if len(s4) > best_size:
            best_size = len(s4)
            best_set = s4
    
    # Fallback
    if best_set is None:
        best_set = set(range(65, 129))
    
    # Verify validity
    if not _is_valid(best_set):
        # Emergency: use known valid set
        best_set = set(range(65, 129))
    
    return [int(x) for x in sorted(best_set)]
# EVOLVE-BLOCK-END
