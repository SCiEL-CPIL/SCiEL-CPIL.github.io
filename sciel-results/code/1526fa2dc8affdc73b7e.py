import time
import random

N = 256

def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True


def _can_add(x, S):
    """
    Check if x can be added to S while maintaining validity.
    S is a set of integers. x is an integer not in S.
    """
    S_list = list(S)
    
    # Case 1: x is 'a' (smallest in triple)
    # a=x, b in S, c in S, x < b <= c
    # Check if x | (b+c)
    for b in S_list:
        if b <= x:
            continue
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
                
    # Case 2: x is 'b' (middle in triple)
    # a in S, b=x, c in S, a < x <= c
    # Check if a | (x+c)
    for a in S_list:
        if a >= x:
            continue
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
                
    # Case 3: x is 'c' (largest in triple)
    # a in S, b in S, c=x, a < b <= x
    # Check if a | (b+x)
    for b in S_list:
        if b > x:
            continue
        for a in S_list:
            if a >= b:
                continue
            if (b + x) % a == 0:
                return False
                
    return True


def _greedy_descending():
    n = N
    S = set()
    for x in range(n, 0, -1):
        if _can_add(x, S):
            S.add(x)
    return S


def _greedy_ascending():
    n = N
    S = set()
    for x in range(1, n + 1):
        if _can_add(x, S):
            S.add(x)
    return S


def _greedy_random(seed):
    random.seed(seed)
    n = N
    S = set()
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(x, S):
            S.add(x)
    return S


def _interval_seed():
    """
    Construct a valid set using the interval [floor(2N/3)+1, N].
    For a >= 171, b+c <= 512 < 3*171 = 513, so a cannot divide b+c.
    """
    n = N
    start = (2 * n) // 3 + 1  # 171 for N=256
    S = set(range(start, n + 1))
    return S


def _local_search(S, deadline):
    """
    Local search: try to add elements, then try swaps.
    Returns improved set.
    """
    S = set(S)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add elements
        for x in range(1, N + 1):
            if x in S:
                continue
            if _can_add(x, S):
                S.add(x)
                improved = True
        
        if time.time() >= deadline:
            break
        
        # Phase 2: Try swaps - remove one element, see if we can add more
        best_swap = None
        best_swap_size = len(S)
        
        for y in list(S):
            if time.time() >= deadline:
                break
            S_prime = S - {y}
            added = 0
            for x in range(1, N + 1):
                if x in S_prime:
                    continue
                if _can_add(x, S_prime):
                    S_prime.add(x)
                    added += 1
            
            if len(S_prime) > best_swap_size:
                best_swap_size = len(S_prime)
                best_swap = S_prime.copy()
        
        if best_swap is not None and len(best_swap) > len(S):
            S = best_swap
            improved = True
    
    return S


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Generate initial candidates
    candidates = []
    
    # Interval seed (known valid, size 86)
    candidates.append(_interval_seed())
    
    # Greedy methods
    candidates.append(_greedy_descending())
    candidates.append(_greedy_ascending())
    
    # Random restarts
    for i in range(10):
        if time.time() >= deadline:
            break
        candidates.append(_greedy_random(i))
    
    # Pick the best valid candidate
    best_set = set()
    best_size = 0
    for S in candidates:
        if _validate(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
        else:
            # Try to repair by removing conflicting elements
            # Simple repair: keep only elements that can be added greedily
            repaired = set()
            for x in sorted(S, reverse=True):
                if _can_add(x, repaired):
                    repaired.add(x)
            if _validate(repaired) and len(repaired) > best_size:
                best_size = len(repaired)
                best_set = repaired
    
    if not best_set:
        # Fallback
        best_set = {N}
        best_size = 1
    
    # Local search to improve
    if time.time() < deadline:
        best_set = _local_search(best_set, deadline)
    
    return sorted(best_set)


if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
