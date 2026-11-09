"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _get_divisors(n):
    """Return sorted list of divisors of n."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort()
    return divs

def _lcm(a, b):
    """Compute lcm of a and b."""
    return a * b // math.gcd(a, b)

def _can_add(x, S, limit=512):
    """Check if x can be added to set S without violating lcm constraint."""
    for a in S:
        if _lcm(x, a) > limit:
            return False
    return True

def _greedy_expand(S, limit=512, deadline=None):
    """
    Greedily add numbers to S that are compatible with all elements.
    Process candidates in order of fewest conflicts first (more compatible first).
    """
    S = set(S)
    candidates = [x for x in range(1, limit + 1) if x not in S]
    
    # Score each candidate by how many elements in S it conflicts with
    # Fewer conflicts = better to add first
    def conflict_count(x):
        count = 0
        for a in S:
            if _lcm(x, a) > limit:
                count += 1
        return count
    
    # Sort candidates by conflict count (ascending), then by value
    candidates.sort(key=lambda x: (conflict_count(x), x))
    
    for x in candidates:
        if deadline is not None and time.time() > deadline:
            break
        if _can_add(x, S, limit):
            S.add(x)
    
    return S

def construct():
    """
    Returns a subset of [1, 512] with lcm(a, b) <= 512 for all pairs,
    maximizing cardinality.
    
    Strategy:
    1. Start with divisors of highly composite numbers <= 512 (360, 420, 480, 504 each have 24 divisors).
    2. Greedily expand by adding compatible numbers, prioritizing those with fewest conflicts.
    3. Try multiple base sets and return the best.
    4. Use a deadline guard to stay within time limits.
    """
    deadline = time.time() + 50  # 50 second budget, leaving 10s safety margin
    
    # Base sets: divisors of highly composite numbers <= 512
    # Numbers with many divisors: 360(24), 420(24), 480(24), 504(24), 240(20), 336(24), 420(24)
    base_numbers = [360, 420, 480, 504, 240, 336, 288, 384, 432, 448, 462, 468, 480, 504]
    base_numbers = list(set(base_numbers))
    
    best_set = None
    best_size = 0
    
    for L in base_numbers:
        if time.time() > deadline:
            break
        divs = _get_divisors(L)
        S = set(divs)
        
        # Greedy expansion
        S_expanded = _greedy_expand(S, limit=512, deadline=deadline)
        
        if len(S_expanded) > best_size:
            best_size = len(S_expanded)
            best_set = S_expanded
    
    # Also try starting from {1} and building up
    if time.time() < deadline:
        S = {1}
        S_expanded = _greedy_expand(S, limit=512, deadline=deadline)
        if len(S_expanded) > best_size:
            best_size = len(S_expanded)
            best_set = S_expanded
    
    # Fallback: if no set was found, return divisors of 360
    if best_set is None:
        best_set = set(_get_divisors(360))
    
    return list(best_set)
# EVOLVE-BLOCK-END
