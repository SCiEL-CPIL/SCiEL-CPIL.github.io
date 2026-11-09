import time

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

def _greedy_descending():
    """Construct a valid set using greedy descending order."""
    n = N
    S = []
    for x in range(n, 0, -1):
        valid = True
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.append(x)
    return S

def _is_valid_in_set(x, S):
    """
    Check if x can be added to S.
    S is a set of integers.
    x is an integer not in S.
    Condition: For all b, c in S with x < b <= c, x does not divide b+c.
    Also, since x is the smallest, we only need to check if x divides b+c.
    We do NOT need to check if b divides x+c or c divides x+b, 
    because those conditions were already satisfied when b and c were added 
    (assuming S was valid before adding x).
    Wait, if S is valid, adding x might create new triples where x is NOT the smallest?
    No, x is smaller than all b, c in S?
    If we are doing local search, x might be LARGER than some elements in S?
    In the greedy construction, we add in descending order, so x is always the smallest.
    In local search, we might add a small x to a set of large numbers.
    But what if we remove a large number and add a small number?
    The set S' contains numbers larger than x?
    If we only consider adding x < min(S'), then x is the smallest.
    If x > min(S'), then x is not the smallest.
    Then we must check:
    1. x divides b+c for b,c in S' with x < b <= c.
    2. a divides x+c for a in S' with a < x.
    3. a divides x+b for a in S' with a < x.
    
    So the fast check `O(|S'|+x)` only works if x < min(S').
    In our context, S is built from large numbers. B contains small numbers.
    So x < min(S') is likely true.
    But to be safe, we should use the full check if x is not the smallest.
    However, the greedy set S consists of large numbers.
    The "holes" B are small numbers.
    So x < min(S) is true for all x in B?
    Not necessarily. S might have gaps.
    But S is "dense" at the top.
    Let's assume x < min(S) for the optimization.
    If not, fall back to slow check.
    """
    # Check if x is the smallest in S U {x}
    # If S is empty, valid.
    if not S:
        return True
    
    min_S = min(S)
    if x >= min_S:
        # Slow check
        # Construct sorted list of S U {x}
        L = sorted(list(S) + [x])
        m = len(L)
        for i in range(m):
            a = L[i]
            if a == x:
                continue # We only care about triples involving x?
                # No, we need to check if adding x breaks validity.
                # Since S was valid, only triples containing x can be invalid.
                # So we check triples where x is a, b, or c.
            
            # If x is a:
            if a == x:
                # Check x | b+c for b,c in S with x < b <= c
                for j in range(i + 1, m):
                    b = L[j]
                    for k in range(j, m):
                        c = L[k]
                        if (b + c) % x == 0:
                            return False
            
            # If x is b:
            # a < x = b <= c
            # Check a | x+c
            # If x is c:
            # a < b <= x
            # Check a | b+x
            
        # The slow check above is complex.
        # Let's just use the full _validate on S U {x}
        # It's O(M^3). M~100. 10^6 ops.
        # If this case is rare, it's fine.
        return _validate(S | {x})

    # Fast check: x < min(S)
    # We need to check if x divides b+c for any b,c in S.
    # Since x < b <= c, x is the smallest.
    # Condition: x | b+c.
    
    # Optimization using mods
    # Count mods of elements in S modulo x
    count_mod = [0] * x
    for val in S:
        count_mod[val % x] += 1
    
    # Check conflicts
    # 1. b = c => 2b = kx. Exists b in S such that 2b % x == 0.
    #    This is equivalent to: exists r such that count_mod[r] >= 1 and (2*r) % x == 0.
    for r in range(x):
        if count_mod[r] > 0 and (2 * r) % x == 0:
            return False
        
    # 2. b != c => b+c = kx.
    #    Exists b, c in S such that b % x + c % x == 0 mod x.
    #    Let r = b % x. Then c % x == (-r) % x.
    #    We need count_mod[r] >= 1 and count_mod[(-r)%x] >= 1.
    #    And b != c.
    #    If r == (-r)%x, we need count_mod[r] >= 2.
    #    If r != (-r)%x, we need count_mod[r] >= 1 and count_mod[(-r)%x] >= 1.
    
    for r in range(x):
        r_neg = (-r) % x
        if r == r_neg:
            if count_mod[r] >= 2:
                return False
        else:
            if count_mod[r] >= 1 and count_mod[r_neg] >= 1:
                return False
                
    return True

def construct():
    S = _greedy_descending()
    S_set = set(S)
    
    # Local search
    improved = True
    while improved:
        improved = False
        # Try removing each y in S
        # To be efficient, iterate over a copy of S
        for y in list(S_set):
            S_prime = S_set - {y}
            # Find addable elements from B
            B = set(range(1, N + 1)) - S_prime
            
            # We want to add as many as possible from B
            # Sort B descending? Or just iterate?
            # Adding larger numbers first might be better?
            # Or smaller?
            # Let's try adding all that are valid.
            # But they might conflict with each other.
            # So we add greedily.
            
            # Heuristic: Try to add elements in descending order
            # This mirrors the original greedy.
            addable = []
            for x in sorted(B, reverse=True):
                if _is_valid_in_set(x, S_prime):
                    addable.append(x)
                    S_prime.add(x)
            
            if len(addable) > 1:
                S_set = S_prime
                improved = True
                break # Restart the outer loop
                
    return sorted(S_set)

if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
