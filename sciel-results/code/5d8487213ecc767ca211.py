import time
from math import gcd

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0:
        return False
    i = 3
    while i * i <= a:
        if a % i == 0:
            return False
        i += 2
    return True

def _valid_set(s):
    L = list(s)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(n):
            if j == i:
                continue
            b = L[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = L[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    t0 = time.time()
    n = 256
    primes = [a for a in range(2, n + 1) if _is_prime(a)]
    
    # Verify prime set is valid
    if not _valid_set(primes):
        # Fallback greedy
        result = []
        for x in range(2, n + 1):
            if time.time() - t0 > 55:
                break
            if all((b * c) % a != 0 for i, a in enumerate(result) for j, b in enumerate(result) if j != i for k, c in enumerate(result) if k != i and k != j):
                result.append(x)
        return result
    
    # Try local search: swap one element for another to see if we can grow
    best = primes[:]
    best_size = len(best)
    best_set = set(best)
    
    # Try adding each composite by removing conflicting primes
    for c in range(2, n + 1):
        if time.time() - t0 > 50:
            break
        if c in best_set:
            continue
        if _is_prime(c):
            continue
        
        # Find which elements in best conflict with adding c
        # c in set means: for all distinct a,b in current set (plus c), a ∤ b*c and a ∤ c*b (same) and c ∤ a*b
        # Actually need to check: with c added, for all distinct x,y,z in new set, x ∤ y*z
        # This is expensive; let's just check if c can be added by removing minimal conflicts
        
        # Check: for c to be in set, for all distinct a,b in set: c ∤ a*b
        # And for all distinct a,b in set: a ∤ b*c (i.e., a ∤ b*c)
        # And for all distinct a,b in set: a ∤ c*b (same as above)
        
        conflicts = set()
        for i, a in enumerate(best):
            for j, b in enumerate(best):
                if i == j:
                    continue
                # Check if a | b*c
                if (b * c) % a == 0:
                    conflicts.add(a)
                    conflicts.add(b)
                # Check if c | a*b
                if (a * b) % c == 0:
                    conflicts.add(a)
                    conflicts.add(b)
        
        if len(conflicts) < 1:  # Need to remove fewer than we add
            new_set = [x for x in best if x not in conflicts] + [c]
            if len(new_set) > best_size and _valid_set(new_set):
                best = new_set
                best_set = set(best)
                best_size = len(best)
    
    return [int(x) for x in best]
