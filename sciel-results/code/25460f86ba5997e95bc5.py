import time

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    N = 256
    # Start with all primes in [2, 256]
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0:
                return False
            i += 6
        return True
    
    primes = [a for a in range(2, N+1) if is_prime(a)]
    S = set(primes)
    
    # Check if a set is valid: for all distinct a,b,c in S, a does not divide b*c
    def is_valid(S):
        lst = sorted(S)
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(n):
                if j == i:
                    continue
                b = lst[j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = lst[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    # Try to add candidates in order (smaller first might be safer? or larger?)
    # Actually, let's try adding candidates that are "harder" to conflict
    # Try all candidates not in S
    candidates = [a for a in range(2, N+1) if a not in S]
    
    # Sort candidates: try ones with fewer conflicts first
    # A candidate x conflicts with current S if there exist distinct b,c in S such that x | b*c
    # We want to add x if for all distinct b,c in S, x does NOT divide b*c
    # AND after adding, the new set is still valid (other elements don't divide products involving x)
    
    def can_add(x, S):
        lst = sorted(S)
        n = len(lst)
        # Check: for all distinct b,c in S, x does not divide b*c
        for i in range(n):
            b = lst[i]
            for j in range(n):
                if j == i:
                    continue
                c = lst[j]
                if (b * c) % x == 0:
                    return False
        # Check: for all distinct a, c in S, a does not divide x*c
        for i in range(n):
            a = lst[i]
            for j in range(n):
                if j == i:
                    continue
                c = lst[j]
                if (x * c) % a == 0:
                    return False
        # Check: for all distinct a, b in S, a does not divide b*x (same as above by symmetry)
        # Already covered above
        return True
    
    # Greedily add candidates
    added = True
    while added and time.time() < deadline:
        added = False
        for x in candidates:
            if x in S:
                continue
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
                added = True
    
    result = sorted(S)
    return list(result)
