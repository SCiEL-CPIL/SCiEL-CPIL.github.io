import time

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 60
    
    n = P['n']
    primes = [a for a in range(2, n + 1) if all(a % d for d in range(2, int(a**0.5) + 1))]
    S = set(primes)
    
    # Precompute conflict pairs for each a
    # conflict[a] is a set of (b, c)
    # We only care about b, c in the current universe (2..256)
    conflict = {a: set() for a in range(2, n + 1)}
    for a in range(2, n + 1):
        for b in range(2, n + 1):
            if b == a: continue
            for c in range(2, n + 1):
                if c == a or c == b: continue
                if (b * c) % a == 0:
                    conflict[a].add((b, c))
                    
    # Local Search
    # Try to add each non-prime
    others = [a for a in range(2, n + 1) if a not in S]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in others:
            if x in S: continue
            
            # Check if x can be added
            # x conflicts with (b, c) if b, c in S and (b*c) % x == 0
            # Also, for any a in S, if (a*b) % x == 0? No, x is the divisor.
            # The condition is: for all distinct a,b,c in S_new, a does not divide bc.
            # New elements are just x.
            # So we need to check:
            # 1. x does not divide b*c for any b,c in S.
            # 2. a does not divide x*c for any a,c in S (a!=x, c!=x, a!=c).
            # 3. a does not divide b*x for any a,b in S.
            
            valid = True
            conflicts = []
            
            # Check 1: x | b*c
            for b, c in conflict[x]:
                if b in S and c in S:
                    valid = False
                    # Record the conflict to potentially resolve
                    conflicts.append((b, c))
                    break
            
            if not valid:
                # Try to remove one element from a conflict to make x valid
                # This is complex. Simple heuristic: skip x.
                continue
            
            # Check 2 & 3: a | x*c or a | b*x
            # This is equivalent to: for any a in S, is there c in S such that a | x*c?
            # And for any a in S, is there b in S such that a | b*x?
            
            # If we add x, we must ensure no new conflicts.
            # New conflicts involve x.
            # a | x*c => x*c = k*a.
            # a | b*x => b*x = k*a.
            
            # We can just check all triples involving x.
            # Triples: (x, b, c), (a, x, c), (a, b, x).
            
            # (x, b, c): x | b*c. Checked above.
            
            # (a, x, c): a | x*c.
            for a in S:
                for c in S:
                    if c == a: continue
                    if (x * c) % a == 0:
                        valid = False
                        break
                if not valid: break
            
            if not valid: continue
            
            # (a, b, x): a | b*x.
            for a in S:
                for b in S:
                    if b == a: continue
                    if (b * x) % a == 0:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                S.add(x)
                improved = True
                
    return list(S)
