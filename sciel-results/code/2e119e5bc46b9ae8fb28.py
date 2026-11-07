import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    odds = [i for i in range(1, 129) if i % 2 != 0]
    evens = [i for i in range(1, 129) if i % 2 == 0]
    
    # Precompute forbidden triples of evens from Odd-Odd-Odd
    forbidden_triples = set()
    for i in range(len(odds)):
        for j in range(i+1, len(odds)):
            for k in range(j+1, len(odds)):
                a, b, c = odds[i], odds[j], odds[k]
                s1, s2, s3 = a+b, a+c, b+c
                if s1 <= 128 and s2 <= 128 and s3 <= 128:
                    forbidden_triples.add((s1, s2, s3))
    
    # Precompute for each even c, the set of sums a+b that are "blocked" by c
    # i.e., if c is in E, then a+b cannot be in E if a+c<=127 and b+c<=127
    # blocked_by[c] = set of a+b
    blocked_by = {e: set() for e in evens}
    for c in evens:
        # a, b in odds, a != b
        # a+c <= 127, b+c <= 127
        # a,b are odds.
        for i in range(len(odds)):
            a = odds[i]
            if a + c > 127:
                continue
            for j in range(len(odds)):
                if i == j:
                    continue
                b = odds[j]
                if b + c > 127:
                    continue
                blocked_by[c].add(a+b)
                
    # Greedy construction
    E = set()
    # Try adding evens in descending order (safer)
    for c in sorted(evens, reverse=True):
        if time.time() > deadline:
            break
            
        # Check if adding c is valid
        valid = True
        
        # 1. Check Odd-Odd-Odd violations
        # c must not form a forbidden triple with any two in E
        # We can check all pairs in E? E is small.
        # Or check all forbidden triples containing c.
        for t in forbidden_triples:
            if c in t:
                # Check if other two are in E
                others = [x for x in t if x != c]
                if others[0] in E and others[1] in E:
                    valid = False
                    break
        
        # 2. Check Mixed violations
        # If c is in E, then for any e in E, e must not be in blocked_by[c]
        # i.e., E & blocked_by[c] must be empty
        if valid:
            if E & blocked_by[c]:
                valid = False
        
        if valid:
            E.add(c)
            
    result = sorted(odds + list(E))
    return result
