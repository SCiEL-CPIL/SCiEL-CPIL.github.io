import time
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe under 90s limit
    
    # Precompute for each pair (b, c) with b <= c, which a values would violate
    # a < b, a divides (b+c)
    # For a given set S, a in S is valid iff for all b,c in S with a < b <= c, a does not divide b+c
    
    # Start with upper tail
    S = set(range(171, 257))
    
    # Precompute: for each a, list of (b, c) pairs where a < b <= c and a | (b+c)
    # Actually, let's precompute for each a, the set of sums b+c that are divisible by a
    # For conflict checking: when adding a, check if any pair (b,c) in S with a < b <= c has a | (b+c)
    # Also check triples where a is middle or largest
    
    # Let's build a conflict graph approach
    # Two elements x, y are "compatible" if there exists no z such that adding all three would violate
    # Actually, the constraint is on triples, so it's not just pairwise
    
    # Better approach: greedy with local search
    # Try to add elements from small to large, checking validity
    
    def is_valid_set(S):
        """Check if S is a valid set."""
        elements = sorted(S)
        for i, a in enumerate(elements):
            for j in range(i + 1, len(elements)):
                b = elements[j]
                for k in range(j, len(elements)):
                    c = elements[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Greedy: try to add elements from 1 to 256
    # Start with upper tail, then try to add smaller elements
    for a in range(1, 171):
        if time.time() > deadline - 5:
            break
        # Check if adding a to S maintains validity
        valid = True
        elements = sorted(S)
        for i, b in enumerate(elements):
            if b <= a:
                continue
            for j in range(i, len(elements)):
                c = elements[j]
                if c < b:
                    continue
                # Check triple (a, b, c) with a < b <= c
                if (b + c) % a == 0:
                    valid = False
                    break
                # Check triple (b, a, c) with b < a <= c -- but b > a here, so skip
                # Check triple (b, c, a) with b < c < a -- but c >= b > a, so c > a, skip
            if not valid:
                break
        if valid:
            S.add(a)
    
    # Local search: try swapping elements
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_size = len(S)
        for a in range(1, 171):
            if a in S:
                continue
            # Find conflicts: elements that prevent a from being added
            conflicts = set()
            elements = sorted(S)
            for i, b in enumerate(elements):
                if b <= a:
                    continue
                for j in range(i, len(elements)):
                    c = elements[j]
                    if c < b:
                        continue
                    if (b + c) % a == 0:
                        conflicts.add(b)
                        conflicts.add(c)
            
            # Try removing conflicts and adding a
            if len(conflicts) == 0:
                S.add(a)
                improved = True
            elif len(conflicts) < 2 and len(S) - len(conflicts) + 1 > len(S):
                # Only worth it if we gain net elements
                new_S = S - conflicts
                new_S.add(a)
                if len(new_S) > len(S):
                    S = new_S
                    improved = True
    
    # Try to remove elements and add multiple smaller ones
    # Branch and bound style: try removing one element, see if we can add more
    for a in range(1, 171):
        if time.time() > deadline - 10:
            break
        if a in S:
            continue
        # What elements conflict with a?
        conflicts = set()
        elements = sorted(S)
        for i, b in enumerate(elements):
            if b <= a:
                continue
            for j in range(i, len(elements)):
                c = elements[j]
                if c < b:
                    continue
                if (b + c) % a == 0:
                    conflicts.add(b)
                    conflicts.add(c)
        
        # If conflicts are few, try removing them
        if 0 < len(conflicts) <= 3:
            new_S = S - conflicts
            new_S.add(a)
            # Check if new_S is valid
            if is_valid_set(new_S) and len(new_S) > len(S):
                S = new_S
    
    # Final local search: try all single swaps
    for a in range(1, 171):
        if time.time() > deadline - 2:
            break
        if a in S:
            continue
        for b in S:
            if b <= a:
                continue
            # Try swapping b for a
            new_S = S - {b}
            new_S.add(a)
            if is_valid_set(new_S) and len(new_S) > len(S):
                S = new_S
                break
    
    return list(S)
