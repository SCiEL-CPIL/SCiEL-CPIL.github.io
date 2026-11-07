def construct():
    import time
    start = time.time()
    DEADLINE = 85.0
    
    N = 256
    numbers = list(range(2, N+1))
    n = len(numbers)
    
    # For each pair (b,c), compute which a's divide b*c
    # Or: for each a, which pairs (b,c) have a|b*c
    
    # Precompute: for each a, set of (b,c) pairs where a|b*c, b<c, b!=a, c!=a
    a_conflicts = {}
    for a in numbers:
        pairs = []
        for i in range(n):
            b = numbers[i]
            if b == a:
                continue
            for j in range(i+1, n):
                c = numbers[j]
                if c == a:
                    continue
                if (b * c) % a == 0:
                    pairs.append((b, c))
        a_conflicts[a] = set(pairs)
    
    # Greedy: try adding numbers in order of "least restrictive" first
    # A number is less restrictive if it has fewer conflicts as divisor
    # But also, it should not be part of many conflicts as b or c
    
    # Let's try: sort by number of conflicts (as divisor), add smallest first? 
    # Or add largest first (large numbers are less likely to divide products)?
    
    # Try: add numbers from largest to smallest
    included = set()
    for a in sorted(numbers, reverse=True):
        # Check if adding a creates any conflict
        # Conflict exists if there exist b,c in included such that a|b*c, or b|a*c, or c|a*b
        # Since b,c > a (we're going from large to small), b|a*c and c|a*b are less likely
        # Actually b,c > a, so b|a*c implies b|c (since gcd(b,a)=1 if b>a and a<b, not necessarily)
        
        # Let's just check all three roles
        conflict = False
        for b in included:
            for c in included:
                if b == c:
                    continue
                # Check if a|b*c, b|a*c, c|a*b
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            included.add(a)
    
    return sorted(included)
