import time
from itertools import combinations

def construct():
    start_time = time.time()
    
    # Primes p ≡ 3 (mod 4), sorted by size
    # Check smaller ones first; 67 is the smallest known valid Paley tournament for k=4
    # However, we must check all smaller candidates to ensure minimality.
    candidates = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]
    
    for p in candidates:
        if time.time() - start_time > 110:
            break
            
        # Generate Paley tournament: i -> j iff (j - i) mod p is a quadratic residue
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # Build adjacency bitmasks for efficient dominator checks
        out = [0] * p
        for i in range(p):
            mask = 0
            for j in range(p):
                if i != j and ((j - i) % p) in squares:
                    mask |= (1 << j)
            out[i] = mask
        
        # Verify: every 4-set must have a common dominator (vertex with arcs to all 4)
        valid = True
        for a, b, c, d in combinations(range(p), 4):
            # Common dominators are vertices that point to a, b, c, d
            dom = out[a] & out[b] & out[c] & out[d]
            # Exclude the four vertices themselves (they can't dominate themselves)
            dom &= ~((1 << a) | (1 << b) | (1 << c) | (1 << d))
            if dom == 0:
                valid = False
                break
        
        if valid:
            # Construct orientation list in lexicographic order
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
    
    # Fallback: should not be reached, but return 83 to be safe
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
