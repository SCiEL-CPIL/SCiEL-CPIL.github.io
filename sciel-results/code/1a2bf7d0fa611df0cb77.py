import random
import time

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Initialize with the star at 0
    triples = set()
    for a in range(1, n):
        for b in range(a + 1, n):
            triples.add((0, a, b))
            
    # Precompute all possible triples not containing 0
    candidates = []
    for a in range(1, n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                candidates.append((a, b, c))
                
    random.shuffle(candidates)
    
    # Precompute pair-to-third-points mapping for faster constraint checking
    # For each pair (x, y), store the set of z such that (x, y, z) is in triples
    pair_map = {}
    for t in triples:
        x, y, z = t
        # Add all three pair mappings
        for p1, p2, p3 in [(x, y, z), (x, z, y), (y, z, x)]:
            key = (min(p1, p2), max(p1, p2))
            if key not in pair_map:
                pair_map[key] = set()
            pair_map[key].add(p3)
    
    def get_thirds(x, y):
        """Get set of z such that (x,y,z) is in triples"""
        key = (min(x, y), max(x, y))
        return pair_map.get(key, set())
    
    def check_add(a, b, c):
        # Check if adding (a,b,c) creates a K(2,2,2)
        # For K(2,2,2) with transversals {a,b,c}, {a,b,c'}, {a,b',c}, {a,b',c'}, {a',b,c}, {a',b,c'}, {a',b',c}, {a',b',c'}
        # We need: (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c') all in triples
        
        # For each a' != a, b' != b, c' != c:
        # The required triples are:
        # (a, b, c') - c' in get_thirds(a, b)
        # (a, b', c) - c in get_thirds(a, b')
        # (a, b', c') - c' in get_thirds(a, b')
        # (a', b, c) - c in get_thirds(a', b)
        # (a', b, c') - c' in get_thirds(a', b)
        # (a', b', c) - c in get_thirds(a', b')
        # (a', b', c') - c' in get_thirds(a', b')
        
        # Strategy: For each pair (a', b'), find c' values that satisfy all constraints
        # Actually, let's iterate more cleverly
        
        # For each a' != a:
        for a_prime in range(n):
            if a_prime == a:
                continue
            
            # For each b' != b:
            for b_prime in range(n):
                if b_prime == b:
                    continue
                
                # Check if (a', b', c) is in triples (i.e., c in get_thirds(a', b'))
                if c not in get_thirds(a_prime, b_prime):
                    continue
                
                # Check if (a, b', c) is in triples (i.e., c in get_thirds(a, b'))
                if c not in get_thirds(a, b_prime):
                    continue
                
                # Check if (a', b, c) is in triples (i.e., c in get_thirds(a', b))
                if c not in get_thirds(a_prime, b):
                    continue
                
                # Now we need c' != c such that:
                # (a, b, c') in triples: c' in get_thirds(a, b)
                # (a, b', c') in triples: c' in get_thirds(a, b')
                # (a', b, c') in triples: c' in get_thirds(a', b)
                # (a', b', c') in triples: c' in get_thirds(a', b')
                
                # Find intersection of these sets, excluding c
                set1 = get_thirds(a, b)
                set2 = get_thirds(a, b_prime)
                set3 = get_thirds(a_prime, b)
                set4 = get_thirds(a_prime, b_prime)
                
                # Find common elements in all four sets, excluding c
                common = set1 & set2 & set3 & set4
                if c in common:
                    common.discard(c)
                
                if common:
                    return False  # Found a c' that completes the K(2,2,2)
        
        return True

    for t in candidates:
        if time.time() > timeout:
            break
        a, b, c = t
        if check_add(a, b, c):
            triples.add((a, b, c))
            # Update pair_map
            for p1, p2, p3 in [(a, b, c), (a, c, b), (b, c, a)]:
                key = (min(p1, p2), max(p1, p2))
                if key not in pair_map:
                    pair_map[key] = set()
                pair_map[key].add(p3)

    # Convert to list of lists for return
    result = [list(t) for t in triples]
    return result
