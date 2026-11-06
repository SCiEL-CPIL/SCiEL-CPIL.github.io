import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    
    # Initialize with the star at 0: all triples (0, a, b) for 1 <= a < b <= 15
    triples = set()
    for a in range(1, n):
        for b in range(a + 1, n):
            triples.add((0, a, b))
    
    # Precompute candidates: all triples not containing 0
    candidates = []
    for a in range(1, n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                candidates.append((a, b, c))
    
    import random
    random.shuffle(candidates)
    
    # Data structure for fast lookup:
    # pair_sets[i][j] = set of k such that sorted((i,j,k)) in triples
    # We'll use a dict: (i,j) -> set of k, but need to handle unordered pairs
    # Actually, let's use: for each unordered pair {i,j}, store set of k
    # Key: (min(i,j), max(i,j)) -> set of k
    
    pair_to_k = {}
    for t in triples:
        a, b, c = t
        pairs = [(a,b), (a,c), (b,c)]
        for p in pairs:
            key = (min(p), max(p))
            k = a + b + c - p[0] - p[1]
            if key not in pair_to_k:
                pair_to_k[key] = set()
            pair_to_k[key].add(k)
    
    def check_add(a, b, c):
        """Check if adding (a,b,c) creates a K(2,2,2)."""
        # We need to find a' != a, b' != b, c' != c such that all 7 other transversals exist
        # The 7 other transversals are:
        # (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
        
        # Get candidate c' values: c' such that (a,b,c') in triples
        key_ab = (min(a,b), max(a,b))
        if key_ab in pair_to_k:
            c_candidates = pair_to_k[key_ab] - {c}
        else:
            c_candidates = set()
        
        # Get candidate b' values: b' such that (a,b',c) in triples
        key_ac = (min(a,c), max(a,c))
        if key_ac in pair_to_k:
            b_candidates = pair_to_k[key_ac] - {b}
        else:
            b_candidates = set()
        
        # Get candidate a' values: a' such that (a',b,c) in triples
        key_bc = (min(b,c), max(b,c))
        if key_bc in pair_to_k:
            a_candidates = pair_to_k[key_bc] - {a}
        else:
            a_candidates = set()
        
        # For each combination of a', b', c', check if all 7 exist
        for a_prime in a_candidates:
            for b_prime in b_candidates:
                for c_prime in c_candidates:
                    # Check the remaining 4: (a,b',c'), (a',b,c'), (a',b',c), (a',b',c')
                    # (a,b',c') : key (min(a,b'), max(a,b')), check c'
                    key1 = (min(a, b_prime), max(a, b_prime))
                    if key1 not in pair_to_k or c_prime not in pair_to_k[key1]:
                        continue
                    # (a',b,c') : key (min(a',b), max(a',b)), check c'
                    key2 = (min(a_prime, b), max(a_prime, b))
                    if key2 not in pair_to_k or c_prime not in pair_to_k[key2]:
                        continue
                    # (a',b',c) : key (min(a',b'), max(a',b')), check c
                    key3 = (min(a_prime, b_prime), max(a_prime, b_prime))
                    if key3 not in pair_to_k or c not in pair_to_k[key3]:
                        continue
                    # (a',b',c') : key (min(a',b'), max(a',b')), check c'
                    if key3 not in pair_to_k or c_prime not in pair_to_k[key3]:
                        continue
                    
                    # All 7 exist! Adding (a,b,c) would create K(2,2,2)
                    return False
        
        return True
    
    for t in candidates:
        if time.time() > deadline:
            break
        a, b, c = t
        if check_add(a, b, c):
            triples.add((a, b, c))
            # Update pair_to_k
            pairs = [(a,b), (a,c), (b,c)]
            for p in pairs:
                key = (min(p), max(p))
                k = a + b + c - p[0] - p[1]
                if key not in pair_to_k:
                    pair_to_k[key] = set()
                pair_to_k[key].add(k)
    
    # Convert to list of lists for return
    result = [list(t) for t in triples]
    return result
