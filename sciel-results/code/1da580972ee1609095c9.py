import itertools

def construct():
    n = 16
    
    # Step 1: Construct STS(15) using PG(3,2)
    # Points are non-zero vectors in F2^4, represented by integers 1..15
    # Map to indices 0..14
    points = list(range(1, 16))
    
    sts15_triples = []
    seen = set()
    for i in range(len(points)):
        u = points[i]
        for j in range(i+1, len(points)):
            v = points[j]
            w = u ^ v
            if w == 0:
                continue
            # Triple is {u, v, w}
            # Map to 0..14
            t = sorted([u-1, v-1, w-1])
            t_tuple = tuple(t)
            if t_tuple not in seen:
                seen.add(t_tuple)
                sts15_triples.append(t)
                
    # Step 2: Find all 5-caps in STS(15)
    triple_set = set(tuple(sorted(t)) for t in sts15_triples)
    
    def is_cap(subset):
        # subset is a tuple of 5 points
        for t in triple_set:
            if t[0] in subset and t[1] in subset and t[2] in subset:
                return False
        return True
        
    caps = []
    for combo in itertools.combinations(range(15), 5):
        if is_cap(combo):
            caps.append(combo)
            
    # Step 3: Find a hitting set of pairs
    # We want a set of pairs P such that every cap contains at least one pair from P.
    # Greedy approach
    uncovered = set(range(len(caps)))
    P = []
    
    # Pre-compute for each pair, which caps it hits
    pair_to_caps = {}
    for i, cap in enumerate(caps):
        for k in range(5):
            p1 = cap[k]
            for m in range(k+1, 5):
                p2 = cap[m]
                pair = (p1, p2)
                if pair not in pair_to_caps:
                    pair_to_caps[pair] = set()
                pair_to_caps[pair].add(i)
                    
    while uncovered:
        # Find pair that covers most uncovered caps
        best_pair = None
        max_covered = -1
        for pair, cap_indices in pair_to_caps.items():
            covered = len(cap_indices & uncovered)
            if covered > max_covered:
                max_covered = covered
                best_pair = pair
        if best_pair is None:
            break
        P.append(best_pair)
        uncovered -= pair_to_caps[best_pair]
        
    # Step 4: Construct final triples
    final_triples = list(sts15_triples)
    # Point 15 is the new point
    for p1, p2 in P:
        final_triples.append([15, p1, p2])
        
    return final_triples
