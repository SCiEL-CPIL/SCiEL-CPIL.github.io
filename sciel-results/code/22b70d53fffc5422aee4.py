"""Combined program: precomputed triples + fast greedy + local search + swaps."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under the 90s kill
    N = 180
    rng = random.Random(20240517)

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = b*c / (b+c), so (b+c) | (b*c).
    # Since 1/a = 1/b + 1/c > 1/c, we have a < c. Similarly a < b.
    # So a is always the smallest of the three distinct elements.
    
    triples = []  # list of (a, b, c) with a < b < c
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if a < b and 1 <= a <= N:
                    triples.append((a, b, c))
    
    # For fast checking, build: for each a, the set of (b, c) pairs that form a triple with a.
    triple_map = {}
    for a, b, c in triples:
        if a not in triple_map:
            triple_map[a] = set()
        triple_map[a].add((b, c))
    
    # Also build reverse maps for faster incremental checks
    # For each x, which (a, y) pairs would form a triple if x is the middle or largest element
    # Specifically: if x is 'b', then for each a < x, check if (x, c) in triple_map[a] for some c
    # If x is 'c', then for each a < x, check if (b, x) in triple_map[a] for some b
    
    # Build: for each element x, the list of (a, other) such that adding x to a set containing a and other creates a triple
    # When adding x:
    #   - If x is 'a': need (b,c) in triple_map[x] both in S
    #   - If x is 'b': need exists a < x and c > x with (a, x, c) in triples, i.e., c in triple_map[a] and c > x
    #   - If x is 'c': need exists a < x and b < x with (a, b, x) in triples, i.e., b in triple_map[a] and b < x
    
    # For efficient checking when adding x to set S:
    # Check 1: x as 'a' - for each (b,c) in triple_map.get(x, []), if b in S and c in S -> conflict
    # Check 2: x as 'b' - for each a in S where a < x, for each (b,c) in triple_map[a], if b == x and c in S -> conflict
    #            Equivalently: precompute for each x, the set of (a, c) where (a, x, c) is a triple
    # Check 3: x as 'c' - for each a in S where a < x, for each (b,c) in triple_map[a], if c == x and b in S -> conflict
    #            Equivalently: precompute for each x, the set of (a, b) where (a, b, x) is a triple
    
    # Precompute: as_b[x] = set of (a, c) such that (a, x, c) is a forbidden triple (x is the middle element)
    # Precompute: as_c[x] = set of (a, b) such that (a, b, x) is a forbidden triple (x is the largest element)
    as_b = {}  # x -> set of (a, c)
    as_c = {}  # x -> set of (a, b)
    for a, b, c in triples:
        if b not in as_b:
            as_b[b] = set()
        as_b[b].add((a, c))
        if c not in as_c:
            as_c[c] = set()
        as_c[c].add((a, b))
    
    def has_triple(S):
        """Check if set S contains any forbidden triple."""
        for a in S:
            if a in triple_map:
                for (b, c) in triple_map[a]:
                    if b in S and c in S:
                        return True
        return False
    
    def can_add(S, x):
        """Check if adding x to S creates a forbidden triple. Faster than has_triple(S | {x})."""
        # x as 'a'
        if x in triple_map:
            for (b, c) in triple_map[x]:
                if b in S and c in S:
                    return False
        # x as 'b'
        if x in as_b:
            for (a, c) in as_b[x]:
                if a in S and c in S:
                    return False
        # x as 'c'
        if x in as_c:
            for (a, b) in as_c[x]:
                if a in S and b in S:
                    return False
        return True
    
    # Greedy construction with a given order
    def greedy_order(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    # Phase 1: Multiple greedy constructions with different orderings
    best = set(range(91, N + 1))  # baseline: 90 elements
    best_size = len(best)
    
    # Descending order
    cand = greedy_order(list(range(N, 0, -1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Ascending order
    cand = greedy_order(list(range(1, N + 1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Random orderings
    for _ in range(200):
        if time.time() >= deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = greedy_order(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
    
    # Phase 2: Local search - try to add elements
    for _ in range(2000):
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            if can_add(best, x):
                best.add(x)
                best_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Phase 3: Swap search - remove one element, add another
    for _ in range(50000):
        if time.time() >= deadline:
            break
        if not best:
            break
        x = rng.randint(1, N)
        if x in best:
            continue
        y = rng.choice(sorted(best))
        # Remove y, try to add x
        best.discard(y)
        if can_add(best, x):
            best.add(x)
        else:
            # Put y back
            best.add(y)
    
    # Phase 4: More local search after swaps
    for _ in range(2000):
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            if can_add(best, x):
                best.add(x)
                best_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Phase 5: Random restarts with diverse seeds
    for _ in range(5000):
        if time.time() >= deadline:
            break
        size = rng.randint(60, 120)
        s = set(rng.sample(range(1, N + 1), size))
        if has_triple(s):
            continue
        remaining = [x for x in range(1, N + 1) if x not in s]
        rng.shuffle(remaining)
        for x in remaining:
            if time.time() >= deadline:
                break
            if can_add(s, x):
                s.add(x)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Final verification
    if has_triple(best):
        best = set(range(91, N + 1))
    
    return list(best)
# EVOLVE-BLOCK-END
