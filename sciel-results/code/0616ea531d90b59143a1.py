import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 180
    
    # Find all forbidden triples (a, b, c) with a < b < c, all in [1,180],
    # such that 1/a = 1/b + 1/c, i.e., a(b+c) = bc
    # Equivalent: (b-a)(c-a) = a^2
    
    forbidden = []  # list of (a, b, c) with a < b < c
    for a in range(1, n + 1):
        a2 = a * a
        # b = a + d1, c = a + d2, d1 * d2 = a2, d1 < d2 (to ensure b < c)
        # d1 and d2 are divisors of a2 with d1 < d2
        for d1 in range(1, int(a2 ** 0.5) + 1):
            if a2 % d1 == 0:
                d2 = a2 // d1
                if d1 == d2:
                    continue  # b == c, not allowed
                b = a + d1
                c = a + d2
                if b <= n and c <= n and b != c:
                    forbidden.append((a, b, c))
    
    # Now we have all forbidden triples. We want max subset of [1,180] with no triple fully contained.
    # This is maximum independent set in a 3-uniform hypergraph.
    # 
    # Strategy: Use a greedy approach with backtracking, or use the fact that
    # for each a, including a forbids certain pairs (b,c).
    #
    # Alternative: Model as a 2-SAT-like problem? No, it's 3-uniform.
    #
    # Let's use a greedy algorithm: process elements in some order, include if possible.
    # Order by "constraint tightness" - elements involved in more forbidden triples first.
    
    # Count how many forbidden triples each element is in
    from collections import defaultdict
    triple_count = defaultdict(int)
    for a, b, c in forbidden:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1
    
    # Sort elements by number of triples they're in (most constrained first)
    elements = list(range(1, n + 1))
    elements.sort(key=lambda x: -triple_count[x])
    
    # Greedy: include element if it doesn't complete any forbidden triple
    included = set()
    for elem in elements:
        # Check if including elem would create a forbidden triple
        conflict = False
        for a, b, c in forbidden:
            if elem in (a, b, c):
                other_two = [x for x in (a, b, c) if x != elem]
                if other_two[0] in included and other_two[1] in included:
                    conflict = True
                    break
        if not conflict:
            included.add(elem)
    
    if time.time() > deadline:
        return sorted(included)
    
    # Try to improve: local search - try to add excluded elements, remove included if needed
    excluded = set(range(1, n + 1)) - included
    
    # Try to add each excluded element
    for elem in sorted(excluded, key=lambda x: -triple_count[x]):
        if time.time() > deadline:
            break
        # Check if elem can be added
        conflict = False
        for a, b, c in forbidden:
            if elem in (a, b, c):
                other_two = [x for x in (a, b, c) if x != elem]
                if other_two[0] in included and other_two[1] in included:
                    conflict = True
                    break
        if not conflict:
            included.add(elem)
    
    return sorted(included)
