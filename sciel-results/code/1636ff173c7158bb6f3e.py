"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    """
    Return a nonempty subset of [1, 180] with no THREE DISTINCT elements a,b,c
    satisfying 1/a = 1/b + 1/c. Maximize cardinality.
    
    Approach:
    1. Enumerate all forbidden triples (a, b, c) with distinct elements where 1/a = 1/b + 1/c.
       Note: 1/a = 1/b + 1/c  <=>  bc = a(b+c)  <=>  1/a = 1/b + 1/c.
       Since a, b, c are distinct positive integers in [1, 180], we can enumerate all such triples.
       
    2. This is a 3-uniform hypergraph independent set problem. We want the largest subset with no hyperedge fully contained.
       Since this is NP-hard in general, we use a greedy heuristic combined with local search.
       
    However, note that the equation 1/a = 1/b + 1/c with a < b < c (WLOG for distinct positive) implies:
       bc = a(b+c)  =>  1/a = 1/b + 1/c
       =>  (b-a)(c-a) = a^2  [by rearranging: bc - ab - ac = 0 => bc - ab - ac + a^2 = a^2 => (b-a)(c-a) = a^2]
       
    So for each a, we can find all factor pairs (d, e) of a^2 such that d*e = a^2, then b = a+d, c = a+e.
    We need b, c <= 180 and distinct from a and each other.
    
    Let's build the list of forbidden triples.
    Then use a greedy approach: start with all numbers, and iteratively remove elements from the most constrained triples,
    or use a max-3SAT style approach.
    
    Given the small size (180 elements), we can try a smarter approach:
    - The set [91, 180] has 90 elements. Does it contain any forbidden triple?
      For a >= 91, b > a, c > a, so b, c > 91. But (b-a)(c-a) = a^2 >= 91^2 = 8281.
      b-a >= 1, c-a >= 1. If b-a = 1, then c-a = a^2, so c = a + 1 + a^2 > 180 for a >= 91 (since 91+1+8281 >> 180).
      So no triple with a >= 91 exists where all three are in [91, 180]. Thus [91, 180] is valid with size 90.
    
    Can we do better? Let's try to include some numbers below 91.
    
    Let's enumerate all forbidden triples and use a maximum independent set heuristic on the 3-uniform hypergraph.
    """
    N = 180
    # Build forbidden triples: sets of 3 distinct elements {a, b, c} such that 1/a = 1/b + 1/c
    # WLOG assume a < b < c. Then (b-a)(c-a) = a^2.
    forbidden = []
    for a in range(1, N + 1):
        a2 = a * a
        # Find all divisors d of a2
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b > N or c > N:
                    continue
                if b == c:
                    continue
                if b == a or c == a:
                    continue
                # Ensure distinct
                if len({a, b, c}) != 3:
                    continue
                triple = tuple(sorted((a, b, c)))
                forbidden.append(triple)
    
    # Remove duplicates
    forbidden = list(set(forbidden))
    
    # Now we want the largest subset of [1, N] with no forbidden triple fully contained.
    # This is the maximum independent set in a 3-uniform hypergraph.
    # Use a greedy + local search approach.
    
    # Start with all elements
    S = set(range(1, N + 1))
    
    # Greedy: while there's a forbidden triple in S, remove the element from it that appears in the most other triples (or just any)
    # Better: use a priority queue based on degree in the hypergraph of remaining triples
    
    from collections import defaultdict
    
    # Build adjacency: for each element, list of triples containing it
    elem_to_triples = defaultdict(list)
    for t in forbidden:
        for x in t:
            elem_to_triples[x].append(t)
    
    # We'll use a simple local search: start with S = [1..N], repeatedly find a violating triple and remove the "best" element
    # Then try to add back elements.
    
    # Actually, let's try a different approach: since the hypergraph is sparse, we can try to find a large independent set using a greedy algorithm that picks elements in order of increasing "constraint".
    
    # Alternative: use the fact that for large a, there are no triples. So [91, 180] is safe. Let's try to extend it.
    
    # Let's try: S = set(range(91, 181))  # size 90
    # Try to add elements from 1 to 90 one by one if they don't create a forbidden triple.
    
    S = set(range(91, N + 1))
    
    # Try to add each element from 1 to 90
    for x in range(1, 91):
        # Check if adding x creates a forbidden triple
        # A forbidden triple containing x would be {x, y, z} where y, z in S
        can_add = True
        for t in elem_to_triples[x]:
            # Check if the other two elements are in S
            others = [v for v in t if v != x]
            if len(others) == 2 and all(v in S for v in others):
                can_add = False
                break
        if can_add:
            S.add(x)
    
    # Now try local search: for each element not in S, try adding it; for each element in S, try removing it and see if we can add more
    # Simple improvement: try to swap
    
    improved = True
    while improved:
        improved = False
        # Try adding each element not in S
        for x in range(1, N + 1):
            if x in S:
                continue
            can_add = True
            for t in elem_to_triples[x]:
                others = [v for v in t if v != x]
                if len(others) == 2 and all(v in S for v in others):
                    can_add = False
                    break
            if can_add:
                S.add(x)
                improved = True
        # Try removing an element from S to allow adding more
        for x in sorted(S):
            S_without = S - {x}
            # Count how many new elements we can add
            new_count = 0
            for y in range(1, N + 1):
                if y in S_without:
                    continue
                can_add = True
                for t in elem_to_triples[y]:
                    others = [v for v in t if v != y]
                    if len(others) == 2 and all(v in S_without for v in others):
                        can_add = False
                        break
                if can_add:
                    new_count += 1
            if new_count > 1:  # Net gain
                S = S_without
                improved = True
                break
    
    return sorted(S)
# EVOLVE-BLOCK-END
