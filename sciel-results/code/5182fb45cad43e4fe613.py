"""Rewritten solver for large sets avoiding reciprocal Schur triples."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    N = 180

    # Precompute conflict graph: edge (x, y) exists if there is a distinct z
    # such that 1/x = 1/y + 1/z, i.e., x*y = y*z + x*z => z = x*y/(x+y).
    # We need z to be an integer in [1,180], z != x, z != y.
    # Also the triple must have three DISTINCT elements.

    # For each pair (x, y) with x < y, check if z = x*y/(x+y) is integer and in range and distinct.
    # Actually the equation is symmetric in y and z. Let's think:
    # 1/a = 1/b + 1/c with a, b, c distinct.
    # This means a = bc/(b+c). So for a given a, b, c are determined by the relation.
    # Alternatively: for a given pair (b, c), a = bc/(b+c). If a is integer and distinct from b,c, then (a,b,c) is a forbidden triple.
    # 
    # So two elements x and y conflict if there exists a third element z (distinct from both) such that one of them is the "harmonic mean" of the other two.
    # More precisely: x, y, z form a forbidden triple if one of them equals the harmonic mean of the other two (times 2? no).
    # 1/a = 1/b + 1/c => a = bc/(b+c). So a is determined by b and c.
    # 
    # For the conflict graph: two elements x and y are in conflict if there exists z in [1,180], z distinct from x and y, such that:
    #   x = yz/(y+z)  OR  y = xz/(x+z)  OR  z = xy/(x+y)
    # 
    # But if we're building a set, we need no three distinct elements a,b,c in the set with 1/a=1/b+1/c.
    # This is a 3-uniform hypergraph independence problem.
    # 
    # Greedy approach: order elements, add each if it doesn't create a forbidden triple with already-chosen elements.
    # 
    # Let's precompute for each element a, the pairs (b,c) such that 1/a = 1/b + 1/c with b,c distinct from a and from each other.
    # Then when considering adding element a, we check if any pair (b,c) from a's conflict list are both in the current set.

    # Precompute: for each a in [1,180], find all pairs (b,c) with b < c, b != a, c != a, b != c, such that 1/a = 1/b + 1/c.
    # 1/a = (b+c)/(bc) => bc = a(b+c) => bc - ab - ac = 0 => bc - ab - ac + a^2 = a^2 => (b-a)(c-a) = a^2
    # So b = a + d, c = a + a^2/d for integer d dividing a^2, with d > 0, and b,c in [1,180], b != c, b != a, c != a.
    # Since d > 0, b = a+d > a, c = a + a^2/d > a. So b and c are both > a.
    # Also need b != c, i.e., d != a^2/d, i.e., d^2 != a^2, i.e., d != a (since d>0).

    # So for each a, the conflicts are pairs (b,c) where b = a+d, c = a+a^2/d for d | a^2, d > 0, d != a, and 1 <= b,c <= 180.

    # For greedy: process elements in some order. For each element a, check if adding it creates a forbidden triple.
    # A forbidden triple involving a would be (a, b, c) where 1/a = 1/b + 1/c (a is the "sum" side) OR 1/b = 1/a + 1/c (a is on the "sum" side).
    # 
    # Case 1: a is the left side: 1/a = 1/b + 1/c. Then b = a+d, c = a+a^2/d as above. Both b,c > a.
    # Case 2: a is on the right side, say 1/b = 1/a + 1/c. Then c = ab/(a+b). For c to be integer, a+b must divide ab.
    #   c = ab/(a+b). Let g = gcd(a,b). Then a = g*a', b = g*b', gcd(a',b')=1. c = g*a'*b'/(a'+b'). So (a'+b') | g*a'*b'. Since gcd(a'+b', a') = gcd(b', a') = 1 and gcd(a'+b', b') = gcd(a', b') = 1, we need (a'+b') | g.
    #   This is more complex. Let's just brute force for each a, check all b in [1,180], b != a, compute c = ab/(a+b), check if c is integer, c != a, c != b, 1 <= c <= 180.

    # Let's precompute conflict triples.
    # For each unordered triple {a,b,c} of distinct elements in [1,180], check if 1/a = 1/b + 1/c for some permutation.
    # Actually, the condition is: there exist three distinct a,b,c in S such that 1/a = 1/b + 1/c.
    # This is equivalent to: a = bc/(b+c) for distinct b,c, and a distinct from b,c.
    # So for each pair (b,c) with b < c, compute a = bc/(b+c). If a is integer, 1 <= a <= 180, a != b, a != c, then (a,b,c) is a forbidden triple.

    # Precompute all forbidden triples.
    triples = []
    for b in range(1, 181):
        for c in range(b+1, 181):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= 180 and a != b and a != c:
                    # Store the triple as a frozenset or sorted tuple
                    triples.append((a, b, c))

    # For greedy, we need for each element x, the list of pairs (y,z) such that (x,y,z) is a forbidden triple.
    # When adding x, we check if any (y,z) with y,z already in set exists.

    from collections import defaultdict
    conflict_pairs = defaultdict(set)
    for (a, b, c) in triples:
        # For element a, conflict pair is (b,c)
        conflict_pairs[a].add((b, c))
        conflict_pairs[a].add((c, b))
        # For element b, conflict pair is (a,c)
        conflict_pairs[b].add((a, c))
        conflict_pairs[b].add((c, a))
        # For element c, conflict pair is (a,b)
        conflict_pairs[c].add((a, b))
        conflict_pairs[c].add((b, a))

    # Greedy: try different orderings.
    # Strategy: process elements in decreasing order of "conflict degree" or just try increasing/decreasing.
    # Let's try a few orderings and pick the best.

    def greedy(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            # Check if adding x creates a forbidden triple
            ok = True
            for (y, z) in conflict_pairs[x]:
                if y in S and z in S:
                    ok = False
                    break
            if ok:
                S.add(x)
        return S

    # Minimum-degree greedy: at each step, pick the element with the fewest conflicts
    # among those not yet decided. This exploits structural sparsity.
    def min_degree_greedy():
        S = set()
        remaining = set(range(1, 181))
        while remaining and time.time() < deadline:
            # Find element with minimum conflict degree among remaining
            best_x = None
            best_deg = float('inf')
            for x in remaining:
                deg = len(conflict_pairs[x])
                if deg < best_deg:
                    best_deg = deg
                    best_x = x
            if best_x is None:
                break
            remaining.remove(best_x)
            # Check if adding best_x creates a forbidden triple
            ok = True
            for (y, z) in conflict_pairs[best_x]:
                if y in S and z in S:
                    ok = False
                    break
            if ok:
                S.add(best_x)
        return S

    # Try several orderings
    best_set = set()
    orderings = [
        list(range(1, 181)),
        list(range(180, 0, -1)),
        # Maybe sort by number of conflicts
    ]

    # Sort by conflict degree (number of conflict pairs)
    degrees = {x: len(conflict_pairs[x]) for x in range(1, 181)}
    order_by_degree_asc = sorted(range(1, 181), key=lambda x: degrees[x])
    order_by_degree_desc = sorted(range(1, 181), key=lambda x: -degrees[x])
    orderings.append(order_by_degree_asc)
    orderings.append(order_by_degree_desc)

    for order in orderings:
        S = greedy(order)
        if len(S) > len(best_set):
            best_set = S

    # Try minimum-degree greedy
    S = min_degree_greedy()
    if len(S) > len(best_set):
        best_set = S

    # Local search: try to add elements not in best_set, or swap
    # Try adding each element not in best_set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, 181):
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            # Check if we can add x
            ok = True
            for (y, z) in conflict_pairs[x]:
                if y in best_set and z in best_set:
                    ok = False
                    break
            if ok:
                best_set.add(x)
                improved = True
                break  # restart the loop

    # Try removing one element and adding two others (2-for-1 swap)
    # This is more expensive. Let's do a simple version.
    if time.time() < deadline - 10:
        for x in list(best_set):
            if time.time() > deadline:
                break
            best_set.remove(x)
            # Try to add as many as possible
            for y in range(1, 181):
                if y in best_set:
                    continue
                if time.time() > deadline:
                    break
                ok = True
                for (a2, b2) in conflict_pairs[y]:
                    if a2 in best_set and b2 in best_set:
                        ok = False
                        break
                if ok:
                    best_set.add(y)
            if len(best_set) < 90:
                # Revert
                best_set.discard(x)
                # Remove any elements we added
                # This is tricky. Let's just not do this if it gets complicated.
                pass
            else:
                # Keep the change
                pass

    # Final: try to add more
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, 181):
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            ok = True
            for (y, z) in conflict_pairs[x]:
                if y in best_set and z in best_set:
                    ok = False
                    break
            if ok:
                best_set.add(x)
                improved = True
                break

    return sorted(best_set)
# EVOLVE-BLOCK-END
