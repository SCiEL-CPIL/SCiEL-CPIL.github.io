import random
from itertools import combinations
from math import lcm

# Precompute bad triples
# bad_conflicts[x] is a set of frozensets({a, b}) such that {a, b, x} is a bad triple
# Actually, storing pairs is enough.
bad_conflicts = {i: set() for i in range(1, 129)}

# Optimization: Use a helper function to check if a,b,c form a bad triple
def is_bad_triple(a, b, c):
    if a == b or b == c or a == c:
        return False
    l_ab = lcm(a, b)
    l_ac = lcm(a, c)
    l_bc = lcm(b, c)
    return l_ab == l_ac == l_bc

# Precompute
# To speed up, we can iterate a, b, c
# But 128^3 is 2M.
# Let's just do it.
for a in range(1, 129):
    for b in range(a + 1, 129):
        for c in range(b + 1, 129):
            if is_bad_triple(a, b, c):
                bad_conflicts[a].add(frozenset((b, c)))
                bad_conflicts[b].add(frozenset((a, c)))
                bad_conflicts[c].add(frozenset((a, b)))

def construct():
    best_set = set()
    # Run multiple greedy attempts with different orderings
    # We can also try sorting by "degree" (number of conflicts) in ascending order
    # Numbers with fewer conflicts are "safer" to add early?
    # Or numbers with more conflicts are more critical?
    # Usually, adding "less constrained" items first is good.
    
    degrees = {i: len(bad_conflicts[i]) for i in range(1, 129)}
    sorted_by_degree = sorted(range(1, 129), key=lambda x: degrees[x])
    
    # Attempt 1: Sorted by degree
    S = set()
    for x in sorted_by_degree:
        can_add = True
        for pair in bad_conflicts[x]:
            a, b = tuple(pair)
            if a in S and b in S:
                can_add = False
                break
        if can_add:
            S.add(x)
    if len(S) > len(best_set):
        best_set = S
        
    # Attempt 2-20: Random shuffles
    for _ in range(20):
        items = list(range(1, 129))
        random.shuffle(items)
        S = set()
        for x in items:
            can_add = True
            for pair in bad_conflicts[x]:
                a, b = tuple(pair)
                if a in S and b in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
        if len(S) > len(best_set):
            best_set = S
            
    return list(best_set)
