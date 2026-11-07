"""
Design: Two disjoint K_{3,3} components (on 0-5 and 6-11).
K_{3,3} is bipartite, so it has no odd cycles.
Every cycle in K_{3,3} has length 4 or 6.
Thus, the set of cycle lengths is {4, 6} -> 2 distinct lengths.
Min degree is 3. Girth is 4.
Score: seed / 2. Assuming seed is 5, score = 2.5.
"""

def construct():
    edges = []
    # First K_{3,3}
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges.append([u, v])
    # Second K_{3,3}
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
BET: {mechanism: "Constructs two disjoint K_{3,3} components, which have only cycle lengths 4 and 6, minimizing distinct cycle lengths to 2", q: 0.9, delta: [1.5, 1.5]}
