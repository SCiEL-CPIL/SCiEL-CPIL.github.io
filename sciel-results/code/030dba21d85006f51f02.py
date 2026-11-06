<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K5 (a K5 has no six-cycle).
    edges = []
    for k in range(8):
        edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    return edges
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n = 40
    start_time = time.time()
    deadline = 120
    safety_margin = 10

    def time_left():
        return deadline - safety_margin - (time.time() - start_time)

    # Start with 8 disjoint K5s as base
    adj = [set() for _ in range(n)]
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)

    best_edges = list(edges)
    best_count = len(edges)

    def would_create_c6(u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 through (u,v): u-a-b-c-d-v-u
        Need path of length 4 from u to v: u-a-b-c-d-v
        """
        # For each a in N(u), b in N(a)\{u}, c in N(b)\{a}, check if v in N(c)\{b}
        for a in adj[u]:
            for b in adj[a]:
                if b == u:
                    continue
                for c in adj[b]:
                    if c == a:
                        continue
                    if v in adj[c] and v != b:
                        return True
        return False

    # Try adding edges greedily in random order
    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)

    while time_left() > 0:
        # Try to add more edges
        for u, v in all_pairs:
            if time_left() <= 0:
                break
            if v in adj[u]:
                continue
            if not would_create_c6(u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
        # If we didn't add any edges in a full pass, break
        if len(edges) == best_count and not any(v not in adj[u] for u, v in all_pairs if not would_create_c6(u, v) and time_left() > 0):
            break

    return best_edges
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, the above approach has issues with the loop logic. Let me rewrite it more carefully:

