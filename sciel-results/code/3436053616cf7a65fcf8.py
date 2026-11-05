"""C4-free graph with high minimum degree - randomized construction + local search."""
import random
import time
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_c4(adj, n):
        # Check for C4 using bitsets: for each pair of vertices, count common neighbors
        # A C4 exists iff there are two vertices with >=2 common neighbors
        adj_bits = [0] * n
        for i in range(n):
            bits = 0
            for j in adj[i]:
                bits |= (1 << j)
            adj_bits[i] = bits
        for i in range(n):
            for j in range(i+1, n):
                common = adj_bits[i] & adj_bits[j]
                if common & (common - 1):  # at least 2 bits set
                    return True
        return False

    def common_neighbors_count(adj_bits, i, j):
        c = adj_bits[i] & adj_bits[j]
        return bin(c).count('1')

    def min_degree(adj):
        return min(len(adj[i]) for i in range(len(adj)))

    def try_build(n, max_time=5.0):
        """Try to build a C4-free graph on n vertices with min degree >= n-39."""
        target_min_deg = n - 39
        if target_min_deg < 1:
            target_min_deg = 1

        best_edges = None
        best_score = -1

        random.seed(42)

        t0 = time.time()
        attempts = 0

        while time.time() - t0 < max_time and elapsed() < DEADLINE:
            attempts += 1
            # Random regular-ish graph construction
            # Start with random edges, then adjust
            edges = set()
            deg = [0] * n

            # Add random edges until we have a reasonable structure
            # Use a randomized greedy approach
            adj = [set() for _ in range(n)]

            # Random permutation of vertices for ordering
            order = list(range(n))
            random.shuffle(order)

            # Greedily add edges
            for i in range(n):
                for j in range(i+1, n):
                    if len(adj[i]) < target_min_deg + 2 and len(adj[j]) < target_min_deg + 2:
                        # Check if adding edge creates C4
                        # Edge (i,j) creates C4 if there exists k such that k connected to both i and j
                        # and k != i, k != j
                        common = adj[i] & adj[j]
                        if len(common) <= 1:
                            adj[i].add(j)
                            adj[j].add(i)
                            deg[i] += 1
                            deg[j] += 1

            # Now try to boost minimum degree
            # Find vertices with low degree and add edges
            for _ in range(200):
                low = [v for v in range(n) if len(adj[v]) < target_min_deg]
                if not low:
                    break
                for v in low:
                    # Try to add an edge from v to a vertex it's not connected to
                    candidates = [u for u in range(n) if u != v and u not in adj[v]]
                    random.shuffle(candidates)
                    for u in candidates:
                        if len(adj[u]) >= n - 1:
                            continue
                        common = adj[v] & adj[u]
                        if len(common) <= 1:
                            adj[v].add(u)
                            adj[u].add(v)
                            break

            # Remove edges that create C4
            changed = True
            while changed:
                changed = False
                adj_bits = [0] * n
                for i in range(n):
                    bits = 0
                    for j in adj[i]:
                        bits |= (1 << j)
                    adj_bits[i] = bits
                for i in range(n):
                    for j in range(i+1, n):
                        if j in adj[i]:
                            common = adj_bits[i] & adj_bits[j]
                            if common & (common - 1):
                                # Remove edge (i,j)
                                adj[i].discard(j)
                                adj[j].discard(i)
                                changed = True
                                break
                    if changed:
                        break

            # Check C4-free
            adj_bits = [0] * n
            for i in range(n):
                bits = 0
                for j in adj[i]:
                    bits |= (1 << j)
                adj_bits[i] = bits
            c4_free = True
            for i in range(n):
                for j in range(i+1, n):
                    common = adj_bits[i] & adj_bits[j]
                    if common & (common - 1):
                        c4_free = False
                        break
                if not c4_free:
                    break

            if c4_free:
                md = min(len(adj[i]) for i in range(n))
                if md >= target_min_deg:
                    edges = [(min(u,v), max(u,v)) for u in range(n) for v in adj[u] if u < v]
                    return edges

            # Score for tracking
            if c4_free:
                md = min(len(adj[i]) for i in range(n))
                if md > best_score:
                    best_score = md
                    best_edges = [(min(u,v), max(u,v)) for u in range(n) for v in adj[u] if u < v]

        return best_edges

    # Try from n=47 down to n=39
    for n in range(47, 38, -1):
        if elapsed() > DEADLINE:
            break
        target_min_deg = n - 39
        if target_min_deg < 1:
            target_min_deg = 1

        max_time = min(20.0, DEADLINE - elapsed() - 5.0)
        if max_time < 2.0:
            break

        edges = try_build(n, max_time=max_time)

        if edges is not None:
            # Verify
            adj = build_adj(n, edges)
            if min_degree(adj) >= target_min_deg and not has_c4(adj, n):
                return {'n': n, 'edges': edges}

    # Fallback: empty graph on 39 vertices
    return {'n': 39, 'edges': []}
