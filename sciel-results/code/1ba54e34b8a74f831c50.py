"""Clique partition of dense random graphs: maximal-clique greedy with edge-pool maintenance."""
import time
from random import Random
from collections import defaultdict

def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    if not edges:
        return []

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Remaining edges as a set for O(1) membership
    remaining = set(edges)
    cliques = []

    # Helper to check if a set of vertices forms a clique in the remaining edge set
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if (min(u, v), max(u, v)) not in remaining:
                    return False
        return True

    # Greedy: repeatedly find a large clique in the remaining graph
    # Strategy: pick a vertex, find its neighbors in remaining, then find a large clique in that subgraph
    # For efficiency, use a simple greedy clique finder on the induced subgraph of a vertex's neighborhood

    while remaining:
        if time.time() > deadline:
            for e in remaining:
                cliques.append(list(e))
            break

        # Pick a starting vertex: choose one with high degree in remaining graph
        # For speed, sample if many edges
        if len(remaining) > 500:
            # Sample vertices
            sample_edges = list(remaining)
            rng2 = Random(12345)
            sample_edges = rng2.sample(sample_edges, min(200, len(sample_edges)))
            deg = defaultdict(int)
            for u, v in sample_edges:
                deg[u] += 1
                deg[v] += 1
            if deg:
                start_v = max(deg.keys(), key=lambda x: deg[x])
            else:
                start_v = next(iter(remaining))[0]
        else:
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            if deg:
                start_v = max(deg.keys(), key=lambda x: deg[x])
            else:
                start_v = next(iter(remaining))[0]

        # Find neighbors of start_v in remaining graph
        neighbors = []
        for w in adj[start_v]:
            e_key = (min(start_v, w), max(start_v, w))
            if e_key in remaining:
                neighbors.append(w)

        if not neighbors:
            # No edges from start_v, just take any edge
            e = next(iter(remaining))
            cliques.append(list(e))
            remaining.discard(e)
            continue

        # Now find a large clique in the subgraph induced by {start_v} ∪ neighbors
        # Greedy: start with start_v, add neighbors that are connected to all current clique members
        clique = [start_v]
        # Sort neighbors by degree in the remaining graph (heuristic for larger cliques)
        def neighbor_degree(w):
            count = 0
            for x in adj[w]:
                e_key = (min(x, w), max(x, w))
                if e_key in remaining:
                    count += 1
            return count

        neighbors.sort(key=neighbor_degree, reverse=True)

        for w in neighbors:
            # Check if w is connected to all current clique members in remaining
            can_add = True
            for x in clique:
                e_key = (min(x, w), max(x, w))
                if e_key not in remaining:
                    can_add = False
                    break
            if can_add:
                clique.append(w)

        # Remove all edges within this clique from remaining
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                e_key = (min(u, v), max(u, v))
                remaining.discard(e_key)

        if len(clique) >= 2:
            cliques.append(clique)

    return cliques

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = _solve_instance(n, p, seed, deadline)
        result.append(cliques)
        if time.time() > deadline - 10:
            # Not enough time for second instance, return edges for second
            edges = _random_graph(n, p, seed)
            result.append([list(e) for e in edges])
            break
    return result
