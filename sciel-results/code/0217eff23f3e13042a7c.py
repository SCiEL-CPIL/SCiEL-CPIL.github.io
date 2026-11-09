"""Clique partition of dense random graphs: hardest-edge-first greedy + local merge."""
import time
from random import Random


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

    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    remaining_edges = set(edges)
    cliques = []

    while remaining_edges:
        if time.time() > deadline:
            # Fallback: add remaining edges as 2-cliques
            for e in remaining_edges:
                cliques.append(list(e))
            break

        # Find the "hardest" edge: one with the fewest common neighbors
        # This helps form larger cliques in sparse regions
        best_edge = None
        best_common = float('inf')
        for e in remaining_edges:
            u, v = e
            common = len(adj[u] & adj[v])
            if common < best_common:
                best_common = common
                best_edge = e
                if common <= 0:
                    break

        if best_edge is None:
            break

        u, v = best_edge
        # Build maximal clique starting from edge (u, v)
        C = [u, v]
        # Candidates: common neighbors of all vertices in C that are in remaining edges
        P = adj[u] & adj[v]
        P -= {u, v}

        # Greedy expansion: repeatedly add the vertex with most connections to C
        while P:
            if time.time() > deadline:
                break
            # Find candidate with most neighbors in C
            best_cand = None
            best_score = -1
            for cand in P:
                score = len(adj[cand] & set(C))
                if score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand is None:
                break
            C.append(best_cand)
            P = P & adj[best_cand]

        # Add clique to result
        cliques.append(C)

        # Remove all edges within C from the graph
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                a, b = C[i], C[j]
                if a > b:
                    a, b = b, a
                edge = (a, b)
                if edge in remaining_edges:
                    remaining_edges.discard(edge)
                    adj[a].discard(b)
                    adj[b].discard(a)

    # Local improvement: try to merge cliques if their union is a clique
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_cliques = []
        i = 0
        while i < len(cliques):
            C1 = set(cliques[i])
            merged = False
            for j in range(i + 1, len(cliques)):
                if time.time() > deadline:
                    return cliques
                C2 = set(cliques[j])
                # Check if union is a clique: every vertex in C1 connects to every vertex in C2
                can_merge = True
                for a in C1:
                    for b in C2:
                        if b not in adj[a]:
                            can_merge = False
                            break
                    if not can_merge:
                        break
                if can_merge:
                    C1.update(C2)
                    cliques[i] = list(C1)
                    del cliques[j]
                    merged = True
                    break
            if not merged:
                new_cliques.append(cliques[i])
            i += 1
        cliques = new_cliques
        if not merged and not changed:
            break

    return cliques


def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
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
