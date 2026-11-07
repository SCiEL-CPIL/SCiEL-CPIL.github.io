"""Clique partition of split graphs: Greedy Edge Cover with Maximal Cliques."""
import time
from random import Random
from itertools import combinations

def construct():
    _T0 = time.time()
    _DEADLINE = 130.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))
        
        # Precompute adjacency for I vertices
        N_I = {}
        for v in range(k, n):
            N_I[v] = set(u for u in range(k) if (u, v) in edge_set)

        # Candidate cliques:
        # 1. The full clique C
        # 2. Maximal cliques for each I vertex: {w} U N(w)
        candidates = []
        # C
        candidates.append(list(range(k)))
        
        # K_w for w in I
        for w in range(k, n):
            if N_I[w]:
                candidates.append([w] + sorted(N_I[w]))
        
        # We also need to cover edges that might not be covered by maximal cliques
        # If we don't pick K_w, we will pick {w, u} later.
        # So we can just run greedy on the maximal candidates first.
        
        uncovered = set(edge_set)
        selected_cliques = []
        
        # Greedy selection from maximal candidates
        # To speed up, we can sort candidates by size descending
        candidates.sort(key=lambda c: len(c), reverse=True)
        
        for clique in candidates:
            if not uncovered:
                break
            if _time_up():
                break
                
            # Calculate uncovered edges in this clique
            clique_edges = set()
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                if e in uncovered:
                    clique_edges.add(e)
            
            if not clique_edges:
                continue
                
            # If this clique covers at least one uncovered edge, select it
            # Note: This simple greedy might not be optimal but is a good heuristic
            # It effectively picks the largest clique that has any uncovered edges
            # Since we sorted by size, we pick the largest available one.
            
            selected_cliques.append(clique)
            uncovered -= clique_edges
            
        # Cover remaining edges with single edges
        if uncovered:
            for e in uncovered:
                selected_cliques.append([e[0], e[1]])
        
        cliques = selected_cliques
        
        # Validation
        edge_count = {}
        for clique in cliques:
            if len(clique) < 2:
                continue
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                edge_count[e] = edge_count.get(e, 0) + 1

        missing = [e for e in edge_set if edge_count.get(e, 0) == 0]
        over = [e for e, c in edge_count.items() if c > 1]

        if missing or over:
            # Fallback to valid but potentially suboptimal solution
            cliques = [list(range(k))]
            for (u, v) in edge_set:
                if u >= k or v >= k:
                    cliques.append([u, v])

        # Ensure all cliques have at least 2 vertices
        cliques = [c for c in cliques if len(c) >= 2]
        if not cliques:
            cliques = [list(range(k))]

        # Convert to ensure Python ints
        cliques = [[int(v) for v in c] for c in cliques]

        results.append(cliques)

    return results
