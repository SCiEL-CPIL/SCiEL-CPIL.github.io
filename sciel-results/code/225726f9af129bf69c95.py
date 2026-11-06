import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))

        cliques = []
        remaining_edges = set(edge_set)

        # Precompute all maximal cliques using Bron-Kerbosch with pivot
        # For n up to 64, this is feasible
        def bron_kerbosch(R, P, X):
            """Find all maximal cliques using Bron-Kerbosch with pivot."""
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            # Choose pivot u from P ∪ X with maximum degree in P
            pivot_candidates = list(P | X)
            if not pivot_candidates:
                return
            u = max(pivot_candidates, key=lambda x: len(P & adj[x]))
            
            # Recurse on P \ N(u)
            for v in list(P - adj[u]):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)

        # Find all maximal cliques
        bron_kerbosch(set(), set(range(n)), set())

        if not cliques:
            # Fallback: each edge as a clique
            for e in remaining_edges:
                cliques.append(list(e))
            return cliques

        # Now we need to select a subset of cliques that covers all edges exactly once
        # This is a set partition problem. Use greedy approach.
        
        # For each clique, compute which edges it covers
        clique_edges = []
        for clique in cliques:
            ce = set()
            for u, v in combinations(clique, 2):
                e = (u, v) if u < v else (v, u)
                if e in edge_set:
                    ce.add(e)
            clique_edges.append(ce)

        # Greedy: repeatedly pick the clique that covers the most remaining edges
        # and doesn't conflict with already-selected cliques
        selected = []
        used_edges = set()
        
        # Sort cliques by size descending for better greedy performance
        clique_order = sorted(range(len(cliques)), key=lambda i: -len(clique_edges[i]))
        
        remaining = set(edge_set)
        
        while remaining:
            if time.time() > deadline:
                for e in remaining:
                    selected.append(list(e))
                break
            
            best_idx = -1
            best_count = 0
            best_clique_edges = None
            
            for i in clique_order:
                ce = clique_edges[i]
                # Check if this clique's edges are all still remaining
                # (i.e., none of them have been used by previously selected cliques)
                if ce & used_edges:
                    continue
                count = len(ce & remaining)
                if count > best_count:
                    best_count = count
                    best_idx = i
                    best_clique_edges = ce
            
            if best_idx == -1:
                # No clique covers any remaining edge without conflict
                # Pick individual edges
                for e in remaining:
                    selected.append(list(e))
                remaining.clear()
                break
            
            selected.append(cliques[best_idx])
            used_edges |= best_clique_edges
            remaining -= best_clique_edges

        return selected

    c1 = solve(random_graph(32, 0.6, 601))
    c2 = solve(random_graph(64, 0.55, 602))

    return [c1, c2]

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 140

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))

        cliques = []
        remaining_edges = set(edge_set)

        # Find all maximal cliques using Bron-Kerbosch with pivot
        def bron_kerbosch(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            pivot_candidates = list(P | X)
            if not pivot_candidates:
                return
            u = max(pivot_candidates, key=lambda x: len(P & adj[x]))
            
            for v in list(P - adj[u]):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)

        bron_kerbosch(set(), set(range(n)), set())

        if not cliques:
            for e in remaining_edges:
                cliques.append(list(e))
            return cliques

        # Compute edges covered by each clique
        clique_edges = []
        for clique in cliques:
            ce = set()
            for u, v in combinations(clique, 2):
                e = (u, v) if u < v else (v, u)
                if e in edge_set:
                    ce.add(e)
            clique_edges.append(ce)

        # Greedy set partition: pick clique covering most remaining edges
        selected = []
        used_edges = set()
        clique_order = sorted(range(len(cliques)), key=lambda i: -len(clique_edges[i]))
        remaining = set(edge_set)

        while remaining:
            if time.time() > deadline:
                for e in remaining:
                    selected.append(list(e))
                break
            
            best_idx = -1
            best_count = 0
            best_clique_edges = None
            
            for i in clique_order:
                ce = clique_edges[i]
                if ce & used_edges:
                    continue
                count = len(ce & remaining)
                if count > best_count:
                    best_count = count
                    best_idx = i
                    best_clique_edges = ce
            
            if best_idx == -1:
                for e in remaining:
                    selected.append(list(e))
                remaining.clear()
                break
            
            selected.append(cliques[best_idx])
            used_edges |= best_clique_edges
            remaining -= best_clique_edges

        return selected

    c1 = solve(random_graph(32, 0.6, 601))
    c2 = solve(random_graph(64, 0.55, 602))

    return [c1, c2]
# EVOLVE-BLOCK-END
