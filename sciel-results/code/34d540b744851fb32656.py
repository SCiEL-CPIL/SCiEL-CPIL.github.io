import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def find_max_clique(adj, remaining_edges, timeout=5.0):
        """Find a maximum clique using Bron-Kerbosch with pivot."""
        active = set()
        for u, v in remaining_edges:
            active.add(u)
            active.add(v)
        if not active:
            return []
        
        best = []
        t_start = time.time()
        
        def bron_kerbosch(R, P, X):
            nonlocal best
            if time.time() - t_start > timeout:
                return
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            U = P | X
            if U:
                pivot = max(U, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)
        
        # Try from each vertex as root to find global max
        # But that's expensive. Instead, use a single call with P = active
        P = set(active)
        X = set()
        bron_kerbosch(set(), P, X)
        
        return best

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
        
        cliques = []
        remaining_edges = set(edge_set)
        
        # Greedy: repeatedly find max clique, add it, remove its edges
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: cover remaining edges individually
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Find max clique in remaining graph
            max_clique = find_max_clique(adj, remaining_edges, timeout=8.0)
            
            if len(max_clique) < 2:
                # Fallback: pick any edge
                u, v = next(iter(remaining_edges))
                cliques.append([u, v])
                remaining_edges.discard((u, v) if u < v else (v, u))
                continue
            
            cliques.append(max_clique)
            
            # Remove all edges within this clique
            for i in range(len(max_clique)):
                for j in range(i + 1, len(max_clique)):
                    u, v = max_clique[i], max_clique[j]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve_split(*spec)
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                clean_res.append(clean_clique)
            results.append(clean_res)
            
    return results
