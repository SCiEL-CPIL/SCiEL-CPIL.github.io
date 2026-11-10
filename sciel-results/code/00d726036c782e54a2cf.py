"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        I = list(range(k, n))
        C = list(range(k))
        
        adj_I = {}
        for i in I:
            nbrs = set()
            for u in C:
                if (u, i) in edges or (i, u) in edges:
                    nbrs.add(u)
            adj_I[i] = nbrs
        
        cliques = []
        used_edges = set()
        
        # For each i in I, create clique covering all its cross edges
        for i in I:
            nbrs = adj_I[i]
            if len(nbrs) >= 2:
                clique = [i] + sorted(nbrs)
                cliques.append(clique)
                for a in range(len(clique)):
                    for b in range(a + 1, len(clique)):
                        u, v = clique[a], clique[b]
                        used_edges.add((min(u, v), max(u, v)))
            elif len(nbrs) == 1:
                v = list(nbrs)[0]
                clique = [i, v]
                cliques.append(clique)
                used_edges.add((min(i, v), max(i, v)))
        
        # Remaining edges: within C, not in used_edges
        remaining = set()
        for u in range(k):
            for v in range(u + 1, k):
                edge = (u, v)
                if edge in edges and edge not in used_edges:
                    remaining.add(edge)
        
        # Cover remaining with cliques on C
        rem_adj = {u: set() for u in range(k)}
        for (u, v) in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        while remaining:
            best_clique = None
            best_size = 0
            for start in range(k):
                if not rem_adj[start]:
                    continue
                clique = [start]
                candidates = set(rem_adj[start])
                while candidates:
                    best_v = max(candidates, key=lambda x: len(rem_adj[x]))
                    if all(best_v in rem_adj[x] for x in clique):
                        clique.append(best_v)
                        candidates = candidates & rem_adj[best_v]
                    else:
                        candidates.discard(best_v)
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique is None or best_size < 2:
                break
            
            cliques.append(sorted(best_clique))
            for a in range(len(best_clique)):
                for b in range(a + 1, len(best_clique)):
                    u, v = best_clique[a], best_clique[b]
                    edge = (min(u, v), max(u, v))
                    if edge in remaining:
                        remaining.discard(edge)
                    if v in rem_adj[u]:
                        rem_adj[u].discard(v)
                    if u in rem_adj[v]:
                        rem_adj[v].discard(u)
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve_split(*spec)
        cleaned = []
        for c in cliques:
            c = sorted(set(int(x) for x in c))
            if len(c) >= 2:
                cleaned.append(c)
        results.append(cleaned)
    
    return results
# EVOLVE-BLOCK-END
