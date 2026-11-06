import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s limit

    def random_graph_edges(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add(frozenset((u, v)))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for e in edges:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Pivot selection: choose vertex in P∪X with max neighbors in P
            if P or X:
                pivot_candidates = P | X
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_clique_partition(edges, n, adj, time_limit):
        uncovered = set(edges)
        if not uncovered:
            return []
        
        partition = []
        max_iterations = 100000
        
        iteration = 0
        while uncovered and iteration < max_iterations:
            if time.time() > time_limit:
                break
            iteration += 1
            
            # Build subgraph of uncovered edges
            sub_adj = [set() for _ in range(n)]
            for e in uncovered:
                u, v = tuple(e)
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            # Find maximal cliques in subgraph
            cliques = bron_kerbosch_maximal_cliques(sub_adj, n)
            
            if not cliques:
                # Should not happen if uncovered non-empty, but fallback
                for e in uncovered:
                    u, v = tuple(e)
                    partition.append([min(u,v), max(u,v)])
                break
            
            # Pick clique covering most uncovered edges
            best_clique = None
            best_edge_count = 0
            for c in cliques:
                # Count edges in c that are uncovered
                cnt = 0
                verts = list(c)
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        if frozenset((verts[i], verts[j])) in uncovered:
                            cnt += 1
                if cnt > best_edge_count:
                    best_edge_count = cnt
                    best_clique = c
            
            if best_clique is None or best_edge_count < 1:
                for e in uncovered:
                    u, v = tuple(e)
                    partition.append([min(u,v), max(u,v)])
                break
            
            # Add best clique, remove its edges
            verts = list(best_clique)
            partition.append(verts)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    uncovered.discard(frozenset((verts[i], verts[j])))
        
        # Any remaining edges as 2-cliques
        for e in uncovered:
            u, v = tuple(e)
            partition.append([min(u,v), max(u,v)])
        
        return partition

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph_edges(n, p, seed)
        adj = build_adj(edges, n)
        result = greedy_clique_partition(edges, n, adj, time_limit)
        return result

    results = []
    remaining_time = DEADLINE - time.time()
    time_per_instance = remaining_time / 2
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        t_start = time.time()
        t_limit = t_start + time_per_instance
        result = solve_instance(n, p, seed, t_limit)
        results.append(result)
    
    return results
