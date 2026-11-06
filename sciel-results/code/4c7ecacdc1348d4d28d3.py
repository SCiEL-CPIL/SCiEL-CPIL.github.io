"""Decompose graphs into minimum number of cycles and single edges."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict
    import sys
    sys.setrecursionlimit(10000)

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, n_nodes):
        best_cycle = []
        
        for start in range(n_nodes):
            if len(adj[start]) < 2:
                continue
            
            path = [start]
            visited = {start}
            
            def dfs(u):
                nonlocal best_cycle
                # Pruning: Check if we can beat the best cycle
                # Count unvisited nodes greater than start
                unvisited_greater = 0
                for v in range(start + 1, n_nodes):
                    if v not in visited:
                        unvisited_greater += 1
                
                if len(path) + unvisited_greater <= len(best_cycle):
                    return
                
                # Sort neighbors to ensure deterministic behavior and potentially better pruning
                # Higher degree nodes first? Or just sorted?
                # Let's try sorting by degree descending to find "core" cycles first?
                # Or just sorted indices.
                neighbors = sorted(adj[u], key=lambda x: -len(adj[x]))
                
                for v in neighbors:
                    if v == start:
                        if len(path) >= 3 and len(path) > len(best_cycle):
                            best_cycle = path[:]
                            # If we found a Hamiltonian cycle, we can stop early?
                            if len(best_cycle) == n_nodes:
                                return
                    elif v > start and v not in visited:
                        visited.add(v)
                        path.append(v)
                        dfs(v)
                        path.pop()
                        visited.remove(v)
                        
            dfs(start)
            
        return best_cycle if len(best_cycle) >= 3 else None

    def decompose(edges, n):
        remaining = set(edges)
        pieces = []
        
        def build_adj(rem):
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        while remaining:
            adj = build_adj(remaining)
            cycle = find_longest_cycle(adj, n)
            
            if cycle:
                pieces.append(cycle)
                # Remove edges of the cycle
                L = len(cycle)
                for i in range(L):
                    u = cycle[i]
                    v = cycle[(i + 1) % L]
                    edge = (u, v) if u < v else (v, u)
                    remaining.discard(edge)
            else:
                break
        
        for u, v in remaining:
            pieces.append([u, v])
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
