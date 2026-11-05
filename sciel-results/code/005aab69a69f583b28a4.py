def construct():
    from itertools import combinations
    from random import Random

    def generate_edges(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def get_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def greedy_initial_partition(n, edges):
        if not edges:
            return []
        
        adj = get_adjacency(n, edges)
        remaining_edges = set(edges)
        cliques = []
        
        def find_large_clique(remaining_edges, adj, n):
            if not remaining_edges:
                return []
            
            deg = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            max_deg = max(deg.values())
            candidates = [v for v in range(n) if deg[v] == max_deg]
            start = candidates[0]
            
            clique = [start]
            neighbors = set(adj[start]) & set(v for e in remaining_edges for v in e if v != start)
            
            while neighbors:
                best = None
                best_score = -1
                for v in neighbors:
                    if all(w in adj[v] for w in clique):
                        score = deg[v]
                        if score > best_score:
                            best_score = score
                            best = v
                if best is None:
                    break
                clique.append(best)
                neighbors = {v for v in neighbors if all(w in adj[v] for w in clique)}
            
            return clique
        
        while remaining_edges:
            clique = find_large_clique(remaining_edges, adj, n)
            if not clique or len(clique) < 2:
                edge = next(iter(remaining_edges))
                cliques.append(frozenset(edge))
                remaining_edges.remove(edge)
            else:
                clique_edges = set(tuple(sorted(pair)) for pair in combinations(clique, 2))
                if clique_edges.issubset(remaining_edges):
                    cliques.append(frozenset(clique))
                    remaining_edges -= clique_edges
                else:
                    edge = next(iter(remaining_edges))
                    cliques.append(frozenset(edge))
                    remaining_edges.remove(edge)
        
        return cliques

    def local_search_merge(cliques, adj, n):
        # Convert to list of sets for mutation
        cliques = [set(c) for c in cliques]
        changed = True
        while changed:
            changed = False
            # Try to merge any pair
            # To be efficient, we can check pairs. n is small (32, 64).
            # Number of cliques is small.
            i = 0
            while i < len(cliques):
                j = i + 1
                while j < len(cliques):
                    c1 = cliques[i]
                    c2 = cliques[j]
                    # Check if union is a clique
                    # They must share at most 1 vertex to be edge-disjoint in original, 
                    # but since they are from a valid partition, they share at most 1 vertex.
                    # We just need to check if all pairs in union are edges.
                    union = c1 | c2
                    is_clique = True
                    # Optimization: if len(union) is small, check all pairs.
                    # Since they are disjoint or share 1, we check cross pairs.
                    # Pairs within c1 are edges. Pairs within c2 are edges.
                    # We only need to check pairs (u,v) with u in c1, v in c2.
                    for u in c1:
                        for v in c2:
                            if u == v:
                                continue
                            if v not in adj[u]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    
                    if is_clique:
                        # Merge
                        cliques[i] = union
                        cliques.pop(j)
                        changed = True
                        # Reset j to i+1 to check new clique against others
                        j = i + 1
                        continue
                    j += 1
                i += 1
        
        return [frozenset(c) for c in cliques]

    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        adj = get_adjacency(n, edges)
        cliques = greedy_initial_partition(n, edges)
        cliques = local_search_merge(cliques, adj, n)
        return [list(c) for c in cliques]

    return [solve_instance(n, p, seed) for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602))]
