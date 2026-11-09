import time
import random

def construct():
    start_time = time.time()
    
    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = set(get_graph(n, p, seed))
        if not edges:
            return []
        
        # Build adjacency list for fast lookups
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() - start_time > 115:
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
            C = [u, v]
            candidates = adj[u] & adj[v]
            
            # Sort candidates by degree in remaining graph
            candidates = list(candidates)
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            # Greedily build clique
            changed = True
            while changed:
                changed = False
                for idx, cand in enumerate(candidates):
                    if all(cand in adj[c] for c in C):
                        C.append(cand)
                        changed = True
                        candidates = [x for x in candidates if x != cand and x in adj[cand]]
                        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
                        break
            
            # Add clique to result
            cliques.append(C)
            
            # Remove all edges within C from the graph
            for i in range(len(C)):
                for j in range(i+1, len(C)):
                    a, b = C[i], C[j]
                    if a > b:
                        a, b = b, a
                    edge = (a, b)
                    if edge in remaining_edges:
                        remaining_edges.discard(edge)
                        adj[a].discard(b)
                        adj[b].discard(a)
            
        return cliques

    def local_improve(cliques, n, adj, edges_set):
        """Try to merge adjacent cliques or rearrange vertices to reduce count."""
        if not cliques:
            return cliques
        
        improved = True
        while improved:
            if time.time() - start_time > 112:
                break
            improved = False
            for i in range(len(cliques)):
                for j in range(i+1, len(cliques)):
                    # Check if all vertices in clique i connect to all vertices in clique j
                    can_merge = True
                    for a in cliques[i]:
                        for b in cliques[j]:
                            if b not in adj[a]:
                                can_merge = False
                                break
                        if not can_merge:
                            break
                    if can_merge:
                        cliques[i] = cliques[i] + cliques[j]
                        del cliques[j]
                        improved = True
                        break
                if improved:
                    break
        return cliques

    results = []
    # Instance 1
    results.append(solve_instance(32, 0.6, 601))
    
    # Instance 2
    results.append(solve_instance(64, 0.55, 602))
    
    return results
