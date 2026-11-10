import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def random_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
        return adj, edges

    def greedy_clique_partition(adj, edges, deadline):
        n = len(adj)
        remaining = set(edges)
        cliques = []
        
        def find_large_clique(remaining_set):
            """Find a large clique whose edges are in remaining_set."""
            if not remaining_set:
                return None
            
            # Try multiple starting edges, pick the one that yields largest clique
            best_clique = None
            best_size = 0
            
            # Sample some edges to try
            edge_list = list(remaining_set)
            # Sort by common neighbors count
            edge_list.sort(key=lambda e: -len(adj[e[0]] & adj[e[1]]))
            
            tried = 0
            for e in edge_list[:min(20, len(edge_list))]:
                if time.time() > deadline - 2:
                    break
                u, v = e
                # Build clique greedily starting from {u,v}
                clique = {u, v}
                # Candidates: common neighbors of all in clique
                cand = adj[u] & adj[v]
                cand -= clique
                
                while cand:
                    # Pick vertex with most connections to cand
                    best_v = max(cand, key=lambda x: len(adj[x] & cand))
                    clique.add(best_v)
                    cand = cand & adj[best_v]
                    cand.discard(best_v)
                
                # Only count edges that are in remaining
                verts = list(clique)
                valid_edges = 0
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        a, b = min(verts[i],verts[j]), max(verts[i],verts[j])
                        if (a,b) in remaining_set:
                            valid_edges += 1
                        else:
                            # This edge is not remaining, so we can't use this clique as-is
                            # We need to restrict to only edges in remaining
                            pass
                
                # Actually, we need the clique to only use edges in remaining
                # Let's re-approach: build clique where all pairwise edges are in remaining
                if valid_edges == len(verts)*(len(verts)-1)//2:
                    if len(verts) > best_size:
                        best_size = len(verts)
                        best_clique = set(verts)
                else:
                    # Need to trim to only use remaining edges
                    # This is more complex; for now, just use the edge itself
                    if len(verts) < 2:
                        continue
                    # Check if the clique's edges are all in remaining
                    all_in = True
                    for i in range(len(verts)):
                        for j in range(i+1, len(verts)):
                            a, b = min(verts[i],verts[j]), max(verts[i],verts[j])
                            if (a,b) not in remaining_set:
                                all_in = False
                                break
                        if not all_in:
                            break
                    if all_in and len(verts) > best_size:
                        best_size = len(verts)
                        best_clique = set(verts)
            
            if best_clique is None:
                # Fallback: just take one edge
                e = next(iter(remaining_set))
                best_clique = set(e)
            
            return best_clique

        while remaining:
            if time.time() > deadline - 2:
                break
            
            clique = find_large_clique(remaining)
            if clique is None:
                break
            
            # Remove all edges in this clique
            verts = list(clique)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    a, b = min(verts[i],verts[j]), max(verts[i],verts[j])
                    remaining.discard((a,b))
            
            cliques.append(verts)
        
        # Handle remaining
        for e in remaining:
            cliques.append(list(e))
        
        return cliques

    def solve_instance(n, p, seed):
        adj, edges = random_graph(n, p, seed)
        return greedy_clique_partition(adj, edges, DEADLINE)

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > DEADLINE - 15:
            _, edges = random_graph(n, p, seed)
            results.append([list(e) for e in edges])
        else:
            results.append(solve_instance(n, p, seed))
    
    return results
