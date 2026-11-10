"""Clique partition of split graphs using structural properties."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Identify cross edges and internal edges
        # internal_edges: set of (u, v) with u < v < k
        # cross_neighbors: dict i -> set of c in C such that (c, i) is edge
        
        internal_edges = set()
        cross_neighbors = {i: set() for i in range(k, n)}
        
        for u, v in edges:
            if u < k and v < k:
                # Internal edge
                if u > v:
                    u, v = v, u
                internal_edges.add((u, v))
            elif (u < k and v >= k) or (u >= k and v < k):
                # Cross edge
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                cross_neighbors[i].add(c)
        
        cliques = []
        
        # Track blocked internal edges
        blocked_edges = set()
        
        # Process each i in I
        for i in range(k, n):
            N_i = cross_neighbors[i]
            if not N_i:
                continue
                
            # Create clique N_i U {i}
            clique = list(N_i) + [i]
            cliques.append(clique)
            
            # Block all internal edges within N_i
            N_i_list = list(N_i)
            for idx in range(len(N_i_list)):
                u = N_i_list[idx]
                for jdx in range(idx + 1, len(N_i_list)):
                    v = N_i_list[jdx]
                    edge = (u, v) if u < v else (v, u)
                    blocked_edges.add(edge)
        
        # Cover remaining internal edges
        # Remaining edges are in internal_edges but not in blocked_edges
        remaining_edges = internal_edges - blocked_edges
        
        if remaining_edges:
            # Find connected components in the graph (C, remaining_edges)
            # Vertices are 0..k-1
            # Use Union-Find or BFS
            
            # Build adjacency for remaining edges
            adj = {u: set() for u in range(k)}
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            visited = set()
            components = []
            
            for u in range(k):
                if u in visited or not adj[u]:
                    continue
                # BFS
                component = []
                queue = [u]
                visited.add(u)
                while queue:
                    node = queue.pop(0)
                    component.append(node)
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if len(component) >= 2:
                    components.append(component)
            
            # Add cliques for each component
            for comp in components:
                cliques.append(comp)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        # Ensure types are Python int and list
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
