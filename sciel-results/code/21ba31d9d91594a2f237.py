import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_maximal_clique_greedy(vertices, adj, remaining_edges, order):
        """
        Find a large clique using a greedy approach based on the given vertex order.
        vertices: list of vertices
        adj: dict of sets, adj[u] contains v if (u,v) is an edge
        remaining_edges: set of edges currently available
        order: list of vertices in the order to try
        """
        best_clique = set()
        best_size = 0
        
        # We try to build a clique starting from each vertex in the order
        # But to be fast, we might just pick the first few promising ones or all if n is small
        # Given n=64, trying all is fine.
        
        for v in order:
            if v not in remaining_vertices_set: # Need to track which vertices still have edges
                continue
            clique = {v}
            candidates = set(adj.get(v, set()))
            # Candidates must also be in remaining graph
            candidates = candidates.intersection(remaining_vertices_set)
            
            # Greedy expansion
            while candidates:
                # Pick candidate with highest degree in the induced subgraph of candidates
                # Or just highest degree in the whole remaining graph
                # For simplicity and speed, let's use degree in remaining graph
                # But we need to ensure connectivity to all in clique
                # Since candidates are neighbors of v, and we intersect with neighbors of new u,
                # they are all connected to v.
                
                # Sort candidates by degree in remaining graph (precomputed or on the fly)
                # To avoid sorting every time, we can just pick one with max degree
                # This is O(|C|) per step.
                
                # Heuristic: Pick u in candidates that has the most neighbors in candidates
                # This is a standard greedy for maximal independent set / clique
                best_u = None
                max_common = -1
                for u in candidates:
                    # Count neighbors of u within candidates
                    common = len(candidates.intersection(adj.get(u, set())))
                    if common > max_common:
                        max_common = common
                        best_u = u
                
                if best_u is None:
                    break
                
                clique.add(best_u)
                candidates = candidates.intersection(adj.get(best_u, set()))
                
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
        
        return best_clique

    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        remaining_vertices_set = set(range(n))
        
        # Remove isolated vertices from consideration for clique start
        # Actually, vertices with no remaining edges can be ignored
        for u in range(n):
            if not any((u, v) in remaining_edges or (v, u) in remaining_edges for v in range(n)):
                 pass # We will handle this in the loop by checking if clique size >= 2
        
        best_partition = None
        best_count = float('inf')
        
        # Multi-start greedy
        # Number of restarts depends on time
        max_restarts = 500
        rng = random.Random(42)
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Create a random order of vertices
            order = list(range(n))
            rng.shuffle(order)
            
            rem_edges = set(remaining_edges)
            rem_adj = {i: set(adj[i]) for i in range(n)} # Deep copy
            # Filter rem_adj to only include vertices that are part of rem_edges
            # This is expensive. Instead, we can just check membership in rem_edges.
            
            partition = []
            current_rem_edges = set(rem_edges)
            
            while current_rem_edges:
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                # Update active vertices
                active_vertices = set()
                for u, v in current_rem_edges:
                    active_vertices.add(u)
                    active_vertices.add(v)
                    
                if not active_vertices:
                    break
                    
                # Find a large clique
                # We use the greedy finder on the subgraph induced by active_vertices
                # But the greedy finder needs to know which edges are present.
                # We can pass current_rem_edges to it?
                # Let's modify the clique finder to work on the current state.
                
                # Simple greedy:
                # Pick a vertex with high degree in current_rem_edges
                # Try to extend it.
                
                best_clique = set()
                best_size = 0
                
                # Try starting from several high-degree vertices
                # Sort active vertices by degree in current_rem_edges
                degs = {v: 0 for v in active_vertices}
                for u, v in current_rem_edges:
                    degs[u] += 1
                    degs[v] += 1
                sorted_vertices = sorted(active_vertices, key=lambda x: degs[x], reverse=True)
                
                # Try top 10 vertices as seeds
                for seed_v in sorted_vertices[:10]:
                    clique = {seed_v}
                    candidates = set()
                    for u, v in current_rem_edges:
                        if u == seed_v:
                            candidates.add(v)
                        elif v == seed_v:
                            candidates.add(u)
                            
                    # Greedy expansion
                    while candidates:
                        # Pick u in candidates with max common neighbors in candidates
                        best_u = None
                        max_common = -1
                        for u in candidates:
                            # Neighbors of u in current_rem_edges
                            neigh_u = set()
                            for a, b in current_rem_edges:
                                if a == u:
                                    neigh_u.add(b)
                                elif b == u:
                                    neigh_u.add(a)
                            common = len(candidates.intersection(neigh_u))
                            if common > max_common:
                                max_common = common
                                best_u = u
                                
                        if best_u is None:
                            break
                            
                        clique.add(best_u)
                        # Update candidates to intersection with neighbors of best_u
                        neigh_best = set()
                        for a, b in current_rem_edges:
                            if a == best_u:
                                neigh_best.add(b)
                            elif b == best_u:
                                neigh_best.add(a)
                        candidates = candidates.intersection(neigh_best)
                        
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = clique
                        
                if best_size < 2:
                    # Should not happen if current_rem_edges is not empty
                    # But if it does, we have an edge that is not in a clique?
                    # Every edge is a clique of size 2.
                    # So best_size should be at least 2.
                    # If best_size is 1, it means no edges?
                    break
                
                partition.append(sorted(best_clique))
                
                # Remove edges of best_clique from current_rem_edges
                clique_edges = set()
                for u, v in combinations(sorted(best_clique), 2):
                    clique_edges.add((u, v))
                current_rem_edges.difference_update(clique_edges)
                
            if current_rem_edges:
                continue # Incomplete partition due to timeout
            
            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = partition
                
        return best_partition

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() - start_time > time_limit:
            # Return whatever we have, or empty if not started
            # But we need to return a valid partition?
            # If we timeout, we should return a valid greedy solution found so far or a trivial one.
            # The trivial one is each edge as a clique.
            edges = generate_edges(n, p, seed)
            trivial = [list(e) for e in edges]
            results.append(trivial)
        else:
            results.append(solve_instance(n, p, seed))
            
    return results

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    time_limit = 115.0  # Safety margin

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        best_partition = None
        best_count = float('inf')
        
        # Multi-start greedy
        max_restarts = 1000
        rng = random.Random(42)
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Create a random order of vertices
            order = list(range(n))
            rng.shuffle(order)
            
            current_rem_edges = set(edges)
            partition = []
            
            while current_rem_edges:
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                # Update active vertices
                active_vertices = set()
                for u, v in current_rem_edges:
                    active_vertices.add(u)
                    active_vertices.add(v)
                    
                if not active_vertices:
                    break
                    
                # Find a large clique
                best_clique = set()
                best_size = 0
                
                # Try starting from several high-degree vertices
                degs = {v: 0 for v in active_vertices}
                for u, v in current_rem_edges:
                    degs[u] += 1
                    degs[v] += 1
                sorted_vertices = sorted(active_vertices, key=lambda x: degs[x], reverse=True)
                
                # Try top 10 vertices as seeds
                for seed_v in sorted_vertices[:10]:
                    clique = {seed_v}
                    candidates = set()
                    for u, v in current_rem_edges:
                        if u == seed_v:
                            candidates.add(v)
                        elif v == seed_v:
                            candidates.add(u)
                            
                    # Greedy expansion
                    while candidates:
                        # Pick u in candidates with max common neighbors in candidates
                        best_u = None
                        max_common = -1
                        # Optimization: Precompute neighbors of candidates in current_rem_edges?
                        # For n=64, this is small.
                        for u in candidates:
                            neigh_u = adj[u].intersection(active_vertices) # Approximation, need to check edges
                            # Actually, we need neighbors that are in current_rem_edges
                            # adj[u] contains all original neighbors.
                            # We need to filter by current_rem_edges.
                            # Let's just iterate current_rem_edges for each u? Too slow.
                            # Let's maintain a dynamic adjacency for current_rem_edges?
                            # Or just check if edge exists in current_rem_edges.
                            
                            # Count common neighbors in candidates
                            common = 0
                            for w in candidates:
                                if (u, w) in current_rem_edges or (w, u) in current_rem_edges:
                                    common += 1
                            
                            if common > max_common:
                                max_common = common
                                best_u = u
                                
                        if best_u is None:
                            break
                            
                        clique.add(best_u)
                        # Update candidates to intersection with neighbors of best_u
                        neigh_best = set()
                        for a, b in current_rem_edges:
                            if a == best_u:
                                neigh_best.add(b)
                            elif b == best_u:
                                neigh_best.add(a)
                        candidates = candidates.intersection(neigh_best)
                        
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = clique
                        
                if best_size < 2:
                    break
                
                partition.append(sorted(best_clique))
                
                # Remove edges of best_clique from current_rem_edges
                clique_edges = set()
                for u, v in combinations(sorted(best_clique), 2):
                    clique_edges.add((u, v))
                current_rem_edges.difference_update(clique_edges)
                
            if current_rem_edges:
                continue # Incomplete partition due to timeout
            
            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = partition
                
        if best_partition is None:
            # Fallback to trivial partition
            best_partition = [list(e) for e in edges]
            
        return best_partition

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() - start_time > time_limit:
            edges = generate_edges(n, p, seed)
            trivial = [list(e) for e in edges]
            results.append(trivial)
        else:
            results.append(solve_instance(n, p, seed))
            
    return results
# EVOLVE-BLOCK-END
