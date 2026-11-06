import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_max_clique_bk(vertices, adj, time_limit=None):
        """Bron-Kerbosch with pivot to find maximum clique."""
        best_clique = []
        vertices_set = set(vertices)
        
        def bk(R, P, X):
            nonlocal best_clique
            if time_limit is not None and time.time() > time_limit:
                return
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = R.copy()
                return
            if not P:
                return
            
            # Choose pivot u from P union X to maximize |N(u) ∩ P|
            pivot = None
            max_count = -1
            for u in P:
                count = len(adj.get(u, set()) & P)
                if count > max_count:
                    max_count = count
                    pivot = u
            for u in X:
                count = len(adj.get(u, set()) & P)
                if count > max_count:
                    max_count = count
                    pivot = u
            
            if pivot is None:
                return
            
            # Recurse on P \ N(pivot)
            candidates = P - adj.get(pivot, set())
            for v in list(candidates):
                if time_limit is not None and time.time() > time_limit:
                    return
                bk(R + [v], P & adj.get(v, set()), X & adj.get(v, set()))
                P.remove(v)
                X.add(v)
        
        bk([], set(vertices), set())
        return best_clique

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency for the full graph
        full_adj = {u: set() for u in range(n)}
        for u, v in edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        
        remaining_edges = set(edges)
        partition = []
        
        while remaining_edges and time.time() < deadline:
            # Get active vertices
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
            
            # Build subgraph adjacency from remaining edges
            sub_adj = {u: set() for u in active_vertices}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            # Find a large clique using BK with time limit
            time_limit = min(deadline - 1, time.time() + 2)
            clique = find_max_clique_bk(list(active_vertices), sub_adj, time_limit)
            
            if not clique or len(clique) < 2:
                # Fallback: pick an edge
                u, v = next(iter(remaining_edges))
                clique = [u, v]
            
            # Verify it's actually a clique in remaining_edges
            valid = True
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if (a, b) not in remaining_edges and (b, a) not in remaining_edges:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Fallback to edge
                u, v = next(iter(remaining_edges))
                clique = [u, v]
            
            partition.append(clique)
            
            # Remove all edges within this clique
            clique_set = set(clique)
            to_remove = set()
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a < b:
                        to_remove.add((a, b))
                    else:
                        to_remove.add((b, a))
            remaining_edges -= to_remove
        
        # If we ran out of time, add remaining edges as single edges
        if remaining_edges and time.time() >= deadline:
            for u, v in remaining_edges:
                partition.append([u, v])
        
        return partition

    result1 = solve_instance(32, 0.6, 601, 60)
    result2 = solve_instance(64, 0.55, 602, 65)
    
    return [result1, result2]
