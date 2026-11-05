import time
import sys
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        """Generate edges of a random graph with given parameters."""
        edges = set()
        rng = random.Random(seed)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition(n, edges_set):
        """Find a clique partition using vertex-centric greedy and merge post-processing."""
        if not edges_set:
            return []
            
        remaining_edges = set(edges_set)
        partition = []
        
        # Helper to check if a set of vertices forms a clique
        def is_clique(verts, edge_set):
            if len(verts) < 2:
                return True
            v_list = list(verts)
            for i in range(len(v_list)):
                for j in range(i + 1, len(v_list)):
                    u, v = v_list[i], v_list[j]
                    if u > v:
                        u, v = v, u
                    if (u, v) not in edge_set:
                        return False
            return True

        # Helper to find a large clique in a set of vertices using greedy
        def find_large_clique(candidates, edge_set):
            if not candidates:
                return set()
            
            # Greedy: pick vertex with max degree in the induced subgraph
            # We approximate degree by counting neighbors within candidates
            # Since candidates change, we do this iteratively
            current_candidates = set(candidates)
            clique = set()
            
            while current_candidates:
                # Find vertex in current_candidates with max degree to other vertices in current_candidates
                best_v = None
                best_deg = -1
                for v in current_candidates:
                    deg = 0
                    for u in current_candidates:
                        if u == v: continue
                        if (min(u, v), max(u, v)) in edge_set:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                if best_v is None:
                    break
                
                clique.add(best_v)
                # Remove best_v and its non-neighbors from current_candidates
                non_neighbors = set()
                for u in current_candidates:
                    if u != best_v:
                        if (min(u, best_v), max(u, best_v)) not in edge_set:
                            non_neighbors.add(u)
                
                current_candidates = current_candidates - {best_v} - non_neighbors
                
            return clique

        while remaining_edges:
            if time.time() - start_time > DEADLINE:
                # Fallback: remaining edges as 2-cliques
                for e in remaining_edges:
                    partition.append(list(e))
                break
            
            # Compute degrees in the remaining graph
            degrees = {v: 0 for v in range(n)}
            adj = [set() for _ in range(n)]
            for (u, v) in remaining_edges:
                degrees[u] += 1
                degrees[v] += 1
                adj[u].add(v)
                adj[v].add(u)
            
            # Pick vertex with max degree
            if not any(deg > 0 for deg in degrees.values()):
                break # Should not happen if remaining_edges is not empty
                
            v = max(range(n), key=lambda x: degrees[x])
            
            # Get neighbors of v in remaining graph
            neighbors = adj[v]
            
            # Find a large clique among neighbors
            # Note: The clique found will be among neighbors, so adding v makes it a clique in the whole graph
            # We need to ensure the edges between neighbors exist in remaining_edges
            # find_large_clique uses edge_set which is remaining_edges, so it's correct.
            clique_neighbors = find_large_clique(neighbors, remaining_edges)
            
            # The full clique is v + clique_neighbors
            full_clique = clique_neighbors | {v}
            
            # Add to partition
            partition.append(sorted(full_clique))
            
            # Remove edges of this clique from remaining_edges
            verts = list(full_clique)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v2 = verts[i], verts[j]
                    if u > v2:
                        u, v2 = v2, u
                    if (u, v2) in remaining_edges:
                        remaining_edges.remove((u, v2))
                        
        return partition

    def solve_instance(n, p, seed):
        edges_set = generate_edges(n, p, seed)
        return greedy_clique_partition(n, edges_set)

    result1 = solve_instance(32, 0.6, 601)
    if time.time() - start_time > DEADLINE:
        # Fallback for second instance
        edges2 = generate_edges(64, 0.55, 602)
        result2 = [[u, v] for (u, v) in edges2]
        return [result1, result2]
    
    result2 = solve_instance(64, 0.55, 602)
    return [result1, result2]
