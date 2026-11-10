import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        remaining_edges = set(edges)
        cliques = []
        
        # Pre-compute adjacency sets for initial graph to speed up neighbor lookups?
        # But adjacency changes as edges are removed.
        # It's better to maintain a set of remaining edges and derive neighbors on the fly or update an adjacency structure.
        # For n=64, deriving neighbors from the edge set is O(E) per query, which is fine.
        
        def get_neighbors(v, edge_set):
            neighbors = set()
            for u, w in edge_set:
                if u == v:
                    neighbors.add(w)
                elif w == v:
                    neighbors.add(u)
            return neighbors

        def is_clique(vertices, edge_set):
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    u, v = vertices[i], vertices[j]
                    if (u, v) not in edge_set and (v, u) not in edge_set:
                        return False
            return True

        def find_maximal_clique(start_v, edge_set):
            # Greedy expansion
            C = [start_v]
            # Candidates are neighbors of start_v
            candidates = get_neighbors(start_v, edge_set)
            
            while candidates:
                # Filter candidates to those connected to all in C
                valid_candidates = []
                for u in candidates:
                    is_connected_to_all = True
                    for v in C:
                        if (u, v) not in edge_set and (v, u) not in edge_set:
                            is_connected_to_all = False
                            break
                    if is_connected_to_all:
                        valid_candidates.append(u)
                
                if not valid_candidates:
                    break
                
                # Choose the best candidate
                # Heuristic: Maximize the number of common neighbors with C in the remaining graph
                # Or simply pick the one with highest degree in the remaining graph
                # Let's try: Score = number of neighbors in C (always len(C)) + number of other neighbors
                # Actually, just picking the one with the highest degree in the remaining graph is a good proxy for "dense area"
                
                best_u = None
                best_score = -1
                for u in valid_candidates:
                    deg = len(get_neighbors(u, edge_set))
                    # Tie-breaker: random or index
                    if deg > best_score:
                        best_score = deg
                        best_u = u
                        
                C.append(best_u)
                # Update candidates: remove best_u, and any that are not connected to best_u
                # Actually, we can just re-filter in the next loop iteration.
                # To be efficient, we can remove best_u and its non-neighbors from candidates immediately.
                candidates.discard(best_u)
                # Remove any candidate not connected to best_u
                to_remove = set()
                for u in candidates:
                    if (u, best_u) not in edge_set and (best_u, u) not in edge_set:
                        to_remove.add(u)
                candidates -= to_remove

            return C

        while remaining_edges:
            if time.time() > deadline:
                # Return what we have, plus any remaining edges as singletons to ensure validity
                # But the contract says "every edge must lie in exactly one listed clique".
                # If we time out, we must complete the partition.
                # Given the speed, this is unlikely, but we handle it.
                pass
            
            # Find a vertex with the highest degree in the remaining graph
            # We can just pick the first edge's endpoint, or compute degrees.
            # Computing degrees is O(E).
            degrees = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                degrees[u] += 1
                degrees[v] += 1
            
            # Pick vertex with max degree
            start_v = max(degrees, key=degrees.get)
            
            # Find a maximal clique starting from start_v
            C = find_maximal_clique(start_v, remaining_edges)
            
            if len(C) < 2:
                # Should not happen if there are edges, but just in case
                # Pick an edge
                u, v = next(iter(remaining_edges))
                C = [u, v]
            
            cliques.append(C)
            
            # Remove all edges in C from remaining_edges
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    if (u, v) in remaining_edges:
                        remaining_edges.remove((u, v))
                    elif (v, u) in remaining_edges:
                        remaining_edges.remove((v, u))
                    
        return cliques

    result_1 = solve_instance(32, 0.6, 601)
    result_2 = solve_instance(64, 0.55, 602)
    
    return [result_1, result_2]
