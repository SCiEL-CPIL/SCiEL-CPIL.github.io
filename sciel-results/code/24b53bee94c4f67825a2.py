import time
import random
import sys
import heapq

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        edge_map = {} # (u,v) -> index
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                    edge_map[(u, v)] = idx
                    edge_map[(v, u)] = idx # Bidirectional for faster lookup
                    idx += 1
        return adj, edges, edge_map

    def solve_instance(adj, edges, edge_map):
        n = len(adj)
        if not edges:
            return []
        
        # We will maintain a set of remaining edge indices
        remaining_edges = set(range(len(edges)))
        partition = []
        
        # To speed up finding the largest clique, we can try greedy expansion from multiple seeds
        # and pick the best one found in a time-bounded search, or just a very good single pass.
        # Given n is small (32, 64), we can afford a decent heuristic.
        
        # Strategy: 
        # 1. Estimate degree of each vertex in the remaining graph.
        # 2. Pick the vertex with highest degree.
        # 3. Greedily expand a clique starting from it.
        # 4. Repeat.
        
        # To make it faster, we can pre-compute adjacency as bitmasks or just use sets.
        # Sets are fine for n=64.
        
        # Optimization: Instead of recalculating degrees from scratch every time (O(E)),
        # we can update them. But since we remove a whole clique's edges, 
        # it might be simpler to just re-scan or maintain a dynamic degree count.
        # For n=64, E is ~2000. Re-scanning is fast enough if the number of iterations is small.
        # How many iterations? Each clique removes at least 1 edge. 
        # In a dense graph, large cliques remove many edges. 
        # If we find cliques of size ~5-10, we remove 10-45 edges.
        # 2000 / 20 ~ 100 iterations.
        # 100 * O(E) = 100 * 2000 = 200,000 ops. Very fast.
        
        # Let's implement a robust greedy clique finder.
        
        def find_largest_clique_greedy(adj, remaining_edges, edge_map, n):
            # Calculate current degree of each vertex in the remaining graph
            degrees = [0] * n
            for e_idx in remaining_edges:
                u, v = edges[e_idx]
                degrees[u] += 1
                degrees[v] += 1
            
            if not remaining_edges:
                return []
            
            # Find vertex with max degree
            # Tie-breaker: random or fixed. Fixed is deterministic.
            max_deg = max(degrees)
            if max_deg == 0:
                return []
            
            # Collect all vertices with max degree
            candidates = [i for i in range(n) if degrees[i] == max_deg]
            
            # To avoid getting stuck in a local optimum, we could try a few candidates,
            # but for speed, let's just pick the first one or the one with highest index?
            # Actually, let's try a few top candidates if time permits.
            # For now, pick the first one to keep it simple and fast.
            # Better: Pick a random one from the top few? 
            # Let's stick to deterministic for reproducibility unless we add a seed.
            # The task uses fixed seeds for graphs, so deterministic algorithm is fine.
            
            start_v = candidates[0]
            
            # Greedy expansion
            clique = [start_v]
            # Potential vertices: neighbors of start_v that are connected in remaining graph
            # A vertex w is a valid addition if it is connected to ALL vertices in clique
            # and the edge (w, v) exists in remaining_edges for all v in clique.
            
            # To check if w can join, we need:
            # 1. (w, v) in remaining_edges for all v in clique.
            
            # We can maintain a set of "candidates" that are still connected to all current clique members.
            # Initially, candidates = neighbors of start_v in remaining graph.
            
            # Get neighbors of start_v in remaining graph
            # We need to iterate over all edges incident to start_v and check if they are in remaining_edges.
            # Or iterate over adj[start_v] and check edge_map.
            
            potential = set()
            for v in adj[start_v]:
                if (start_v, v) in edge_map:
                    e_idx = edge_map[(start_v, v)]
                    if e_idx in remaining_edges:
                        potential.add(v)
            
            # Now iteratively add a vertex from potential that is connected to all others in potential?
            # No, the standard greedy algorithm for maximal clique is:
            # 1. Start with R={v}
            # 2. P = N(v)
            # 3. While P is not empty:
            #    a. Pick u in P with max degree in P (or just max degree in graph?)
            #       Actually, to ensure it's a clique, we must pick u that is connected to all other vertices in R?
            #       No, if we build R incrementally, and only add u if u is connected to ALL vertices in R,
            #       then R remains a clique.
            #       The set of valid candidates is the intersection of neighbors of all vertices in R.
            
            # Let's refine the greedy expansion:
            # current_clique = [start_v]
            # valid_candidates = set(adj[start_v] & remaining_neighbors)
            
            # At each step, pick a vertex u from valid_candidates.
            # Which one? The one with the highest degree within valid_candidates?
            # This is a common heuristic.
            
            while potential:
                # Calculate degree of each vertex in potential, restricted to potential
                # i.e., number of neighbors in potential
                deg_in_potential = {v: 0 for v in potential}
                for v in potential:
                    for w in adj[v]:
                        if w in potential:
                            deg_in_potential[v] += 1
                
                # Pick vertex with max degree in potential
                # Tie-break: smallest index for determinism
                best_v = min(potential, key=lambda x: (-deg_in_potential[x], x))
                
                clique.append(best_v)
                
                # Update potential: remove best_v, and any vertex not connected to best_v
                # New potential = (potential - {best_v}) & N(best_v)
                new_potential = set()
                for v in potential:
                    if v != best_v and best_v in adj[v]:
                        # Also need to ensure edge (v, best_v) is in remaining_edges
                        # Since we started with edges in remaining_edges and only remove them,
                        # and we only added vertices connected via remaining edges,
                        # the adjacency check `best_v in adj[v]` is sufficient for existence in original graph.
                        # But we must ensure the edge is still in `remaining_edges`.
                        # Wait, `potential` was built from `remaining_edges`.
                        # When we pick `best_v`, we remove it.
                        # The remaining vertices in `potential` are already connected to all previous clique members.
                        # Are they connected to `best_v`?
                        # We filter by `best_v in adj[v]`.
                        # Is the edge (v, best_v) in `remaining_edges`?
                        # It was in `remaining_edges` when we added `v` to `potential`?
                        # Not necessarily. `v` was in `potential` because it was connected to `start_v`.
                        # It might not be connected to `best_v` in the remaining graph?
                        # No, if `v` is in `potential`, it is connected to all vertices in `clique` (except the one we are about to add, which is `best_v`).
                        # Wait, the standard greedy maximal clique algorithm maintains `P` as the set of vertices connected to ALL vertices in `R`.
                        # So if `v` is in `P`, it is connected to `best_v` (which is in `R` before we add it? No, `best_v` is chosen from `P`).
                        # So `v` is connected to `best_v` in the original graph.
                        # Is the edge in `remaining_edges`?
                        # Yes, because `v` is in `P`, which means all edges from `v` to `R` are in `remaining_edges`.
                        # `best_v` is in `R` (conceptually, we are adding it).
                        # So the edge is in `remaining_edges`.
                        
                        new_potential.add(v)
                
                potential = new_potential
            
            return clique
        
        # Main loop
        while remaining_edges and time.time() < deadline - 5.0:
            clique = find_largest_clique_greedy(adj, remaining_edges, edge_map, n)
            
            if not clique:
                # Should not happen if remaining_edges is not empty, unless logic error.
                # Fallback: take one edge
                e_idx = next(iter(remaining_edges))
                partition.append(list(edges[e_idx]))
                remaining_edges.remove(e_idx)
                continue
            
            # Remove all edges in this clique from remaining_edges
            clique_set = set(clique)
            edges_to_remove = set()
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v: u, v = v, u
                    if (u, v) in edge_map:
                        edges_to_remove.add(edge_map[(u, v)])
            
            # Verify these edges are actually in remaining_edges (safety)
            edges_to_remove = edges_to_remove.intersection(remaining_edges)
            
            if not edges_to_remove:
                # Something went wrong, take one edge
                e_idx = next(iter(remaining_edges))
                partition.append(list(edges[e_idx]))
                remaining_edges.remove(e_idx)
                continue
                
            partition.append(clique)
            remaining_edges -= edges_to_remove
            
        # Fallback for any remaining edges (should be none or very few if logic is correct)
        if remaining_edges:
            for e_idx in remaining_edges:
                partition.append(list(edges[e_idx]))
        
        return partition

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 15.0:
            # Not enough time
            adj, edges, edge_map = random_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        adj, edges, edge_map = random_graph(n, p, seed)
        cliques = solve_instance(adj, edges, edge_map)
        results.append(cliques)
        
    return results
