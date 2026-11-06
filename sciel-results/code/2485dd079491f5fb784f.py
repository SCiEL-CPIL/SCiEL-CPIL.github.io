import numpy as np
import random
import sys

def construct():
    def get_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = get_edges(n, p, seed)
        if not edges:
            return []
        
        # Adjacency matrix for fast lookup
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
            
        # Set of remaining edges to cover
        # Using a set of tuples for O(1) removal
        remaining_edges = set(edges)
        
        cliques = []
        
        # Helper to get neighbors with uncovered edges
        def get_uncovered_neighbors(v):
            return [u for u in range(n) if adj[v, u] and ((min(v,u), max(v,u)) in remaining_edges)]

        # Greedy clique finding in an induced subgraph defined by a set of nodes
        # Returns a list of nodes forming a clique
        def find_greedy_clique(nodes):
            if not nodes:
                return []
            # Sort nodes by degree in the induced subgraph (descending)
            # Degree in induced subgraph: count of neighbors within 'nodes'
            node_list = list(nodes)
            
            # Calculate degrees within the subgraph
            deg = {}
            for u in node_list:
                deg[u] = 0
            for i in range(len(node_list)):
                for j in range(i+1, len(node_list)):
                    u, v = node_list[i], node_list[j]
                    if adj[u, v]:
                        deg[u] += 1
                        deg[v] += 1
            
            node_list.sort(key=lambda x: deg[x], reverse=True)
            
            clique = []
            candidates = set(node_list)
            
            while candidates:
                # Pick the candidate with highest degree among remaining candidates
                # We re-sort or pick max
                best_node = max(candidates, key=lambda x: deg[x])
                clique.append(best_node)
                candidates.remove(best_node)
                
                # Filter candidates to those connected to best_node
                # And update degrees? For simplicity, just filter.
                # In a clique, all must be connected to all previous choices.
                new_candidates = set()
                for c in candidates:
                    if adj[best_node, c]:
                        new_candidates.add(c)
                candidates = new_candidates
                
            return clique

        while remaining_edges:
            best_clique = []
            max_covered = 0
            
            # Get active vertices
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
                
            # For each active vertex, try to build a clique centered around it
            # We only need to check a subset if n is large, but n=64 is small.
            # To save time, we can just iterate all active vertices.
            for v in active_vertices:
                unc_neigh = get_uncovered_neighbors(v)
                if not unc_neigh:
                    continue
                
                # Find a greedy clique in the subgraph induced by unc_neigh
                # Note: The clique will be clique_in_neigh + [v]
                # We want to maximize the number of uncovered edges in (clique_in_neigh + [v])
                
                clique_in_neigh = find_greedy_clique(unc_neigh)
                candidate_clique = clique_in_neigh + [v]
                
                # Count uncovered edges in candidate_clique
                covered = 0
                for i in range(len(candidate_clique)):
                    for j in range(i+1, len(candidate_clique)):
                        u, w = candidate_clique[i], candidate_clique[j]
                        e = (min(u,w), max(u,w))
                        if e in remaining_edges:
                            covered += 1
                
                if covered > max_covered:
                    max_covered = covered
                    best_clique = candidate_clique
            
            if not best_clique:
                # Fallback: just take any edge
                e = next(iter(remaining_edges))
                best_clique = list(e)
                max_covered = 1
                
            # Add best clique to result
            cliques.append(sorted(best_clique))
            
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i+1, len(best_clique)):
                    u, w = best_clique[i], best_clique[j]
                    e = (min(u,w), max(u,w))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        
        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    return [c1, c2]

# EVOLVE-BLOCK-END
