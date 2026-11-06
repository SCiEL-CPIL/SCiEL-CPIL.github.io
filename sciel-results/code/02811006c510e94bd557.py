"""
Rewritten program for Clique Partitions of Dense Random Graphs.

Strategy:
1. Generate the graphs as specified.
2. Use a greedy set-cover heuristic to partition edges into cliques.
   - Maintain a set of uncovered edges.
   - Iteratively find a large clique in the graph formed by the uncovered edges.
     The heuristic prioritizes cliques that cover the most edges (equivalent to largest size).
     We use a greedy expansion algorithm starting from each vertex to find a large clique.
   - Assign all edges in that clique to a single partition element.
   - Remove those edges from the uncovered set.
   - Repeat until all edges are covered.
"""

def construct():
    from random import Random
    import collections
    import itertools

    def generate_edges(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_large_clique(adj, active_vertices):
        """
        Finds a large clique in the subgraph induced by active_vertices using adj.
        Uses a greedy heuristic: try starting from each vertex, expand greedily.
        """
        if not active_vertices:
            return []
        
        best_clique = []
        best_size = 0
        
        # We want to find the largest clique. 
        # Greedy approach: for each start vertex, expand to neighbors with highest degree in candidate set.
        for start in active_vertices:
            clique = [start]
            candidates = set(adj[start])
            candidates.intersection_update(active_vertices)
            
            while candidates:
                # Pick vertex with max degree in candidates
                max_deg = -1
                next_v = None
                for v in candidates:
                    # Count neighbors of v that are also in candidates
                    deg = 0
                    for u in adj[v]:
                        if u in candidates:
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        next_v = v
                
                if next_v is None:
                    break
                    
                clique.append(next_v)
                # Update candidates to be intersection of current candidates and neighbors of next_v
                new_candidates = set()
                for u in adj[next_v]:
                    if u in candidates:
                        new_candidates.add(u)
                candidates = new_candidates
                
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
                
        return best_clique

    def solve_instance(n, edges):
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            # Build adjacency set for the remaining edges
            adj = collections.defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
                
            best_clique = find_large_clique(adj, active_vertices)
            
            if best_clique and len(best_clique) >= 2:
                cliques.append(best_clique)
                # Remove all edges within best_clique from remaining_edges
                to_remove = set()
                for u, v in itertools.combinations(best_clique, 2):
                    if u > v:
                        u, v = v, u
                    to_remove.add((u, v))
                remaining_edges -= to_remove
            else:
                # Fallback: pick an arbitrary edge (should not happen if remaining_edges is not empty)
                if remaining_edges:
                    u, v = next(iter(remaining_edges))
                    cliques.append([u, v])
                    remaining_edges.discard((u, v))
                else:
                    break
                    
        return cliques

    # Instance 1
    n1, p1, seed1 = 32, 0.6, 601
    edges1 = generate_edges(n1, p1, seed1)
    cliques1 = solve_instance(n1, edges1)
    
    # Instance 2
    n2, p2, seed2 = 64, 0.55, 602
    edges2 = generate_edges(n2, p2, seed2)
    cliques2 = solve_instance(n2, edges2)
    
    return [cliques1, cliques2]

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Instance 1 cliques: {len(res[0])}")
    print(f"Instance 2 cliques: {len(res[1])}")
