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
        internal_edges = set()
        cross_edges = {} # i -> list of c in C such that (c,i) is an edge
        
        for u, v in edges:
            if u < k and v < k:
                # Internal edge
                internal_edges.add((u, v) if u < v else (v, u))
            elif (u < k and v >= k) or (u >= k and v < k):
                # Cross edge
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                if i not in cross_edges:
                    cross_edges[i] = []
                cross_edges[i].append(c)
        
        cliques = []
        
        # Track blocked internal edges
        # blocked_edges is a set of tuples (u, v) with u < v
        blocked_edges = set()
        
        # Helper to check if a set of vertices forms a clique with no blocked edges
        # i.e., for all pairs in S, the edge is NOT in blocked_edges
        def is_valid_clique(S):
            S_list = list(S)
            for idx, u in enumerate(S_list):
                for v in S_list[idx+1:]:
                    edge = (u, v) if u < v else (v, u)
                    if edge in blocked_edges:
                        return False
            return True

        # Helper to find a large independent set in the graph of blocked edges induced by V
        # Greedy: pick vertex with min degree in induced subgraph, remove it and neighbors
        def find_large_independent_set(V):
            if not V:
                return set()
            V_list = list(V)
            # Build adjacency in blocked graph for V
            # deg[v] = number of blocked edges incident to v within V
            deg = {v: 0 for v in V_list}
            adj = {v: set() for v in V_list}
            for i in range(len(V_list)):
                for j in range(i+1, len(V_list)):
                    u, v = V_list[i], V_list[j]
                    edge = (u, v) if u < v else (v, u)
                    if edge in blocked_edges:
                        deg[u] += 1
                        deg[v] += 1
                        adj[u].add(v)
                        adj[v].add(u)
            
            remaining = set(V_list)
            independent_set = set()
            
            while remaining:
                # Find vertex with min degree
                min_deg_vertex = min(remaining, key=lambda x: deg[x])
                independent_set.add(min_deg_vertex)
                # Remove min_deg_vertex and its neighbors from remaining
                neighbors = adj[min_deg_vertex] & remaining
                remaining.discard(min_deg_vertex)
                remaining.difference_update(neighbors)
                
            return independent_set

        # Process each i in I
        # Sort by size of N(i) descending to greedily cover large sets first?
        # Actually, sorting might not matter much, but let's try.
        i_list = sorted(cross_edges.keys(), key=lambda x: len(cross_edges[x]), reverse=True)
        
        for i in i_list:
            N_i = set(cross_edges[i])
            while N_i:
                # Find large independent set in blocked graph induced by N_i
                S = find_large_independent_set(N_i)
                if not S:
                    # Should not happen if N_i is not empty, unless all vertices are isolated?
                    # Even if isolated, deg is 0, so it will be picked.
                    # If N_i has 1 vertex, S is that vertex.
                    break 
                
                # Add clique S U {i}
                clique = list(S) + [i]
                cliques.append(clique)
                
                # Block all edges within S
                S_list = list(S)
                for idx, u in enumerate(S_list):
                    for v in S_list[idx+1:]:
                        edge = (u, v) if u < v else (v, u)
                        blocked_edges.add(edge)
                        
                # Remove S from N_i
                N_i.difference_update(S)
        
        # Cover remaining internal edges
        # Remaining edges are in internal_edges but not in blocked_edges
        remaining_edges = internal_edges - blocked_edges
        
        # Group remaining edges into cliques
        # Simple greedy: pick an edge, expand to max clique, remove edges
        rem_edges_list = list(remaining_edges)
        # To be efficient, we can just iterate
        # Since k is small, we can just add 2-cliques for each remaining edge if the set is small
        # Or try to merge.
        # Given the potential size, let's just use 2-cliques for safety and simplicity,
        # unless the number is very small.
        # Actually, we can do a simple pass to find cliques.
        # But 2-cliques are valid.
        for u, v in remaining_edges:
            cliques.append([u, v])
            
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
