"""Clique partition of split graphs using structural properties."""
import time
from random import Random
import networkx as nx

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
        internal_edges_set = set()
        cross_edges = {} # i -> set of c in C such that (c,i) is an edge
        all_cross_edges_set = set()
        
        for u, v in edges:
            if u < k and v < k:
                # Internal edge
                internal_edges_set.add((u, v) if u < v else (v, u))
            elif (u < k and v >= k) or (u >= k and v < k):
                # Cross edge
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                if i not in cross_edges:
                    cross_edges[i] = set()
                cross_edges[i].add(c)
                all_cross_edges_set.add((c, i)) # Store as (c, i) for consistency
        
        cliques = []
        
        # Available internal edges
        available_internal_edges = set(internal_edges_set)
        
        # Process each i in I
        # Sort by size of N(i) descending to greedily cover large sets first
        i_list = sorted(cross_edges.keys(), key=lambda x: len(cross_edges[x]), reverse=True)
        
        for i in i_list:
            if time.time() > deadline - 10:
                break
                
            N_i = cross_edges[i]
            
            # Build graph of available internal edges induced by N_i
            # Vertices are in N_i
            # Edge (u, v) exists in this subgraph if (u, v) is in available_internal_edges
            
            # To find the largest clique in this subgraph, we can use networkx or a simple backtracking
            # Given k <= 30, networkx is safe.
            
            if not N_i:
                continue
                
            # Create subgraph
            # Vertices are N_i
            # Edges are pairs in N_i that are in available_internal_edges
            subgraph_edges = []
            N_i_list = list(N_i)
            for idx, u in enumerate(N_i_list):
                for v in N_i_list[idx+1:]:
                    edge = (u, v) if u < v else (v, u)
                    if edge in available_internal_edges:
                        subgraph_edges.append((u, v))
                        
            if subgraph_edges:
                G_sub = nx.Graph()
                G_sub.add_nodes_from(N_i_list)
                G_sub.add_edges_from(subgraph_edges)
                
                # Find maximum clique
                # For small graphs, this is fast
                max_clique = max(nx.find_cliques(G_sub), key=len)
            else:
                # No internal edges available among neighbors
                # We can still form cliques of size 2: {i, c}
                # This covers cross edge (i, c) but no internal edges
                # We should do this for all c in N_i to cover cross edges
                # But wait, if we do {i, c}, we don't consume internal edges.
                # So we can cover all cross edges for i with 2-cliques if no internal edges available.
                # However, it's better to check if we can form larger cliques later? No, internal edges are only consumed.
                # If no internal edges available, we MUST use 2-cliques for cross edges to cover them.
                # But maybe we should save some 2-cliques? No, cross edges must be covered.
                # So if max clique size is 1, we just add 2-cliques for all c in N_i.
                max_clique = [] # Empty means we handle via 2-cliques below

            if max_clique:
                # We found a clique S in N_i using available internal edges
                S = set(max_clique)
                
                # Create clique {i} U S
                clique = list(S) + [i]
                cliques.append(clique)
                
                # Remove internal edges in S from available
                S_list = list(S)
                for idx, u in enumerate(S_list):
                    for v in S_list[idx+1:]:
                        edge = (u, v) if u < v else (v, u)
                        available_internal_edges.discard(edge)
                        
                # Remove cross edges (i, s) for s in S from consideration
                # We don't explicitly track uncovered cross edges, but we implicitly "cover" them.
                # We must ensure we don't cover them again.
                # Since we process each i only once, and we assign a specific S to i,
                # the cross edges (i, s) for s in S are covered.
                # What about cross edges (i, c) for c in N_i \ S?
                # They are NOT covered by this clique.
                # We need to cover them in subsequent steps or with 2-cliques.
                
                # Let's handle the remaining cross edges for i immediately with 2-cliques
                # because we can't form a larger clique for i (we already took the max available internal clique).
                # Actually, we could potentially form another clique for i using different internal edges?
                # No, i can only be in one clique? No, i can be in multiple cliques?
                # "Every edge must lie in exactly one listed clique".
                # If i is in clique K1 and K2, edges (i, u) in K1 and (i, v) in K2 are distinct.
                # So yes, i can be in multiple cliques.
                # However, we just took the MAXIMUM clique of available internal edges.
                # Are there other internal edges available in N_i \ S?
                # Yes. We can form another clique {i} U S2 where S2 is a clique in the remaining available graph induced by N_i \ S.
                # This would cover more cross edges for i.
                
                # Let's loop to extract all possible cliques for i
                remaining_N_i = N_i - S
                while remaining_N_i:
                    if time.time() > deadline - 5:
                        break
                    # Rebuild subgraph for remaining_N_i
                    rem_edges = []
                    rem_list = list(remaining_N_i)
                    for idx, u in enumerate(rem_list):
                        for v in rem_list[idx+1:]:
                            edge = (u, v) if u < v else (v, u)
                            if edge in available_internal_edges:
                                rem_edges.append((u, v))
                    
                    if rem_edges:
                        G_rem = nx.Graph()
                        G_rem.add_nodes_from(rem_list)
                        G_rem.add_edges_from(rem_edges)
                        next_clique = max(nx.find_cliques(G_rem), key=len)
                    else:
                        next_clique = []
                        
                    if next_clique:
                        S2 = set(next_clique)
                        clique = list(S2) + [i]
                        cliques.append(clique)
                        
                        # Remove internal edges
                        S2_list = list(S2)
                        for idx, u in enumerate(S2_list):
                            for v in S2_list[idx+1:]:
                                edge = (u, v) if u < v else (v, u)
                                available_internal_edges.discard(edge)
                                
                        remaining_N_i.difference_update(S2)
                    else:
                        # No more internal edges available in remaining_N_i
                        # Cover remaining cross edges with 2-cliques
                        for c in remaining_N_i:
                            cliques.append([i, c])
                        break # All cross edges for i are now covered

        # Cover remaining internal edges
        # These are edges in C not covered by any cross-clique
        # We can cover them with 2-cliques
        # To be slightly more efficient, we could try to find cliques in the remaining internal graph
        # But 2-cliques are safe and simple. Given k is small, the number of remaining edges is small.
        for u, v in available_internal_edges:
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
