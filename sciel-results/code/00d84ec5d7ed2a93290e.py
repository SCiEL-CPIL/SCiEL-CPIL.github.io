def contains_q3_exact(adj_subset):
    # adj_subset: list of 8 ints
    # Check if there is a 3-regular spanning subgraph
    # and if it is bipartite.
    
    # 1. Check degrees
    degs = [bin(adj_subset[i]).count('1') for i in range(8)]
    if any(d < 3 for d in degs):
        return False
    
    # 2. If all degrees are 3, check bipartite
    if all(d == 3 for d in degs):
        # Bipartite check
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in range(8):
                if (adj_subset[u] >> v) & 1:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
        return True
        
    # 3. Some degrees > 3
    # We need to check if a 3-regular spanning subgraph exists.
    # We can try to remove edges to make it 3-regular.
    # This is equivalent to checking if the complement (within K8) has a 2-regular spanning subgraph (2-factor).
    # Let comp_adj be the complement.
    # deg_comp[i] = 7 - degs[i]
    # We need a subgraph H of comp_adj such that deg_H[i] = 7 - 3 = 4? No.
    # If G' is the 3-regular subgraph of G, then E(G') = 12.
    # E(comp(G')) = 28 - 12 = 16.
    # deg_comp(G')[i] = 7 - 3 = 4.
    # So we need a 4-regular spanning subgraph in the complement?
    # No, G' is a subgraph of G. So E(G') \subseteq E(G).
    # The edges NOT in G' are E(K8) \ E(G').
    # This is getting complicated.
    
    # Simpler: Backtracking on G.
    # We want to select 12 edges such that each vertex has degree 3.
    # We can iterate through vertices and assign their 3 neighbors.
    
    # To be very fast, we can use a necessary condition:
    # If the number of edges is 13, we must remove 1 edge to get 12.
    # If we remove edge (u,v), deg[u] and deg[v] decrease by 1.
    # For the resulting graph to be 3-regular, we need deg[u]-1=3 => deg[u]=4, and deg[v]-1=3 => deg[v]=4.
    # And all other degrees must be 3.
    # So if edges == 13, check if there is an edge (u,v) with deg[u]==4 and deg[v]==4, and all other deg==3.
    
    # If edges == 14, we remove 2 edges.
    # This is getting complex.
    
    # Given the time limit, let's use a probabilistic check for the "hard" cases (edges > 12)?
    # No, validity is key.
    
    # Let's implement the backtracking. It's only 8 vertices.
    # We can use a simple recursive search with pruning.
    
    # State: current degrees, edges used
    # Target: all degrees 3
    
    # We can just check if the graph contains a Q3 by trying all 8! mappings?
    # 40,320 mappings.
    # For each mapping, check if the 12 Q3 edges are present.
    # 12 checks per mapping. 40,320 * 12 = 483,840 operations.
    # For 12,870 subsets, that's 6 billion operations. Too slow.
    
    # BUT, we can prune.
    # Most subsets will fail the degree check.
    # Only subsets with min degree >= 3 and edges >= 12 will proceed.
    # In a dense graph, many will proceed.
    # In a sparse graph (near optimal), few will proceed.
    
    # Let's optimize the mapping check.
    # Precompute the 12 edge masks of Q3 in 8-bit space.
    # q3_edge_masks = [ (1<<0)|(1<<1), ... ]
    
    # For a given subset, we map the 8 vertices to 0..7.
    # We get the induced adjacency matrix (8x8).
    # We check if there is a permutation P such that for all q3_edges (u,v), adj[P[u]][P[v]] is 1.
    
    # We can iterate over permutations of 8 elements.
    # To speed up, we can use a backtracking permutation generator that stops early if a constraint fails.
    
    # Let's implement this.
