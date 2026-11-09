"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n = 16
    
    # Precompute Q3 structure
    # Q3 vertices are 0..7. Edges connect i,j if popcount(i^j)==1
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.add((i, j))
    
    # Map Q3 vertices to bit positions (0..7)
    # We will use bitmasks to represent the adjacency of the current graph
    # adj_mask[v] is an integer where bit u is 1 if (u,v) is an edge
    
    # Initialize with complete graph K16
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                adj_mask[i] |= (1 << j)
    
    current_edges = n * (n - 1) // 2
    
    # Precompute all embeddings of Q3 in the complete graph
    # An embedding is a tuple of 8 distinct vertices (v0, ..., v7)
    # such that the induced subgraph is isomorphic to Q3.
    # In K16, every set of 8 vertices induces K8, which contains Q3.
    # We need to list all distinct mappings of Q3 vertices to graph vertices.
    # Total embeddings = P(16, 8) = 16! / 8! = 518,918,400. This is too many to precompute.
    # Instead, we will generate them on the fly or use a smarter counting method.
    
    # Alternative: Count Q3 subgraphs by iterating over all 8-sets?
    # C(16,8) = 12,870.
    # For each 8-set, count the number of isomorphisms to Q3.
    # This is feasible.
    
    # Let's precompute the number of Q3 embeddings for each 8-set in the current graph.
    # Since the graph changes, we need to update counts efficiently.
    
    # Efficient Strategy:
    # 1. Maintain a list of all 8-sets.
    # 2. For each 8-set, maintain the number of Q3 embeddings in the induced subgraph.
    # 3. When removing an edge (u,v), update the counts for all 8-sets containing u and v.
    #    How does removing (u,v) affect the count?
    #    It decreases the count by the number of Q3 embeddings in the 8-set that USE the edge (u,v).
    
    # Let's precompute for each 8-set S:
    # - The list of all Q3 embeddings (mappings phi: {0..7} -> S) in the complete graph K_S.
    # - For each embedding, the set of edges it uses.
    # This is too much memory/time.
    
    # Simpler Approach:
    # Use the fact that we are doing local search.
    # We want to find the edge (u,v) whose removal eliminates the most Q3 subgraphs.
    # Let C(e) be the number of Q3 subgraphs containing edge e.
    # We want to minimize C(e) for the edge we keep, or maximize for the one we remove.
    # Actually, we want to remove the edge that is in the MOST Q3s.
    
    # How to compute C(u,v) quickly?
    # A Q3 subgraph containing edge (u,v) corresponds to an embedding of Q3 where u and v are mapped to adjacent vertices in Q3 (say 0 and 1).
    # The remaining 6 vertices of the Q3 must be mapped to neighbors of u and v appropriately.
    
    # Let's precompute the "local structure" of Q3 relative to an edge (0,1).
    # Q3 vertices: 0,1,2,3,4,5,6,7
    # Edge (0,1) exists.
    # Neighbors of 0: 1, 2, 4
    # Neighbors of 1: 0, 3, 5
    # So if u maps to 0 and v maps to 1:
    # - We need 2 vertices x,y from N(u) \ {v} to map to 2,4.
    # - We need 2 vertices z,w from N(v) \ {u} to map to 3,5.
    # - x (2) and y (4) must have a common neighbor p (6) in the graph.
    # - z (3) and w (5) must have a common neighbor q (7) in the graph.
    # - p and q must be connected by an edge.
    # - All 8 vertices must be distinct.
    
    # This check is similar to `check_q3_free_fast` in the seed program but counts all such structures.
    # We can compute the number of such structures for a given edge (u,v) in the current graph.
    
    def count_q3_containing_edge(u, v):
        """Count the number of Q3 subgraphs in the current graph that contain edge (u,v)."""
        count = 0
        # u maps to 0, v maps to 1 (and symmetrically v to 0, u to 1, but Q3 is symmetric, so we just count embeddings where u,v are the 0-1 pair)
        # Actually, since we count subgraphs (sets of vertices), we need to be careful not to double count.
        # A Q3 subgraph has 12 edges. If we sum counts over all edges, each subgraph is counted 12 times.
        # So if we find the edge with max count, removing it removes count/12 subgraphs on average? No, it removes exactly the subgraphs containing that edge.
        
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        
        # Iterate over pairs (x,y) from N(u) and (z,w) from N(v)
        # x,y must be distinct, z,w must be distinct
        # x,y,z,w must be distinct from each other and u,v
        
        # To speed up, we can iterate over x in N(u), y in N(u), x<y
        # and z in N(v), w in N(v), z<w
        
        # Optimization: Precompute common neighbors for pairs?
        # Given the small size, we can just iterate.
        
        # Get lists of neighbors
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        for i in range(len(nu_list)):
            x = nu_list[i]
            for j in range(i+1, len(nu_list)):
                y = nu_list[j]
                # Check if x and y have common neighbors (candidates for p=6)
                # Common neighbors of x and y in the current graph
                common_xy = adj_mask[x] & adj_mask[y]
                # p must be distinct from u, v, x, y
                # p must also be distinct from z, w (checked later)
                
                # Iterate over z, w in N(v)
                for k in range(len(nv_list)):
                    z = nv_list[k]
                    if z == x or z == y:
                        continue
                    for l in range(k+1, len(nv_list)):
                        w = nv_list[l]
                        if w == x or w == y or w == z:
                            continue
                        
                        # Candidates for p: common_xy minus {u,v,x,y,z,w}
                        forbidden_p = (1 << u) | (1 << v) | (1 << x) | (1 << y) | (1 << z) | (1 << w)
                        p_candidates = common_xy & ~forbidden_p
                        
                        if p_candidates == 0:
                            continue
                        
                        # Candidates for q: common neighbors of z and w
                        common_zw = adj_mask[z] & adj_mask[w]
                        
                        # For each p, check if there is a q in common_zw such that (p,q) is an edge
                        # q must be distinct from u,v,x,y,z,w,p
                        # Iterate over p candidates
                        p_list = [i for i in range(n) if (p_candidates >> i) & 1]
                        for p in p_list:
                            forbidden_q = forbidden_p | (1 << p)
                            q_candidates = common_zw & ~forbidden_q
                            
                            # Check if any q in q_candidates is connected to p
                            # adj_mask[p] & q_candidates != 0
                            if adj_mask[p] & q_candidates:
                                # We found at least one Q3.
                                # How many?
                                # The number of valid q's.
                                count += (adj_mask[p] & q_candidates).bit_count()
        
        return count

    # Main Loop: Remove edges to eliminate Q3s
    # We will repeatedly find the edge with the highest count of Q3s containing it and remove it.
    # Stop when no Q3s remain (or time runs out).
    
    # To speed up finding the max edge, we can compute counts for all edges initially,
    # then update only the affected edges after each removal.
    # However, computing all counts initially is O(E * Cost(count)).
    # E = 120. Cost(count) is roughly O(D^4). D ~ 15. 15^4 = 50k. 120 * 50k = 6M ops. Feasible.
    
    # Let's maintain a dictionary or list of counts for all edges.
    # edge_counts[(u,v)] = count
    
    # Initialize counts
    edge_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_counts[(i,j)] = count_q3_containing_edge(i, j)
    
    # Find the edge with max count
    # If max count is 0, we are done.
    
    while True:
        if time.time() > deadline - 5:
            break
        
        # Find edge with max count
        max_edge = None
        max_count = 0
        for e, c in edge_counts.items():
            if c > max_count:
                max_count = c
                max_edge = e
        
        if max_count == 0:
            break # No Q3s left
        
        u, v = max_edge
        
        # Remove edge (u,v)
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        current_edges -= 1
        
        # Update counts for all edges that share a vertex with (u,v) or are part of the same Q3s?
        # Actually, removing (u,v) only affects the count of edges that are in the SAME Q3s as (u,v).
        # An edge (a,b) is in a Q3 with (u,v) if (a,b) and (u,v) are in the same Q3 subgraph.
        # This is complex to track incrementally.
        # Given the small number of edges (120), it might be faster to just recompute all counts?
        # Recomputing all 120 counts takes ~6M ops.
        # If we do this for, say, 50 steps, that's 300M ops. Might be tight in 145s.
        # Let's try incremental update first. If too slow, fall back to recompute.
        
        # Incremental Update Strategy:
        # The count of edge (a,b) decreases by the number of Q3s containing BOTH (a,b) and (u,v).
        # Let's call this delta(a,b).
        # We need to compute delta(a,b) for all edges (a,b).
        
        # How many Q3s contain both (u,v) and (a,b)?
        # This is equivalent to counting Q3 embeddings where u,v map to 0,1 AND a,b map to some other edge in Q3.
        
        # To keep it simple and robust, let's just recompute all counts every step.
        # Optimization: Only recompute if we haven't done it recently? No, state changes.
        
        # Let's estimate time.
        # count_q3_containing_edge is the bottleneck.
        # Let's optimize count_q3_containing_edge using bit operations more aggressively.
        
        # Actually, let's just recompute.
        for i in range(n):
            for j in range(i+1, n):
                edge_counts[(i,j)] = count_q3_containing_edge(i, j)
                
        # Safety check: if we are running out of time, break
        if time.time() > deadline - 10:
            break

    # Construct the final edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_mask[i] >> j) & 1:
                final_edges.append([i, j])
                
    return final_edges
# EVOLVE-BLOCK-END
