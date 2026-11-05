"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # We want to find a C5-free graph with no independent set of size 9.
    # The seed uses 7 disjoint K4s (28 vertices, alpha=7).
    # We want to push n higher. Best known is 32.
    
    # Strategy:
    # 1. Start with a known valid structure or random valid structure.
    # 2. Use local search / simulated annealing to add vertices/edges while maintaining constraints.
    # 
    # Constraints:
    # A. No C5 (5-cycle) as a subgraph.
    # B. No independent set of size 9.
    
    # Let's try a constructive approach.
    # A graph is C5-free if it has no induced 5-cycles or non-induced 5-cycles.
    # Actually, "no 5-cycle as a subgraph" means no cycle of length 5.
    # This is a strong condition.
    
    # Let's look at the structure of the seed: 7 K4s.
    # K4 is C5-free. Disjoint union is C5-free.
    # Alpha of disjoint union is sum of alphas. Alpha(K4)=1. So alpha=7.
    # We can add vertices.
    
    # Idea: Start with a random C5-free graph with small alpha, then grow it.
    # Or, start with a specific family.
    
    # Let's try to use a random greedy construction or a local search on the edge set.
    # However, checking C5-freeness is expensive.
    
    # Alternative:
    # Use the fact that C5-free graphs have limited density?
    # Ex(n, C5) is known to be O(n^(1+1/3)) or similar? No, for C_{2k+1}, it's n^(1+1/k).
    # For C5 (k=2), exponent is 1+1/2 = 1.5?
    # Actually, Reiman's bound for C5-free graphs is m <= n/4 (1+sqrt(1+4n)).
    # For n=35, m <= 35/4 * (1+sqrt(141)) ≈ 8.75 * (1+11.87) ≈ 8.75 * 12.87 ≈ 112.
    # So density is not the bottleneck.
    
    # The bottleneck is likely the independence number.
    # We need alpha < 9.
    
    # Let's try a random search over edge sets for a fixed n, starting from n=30 and going up.
    # But 2^(n^2/2) is huge.
    
    # Better: Local search on the graph.
    # Start with a valid graph (e.g., seed).
    # Try to add a vertex and connect it to some existing vertices.
    # Or, start with a random graph and prune edges to remove C5s, then check alpha.
    
    # Let's try a different structural family.
    # What about the complement of a C5-free graph? No, that's not helpful directly.
    
    # Let's try to construct a graph with n=32.
    # Known constructions often use finite geometry or specific block designs.
    # One known C5-free graph with high minimum degree is the Clebsch graph?
    # Clebsch graph: 16 vertices, 5-regular, C5-free?
    # Clebsch graph is the complement of the line graph of K5? No.
    # Clebsch graph is a distance-regular graph. It is C5-free?
    # Let's check: Clebsch graph has girth 4? No, girth is 4?
    # If girth is 4, it has no C3 or C5? No, girth 4 means no C3, but could have C5.
    # Actually, Clebsch graph is C5-free.
    # Alpha(Clebsch) = 5.
    # If we take disjoint union of Clebsch graphs?
    # 2 Clebsch = 32 vertices. Alpha = 5+5=10. Too high.
    # 1 Clebsch + something?
    # 16 + 16 = 32. Alpha 10.
    # We need alpha < 9.
    
    # What if we merge two Clebsch graphs by identifying some vertices?
    # Or add edges between them?
    # Adding edges reduces alpha.
    
    # Let's try: Start with 2 Clebsch graphs (32 vertices).
    # Add edges between them to reduce alpha from 10 to 8.
    # Each added edge between the two components can reduce alpha by at most 1 (if it hits a max independent set).
    # We need to reduce alpha by 2. So 2 edges might suffice?
    # But we must ensure no C5 is created.
    # Adding an edge between two C5-free graphs can create a C5 if there is a path of length 4 between the endpoints?
    # No, a C5 is a cycle of length 5.
    # If we add edge (u,v) with u in G1 and v in G2.
    # A C5 involving this edge would be u-v-...-u.
    # The path ...-u must be in G1 and v-... in G2?
    # No, the path must go from v back to u.
    # Since G1 and G2 were disjoint, any path from v to u must use the new edge?
    # No, the cycle is u-v-w-x-y-u.
    # If w,x,y are in G1 or G2?
    # If w is in G2, then v-w is in G2.
    # If x is in G2, w-x in G2.
    # If y is in G1, x-y must be an edge? But x is in G2, y in G1. No edge between G1 and G2 except the new one?
    # If we add only one edge, there is no other connection.
    # So a cycle using the new edge must go u-v (new), then v...u in the rest of the graph.
    # But the rest of the graph is G1 union G2 (disjoint).
    # So v must be connected to u via a path in G1 union G2.
    # But v is in G2, u is in G1. There is no path between G1 and G2 in the original graph.
    # So adding a single edge between two disjoint components creates NO cycles at all!
    # Wait. If I add an edge between two disjoint graphs, I create a bridge. No cycles are formed.
    # So I can add arbitrary edges between the two Clebsch graphs without creating any C5s (or any cycles) that cross the cut?
    # No, that's wrong.
    # If I add edge (u,v), and there is a path from v to u in the existing graph, I create a cycle.
    # But G1 and G2 are disjoint. There is NO path from v (in G2) to u (in G1) in the existing graph.
    # So adding a single edge between two disjoint components creates NO cycles.
    # Therefore, I can add as many edges between the two Clebsch graphs as I want, and I will NOT create any new cycles that involve vertices from both components?
    # Wait. If I add edge (u1, v1) and edge (u2, v2).
    # Can I form a cycle u1-v1-...-v2-u2-...-u1?
    # The path v1-...-v2 must be in G2.
    # The path u2-...-u1 must be in G1.
    # So yes, if I add multiple edges, I can create cycles that go G1 -> G2 -> G1 -> G2...
    # Specifically, if I add edges (u1, v1) and (u2, v2), and there is a path v1 to v2 in G2 and u2 to u1 in G1, I form a cycle of length:
    # 1 (u1-v1) + len(path v1-v2 in G2) + 1 (v2-u2) + len(path u2-u1 in G1).
    # If len(path v1-v2)=2 and len(path u2-u1)=1, total length = 1+2+1+1 = 5. C5!
    # So I must be careful.
    
    # However, I can start with 2 Clebsch graphs and add edges one by one, checking for C5s.
    # And check alpha.
    
    # Let's implement this.
    
    # 1. Generate Clebsch graph.
    # Clebsch graph vertices: {0,1}^4.
    # Edges: (x,y) if x and y differ in exactly 2 bits.
    # n=16.
    
    def get_clebsch():
        n = 16
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                # Check if i and j differ in exactly 2 bits
                diff = bin(i ^ j).count('1')
                if diff == 2:
                    edges.append([i, j])
        return n, edges
    
    n1, e1 = get_clebsch()
    n2, e2 = get_clebsch()
    
    # Offset e2 by n1
    e2 = [[u + n1, v + n1] for u, v in e2]
    
    current_edges = e1 + e2
    current_n = n1 + n2 # 32
    
    # Check alpha and C5 for this base graph.
    # Alpha of 2 disjoint Clebsch is 10.
    # We need to reduce alpha to < 9.
    # We will add edges between the two parts.
    
    # To efficiently check C5 and Alpha, we might need bitmasks or adjacency lists.
    # Given n=32, we can use bitmasks (integers).
    
    def check_c5_and_alpha(n, edges):
        # Build adjacency bitmask
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check C5
        # A C5 is a cycle of length 5.
        # We can check for cycles of length 5 by iterating over all 5-tuples? 32^5 is too big.
        # Better: For each edge (u,v), look for a path of length 4 from v to u?
        # Or, use the property that a graph is C5-free iff...
        # Let's just do a DFS/BFS for cycles of length 5?
        # Or, check all subsets of 5 vertices? C(32,5) = 201376.
        # For each subset, check if it contains a C5.
        # A 5-vertex induced subgraph is a C5 if it has 5 edges and is a cycle.
        # Or if it contains a C5 as a subgraph.
        # Since we only care about "no 5-cycle as a subgraph", we just need to ensure no 5 vertices form a cycle (induced or not).
        # Actually, if a graph contains a C5, it contains an induced C5? No.
        # But if it contains a C5, there is a set of 5 vertices that form a cycle.
        # So checking all 5-vertex subsets for "is there a cycle of length 5 using these vertices" is sufficient.
        
        # Optimization:
        # For each pair of non-adjacent vertices (u,v), is there a path of length 4?
        # This is getting complex.
        
        # Let's use a simpler check for C5 in small graphs.
        # Iterate over all edges (u,v).
        # Find common neighbors of u and v? No.
        
        # Let's stick to the subset check but optimize it.
        # For each 5-subset S, check if the induced subgraph G[S] has a Hamiltonian cycle?
        # No, we just need ANY cycle of length 5.
        # In a 5-vertex graph, a cycle of length 5 is the only possible cycle of length 5.
        # So we just check if G[S] contains a C5.
        
        # To speed up, we can precompute something?
        # With n=32, 200k subsets * (check 5x5 matrix) might be slow in Python if done naively.
        # 200k * 25 ops = 5M ops. Should be fine within 1s?
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        for m in range(l+1, n):
                            verts = [i, j, k, l, m]
                            # Check if these 5 vertices form a C5
                            # Count edges in G[S]
                            edge_count = 0
                            for a in range(5):
                                for b in range(a+1, 5):
                                    if (adj[verts[a]] >> verts[b]) & 1:
                                        edge_count += 1
                            # A C5 has exactly 5 edges.
                            # If edge_count < 5, no C5.
                            # If edge_count == 5, it IS a C5 (since 5 vertices, 5 edges, connected implies cycle).
                            # Wait, 5 vertices and 5 edges can be a C5 or a "house" (C4+triangle)?
                            # No, a house has 5 vertices and 5 edges?
                            # House: 1-2-3-4-1 (4 edges) + 2-5, 3-5 (2 edges) = 6 edges.
                            # What graph has 5 vertices and 5 edges?
                            # C5.
                            # Or a triangle with two pendant edges?
                            # Triangle (3 edges) + 2 edges attached to 2 vertices = 5 edges.
                            # Does that contain a C5? No.
                            # So edge_count == 5 is not sufficient.
                            # We must check if it IS a C5.
                            # A graph on 5 vertices is a C5 iff it is 2-regular.
                            # So check if all degrees in G[S] are 2.
                            
                            if edge_count == 5:
                                # Check degrees
                                degs = [0]*5
                                for a in range(5):
                                    for b in range(5):
                                        if a != b and (adj[verts[a]] >> verts[b]) & 1:
                                            degs[a] += 1
                                if all(d == 2 for d in degs):
                                    return False # C5 found
                                    
        # Check Alpha
        # Max independent set.
        # For n=32, we can use a simple branch and bound or just check all subsets?
        # 2^32 is too big.
        # But we just need to know if alpha >= 9.
        # We can try to find an independent set of size 9.
        # Greedy?
        # Or, use the fact that if alpha < 9, we are good.
        # Let's use a simple backtracking to find an independent set of size 9.
        
        def has_independent_set_of_size_k(n, adj, k):
            # Backtracking
            # Order vertices by degree (descending) to prune?
            # Simple recursion
            def backtrack(available_mask, depth, current_set):
                if depth == k:
                    return True
                if bin(available_mask).count('1') < k - depth:
                    return False
                
                # Pick a vertex
                # Try to pick a vertex with high degree?
                # Or just pick the first available?
                # Let's pick the first available vertex v.
                # Branch 1: Include v.
                # Branch 2: Exclude v.
                
                # Find first set bit
                if available_mask == 0:
                    return False
                
                # Get least significant set bit
                v = (available_mask & -available_mask).bit_length() - 1
                # Wait, bit_length-1 gives index.
                # If mask is 1 (0b1), bit_length=1, idx=0. Correct.
                
                # Option 1: Include v
                # Remove v and its neighbors from available
                new_mask = available_mask & ~(1 << v) & ~adj[v]
                if backtrack(new_mask, depth + 1, current_set + [v]):
                    return True
                
                # Option 2: Exclude v
                new_mask2 = available_mask & ~(1 << v)
                if backtrack(new_mask2, depth, current_set):
                    return True
                
                return False
            
            return backtrack((1 << n) - 1, 0, [])
        
        if has_independent_set_of_size_k(n, adj, 9):
            return False # Alpha >= 9
        
        return True # Valid
    
    # Initial check
    if not check_c5_and_alpha(current_n, current_edges):
        # Should not happen for 2 Clebsch
        pass
    
    # Now, add edges between the two halves to reduce alpha.
    # We want to add edges such that no C5 is formed and alpha drops to 8.
    
    # Candidate edges: all pairs (u, v) with u < 16, v >= 16.
    candidate_edges = [(u, v) for u in range(16) for v in range(16, 32)]
    random.shuffle(candidate_edges)
    
    # We will try to add edges one by one.
    # If adding an edge creates a C5 or keeps alpha >= 9, we might still add it if it helps later?
    # No, we must maintain C5-free at all times.
    # And we want to eventually get alpha < 9.
    
    # Strategy: Randomly add edges. If C5 appears, revert.
    # Check alpha periodically.
    
    best_graph = {'n': current_n, 'edges': current_edges}
    
    # To speed up C5 check, we can incrementally check.
    # But for simplicity, let's just re-check.
    # 32 vertices, 200k subsets.
    # If we do this 100 times, it's 20M ops. Might be slow in Python.
    
    # Let's optimize the C5 check.
    # Instead of checking all 5-subsets, we can check only those involving the new edge?
    # A new C5 must involve the new edge (u,v).
    # So we only need to check 5-subsets containing u and v.
    # Number of such subsets: C(30, 3) = 4060.
    # This is much faster!
    
    def check_c5_with_new_edge(n, adj, u, v):
        # Check if adding edge (u,v) creates a C5.
        # The C5 must contain edge (u,v).
        # So we look for a path of length 4 from v to u?
        # Or, find 3 vertices a,b,c such that u-a-b-c-v is a path?
        # No, the cycle is u-v-c-b-a-u.
        # So we need a path of length 3 from v to u?
        # u-v (new), v-c, c-b, b-a, a-u.
        # So we need a path of length 3 from v to u in the OLD graph?
        # No, the path v-c-b-a-u uses edges in the OLD graph (except u-v is new).
        # Wait, the edge a-u must exist in the OLD graph.
        # And v-c, c-b, b-a must exist in the OLD graph.
        # So we are looking for a path of length 3 from v to u in the OLD graph?
        # No, the path is v-c-b-a, and then a-u is an edge.
        # So we need a path of length 3 from v to a, and a-u is an edge.
        # Or simply: Is there a path of length 4 from u to v in the OLD graph?
        # If there is a path u-x-y-z-v of length 4, then u-v-x-y-z-u is a C5?
        # No, the cycle would be u-v (new) + path v-z-y-x-u.
        # So yes, if there is a path of length 4 from u to v in the old graph, adding (u,v) creates a C5.
        
        # How to check for path of length 4?
        # BFS from u up to depth 4.
        
        from collections import deque
        q = deque()
        q.append((u, 0))
        visited = {u: 0}
        
        while q:
            node, dist = q.popleft()
            if dist == 4:
                if node == v:
                    return True # Found path of length 4
                continue
            
            for neighbor in range(n):
                if (adj[node] >> neighbor) & 1:
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        q.append((neighbor, dist + 1))
        
        return False
    
    # Let's use this faster check.
    
    # Re-initialize adj for the current graph
    adj = [0] * current_n
    for u, v in current_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # We will try to add edges.
    # We need to track alpha.
    # Checking alpha is the slow part.
    # Let's check alpha every 10 edges or so.
    
    added_edges = 0
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
            
        # Check if adding (u,v) creates a C5
        if check_c5_with_new_edge(current_n, adj, u, v):
            continue
            
        # Add the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        current_edges.append([u, v])
        added_edges += 1
        
        # Check alpha periodically
        if added_edges % 5 == 0:
            # Check if alpha < 9
            if not has_independent_set_of_size_k(current_n, adj, 9):
                # We found a valid graph with alpha < 9!
                # Return it
                return {'n': current_n, 'edges': current_edges}
                
    # If we didn't find a valid graph, return the best we have?
    # The seed is 28. We have 32 vertices but alpha might be >= 9.
    # If alpha >= 9, it's invalid.
    # We should return a valid graph.
    # The seed is valid.
    # Let's return the seed if we fail.
    
    # But we can try to prune the 32-vertex graph to make it valid?
    # Or just return the seed.
    
    # Let's try one more thing:
    # If we can't reduce alpha with edges, maybe we can remove some vertices?
    # No, we want to maximize n.
    
    # Let's return the seed if we fail.
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}

# EVOLVE-BLOCK-END
