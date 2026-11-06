import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    # Known valid: 8 copies of K4 (n=32)
    # This graph is C5-free (components are K4, max cycle 4)
    # and has alpha=8 (one vertex from each K4)
    n = 32
    adj = [set() for _ in range(n)]
    
    for b in range(8):
        for i in range(4):
            for j in range(i+1, 4):
                u = 4*b + i
                v = 4*b + j
                adj[u].add(v)
                adj[v].add(u)
    
    # Fast C5 check: try all paths of length 4 and see if they close
    def has_c5(adj, n):
        # For each ordered pair (u,v) with u<v, try to find a path u-a-b-c-v
        # where a,b,c are distinct and not u,v
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Path: u - a - b - c - v
                # a must be neighbor of u, not v
                # b must be neighbor of a, not u,v
                # c must be neighbor of b, not u,v,a
                # and c must be neighbor of v
                for a in adj[u]:
                    if a == v:
                        continue
                    for b in adj[a]:
                        if b == u or b == v:
                            continue
                        for c in adj[b]:
                            if c == u or c == v or c == a:
                                continue
                            if v in adj[c]:
                                # Check all vertices distinct: u,a,b,c,v
                                # u != a (a in adj[u], simple graph), a != b, b != c
                                # c != v (checked), v != u (v > u)
                                # a != v (checked), b != v (checked)
                                return True
        return False

    # Check if adding edge (u,v) creates a C5
    def adding_edge_creates_c5(adj, u, v, n):
        # Adding edge (u,v) creates a C5 iff there exists a path of length 4
        # from u to v in the current graph (without the edge u-v)
        # Path: u - a - b - c - v, all distinct
        # a in adj[u], b in adj[a], c in adj[b], v in adj[c]
        # a != v, b != u, b != v, c != u, c != v, c != a, b != a
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[a]:
                if b == u or b == v:
                    continue
                for c in adj[b]:
                    if c == u or c == v or c == a:
                        continue
                    if v in adj[c]:
                        return True
        return False

    # Independence number check via greedy coloring on complement
    def has_independent_set_9(adj, n):
        # Complement graph: edge (u,v) in complement iff NOT edge in adj
        # We need to check if alpha(complement) >= 9, i.e., if complement has
        # a clique of size 9, i.e., if original has independent set of size 9
        # Use greedy approach: try to build independent set greedily
        # More reliable: use branch and bound for small n
        # For n=32, let's use a simple but effective method
        
        # Greedy independent set with multiple orderings
        vertices = list(range(n))
        best = 0
        
        # Try greedy with different orderings
        for _ in range(50):
            random.shuffle(vertices)
            indep = set()
            for v in vertices:
                # Check if v can be added to independent set
                can_add = True
                for u in indep:
                    if u in adj[v]:
                        can_add = False
                        break
                if can_add:
                    indep.add(v)
            if len(indep) > best:
                best = len(indep)
            if best >= 9:
                return True
        
        # If greedy found 9, we're done
        # But greedy might miss it. Let's do a more careful check.
        # For the specific structure (8 K4s + some cross edges), 
        # we can reason: without cross edges, alpha = 8.
        # Cross edges can only decrease alpha (by connecting vertices
        # that were previously in different components' independent sets)
        # Wait, that's wrong. Cross edges in the original graph mean
        # those vertices CANNOT both be in an independent set.
        # So adding edges can only DECREASE alpha.
        # Therefore, if alpha was 8 before, it stays <= 8 after adding edges.
        # This is a key insight!
        
        return False

    # Key insight: Adding edges to a graph can only DECREASE the independence number.
    # Since our base graph has alpha = 8, any supergraph also has alpha <= 8.
    # So we only need to check C5-freeness when adding edges!
    
    # Now, try to add cross-edges between different K4 blocks
    # to potentially allow more vertices or just make the graph denser
    # But wait - we want to MAXIMIZE n, not density.
    # The constraint is n <= 35, so we could add up to 3 more vertices.
    
    # Actually, let me reconsider. We have n=32 vertices. We can't add vertices
    # without also adding edges. Let me think about this differently.
    
    # Can we construct a graph on n=33, 34, or 35 vertices?
    # We need C5-free and alpha <= 8.
    
    # Strategy: Start with 8 K4s (32 vertices, alpha=8).
    # Try to add a 33rd vertex connected to some existing vertices.
    # The new vertex must not create a C5 and must not increase alpha.
    
    # If we add vertex 32 connected to some vertices, alpha might stay <= 8
    # if the new vertex is "well-connected" enough.
    
    # Let's try: add vertex 32, connect it to one vertex from each K4 block.
    # Then any independent set can include at most one from each K4 (8 total from K4s)
    # plus possibly vertex 32. But if vertex 32 is connected to at least one vertex
    # in each independent set choice... hmm, this is tricky.
    
    # Actually, if vertex 32 is connected to vertex v_b in block b for each b,
    # then an independent set that includes vertex 32 cannot include any v_b.
    # But it could include other vertices from each block.
    # From each K4 block, if we exclude v_b, we can still include at most 1 vertex.
    # So max independent set including vertex 32: 1 (for v32) + 8 (one from each block,
    # but not v_b) = 9. That's too many!
    
    # So we need vertex 32 to be connected to MORE vertices to reduce alpha.
    # If vertex 32 is connected to 2 vertices in each block, then from each block
    # we can pick at most 1 vertex that's not connected to v32.
    # Still gives 1 + 8 = 9.
    
    # If vertex 32 is connected to 3 vertices in each block, then from each block
    # we can pick at most 1 vertex not connected to v32 (since K4, 4-3=1).
    # Still 1 + 8 = 9.
    
    # If vertex 32 is connected to all 4 vertices in each block, then from each block
    # we can pick 0 vertices not connected to v32.
    # So max independent set including v32 = 1 + 0 = 1.
    # Max independent set not including v32 = 8 (one from each K4).
    # Total alpha = 8. 
    # But then v32 is connected to all 32 other vertices, making it a universal vertex.
    # Does this create a C5? A universal vertex in a C5-free graph...
    # If v32 is connected to all, then any cycle through v32 uses v32 and two neighbors.
    # A C5 through v32 would be v32-a-b-c-v32, which is a 4-cycle, not 5.
    # Actually v32-a-b-c-d-v32 would be a C5. Need a,b,c,d distinct, not v32,
    # with edges v32-a, a-b, b-c, c-d, d-v32.
    # Since v32 is universal, we need a-b-c-d to be a path of length 3 in G-v32.
    # G-v32 is 8 disjoint K4s. In a K4, the longest path is 3 (e.g., 0-1-2-3).
    # So a path of length 3 exists within a single K4!
    # Example: in block 0, vertices 0,1,2,3 form K4. Path 0-1-2-3 has length 3.
    # Then v32-0-1-2-3-v32 is a C5!
    # So making v32 universal creates C5s.
    
    # We need a more careful construction for n=33.
    
    # Alternative: Use a different base structure.
    # What about using fewer, larger cliques?
    # C5-free graphs: complete graphs K_k for k>=5 contain C5? No!
    # K5 contains C5 as a subgraph. So max clique size is 4.
    # So we can't use K5 or larger.
    
    # What about other C5-free structures?
    # Complete bipartite graphs K_{m,n} are C5-free (bipartite).
    # K_{4,4} has 8 vertices, alpha = max(4,4) = 4.
    # 4 copies of K_{4,4}: 32 vertices, alpha = 4*4 = 16. Too high.
    
    # Turan graph T(n,4) is C5-free? No, Turan graphs can contain C5.
    # Actually, Turan graph T(n,r) has no K_{r+1}, but may have C5.
    
    # Let me think about what C5-free graphs look like.
    # A graph is C5-free if and only if... there's no simple characterization.
    # But we know: forests are C5-free, complete graphs K_k for k<=4 are C5-free,
    # complete bipartite graphs are C5-free.
    
    # For n=33, let me try a different approach:
    # Use 7 copies of K4 (28 vertices) + 5 more vertices arranged carefully.
    # Or use a mix of structures.
    
    # Actually, let me just try to add vertices to the 8 K4 structure
    # with careful edge choices and verify with the C5 checker.
    
    # Let's try n=33: add vertex 32, connect it to some subset of existing vertices.
    # We need: (1) no C5 created, (2) alpha <= 8.
    
    # For alpha: as argued, if v32 is connected to at least 3 vertices in each K4 block,
    # then alpha <= 8. But that creates C5s.
    
    # Let me try a different connectivity pattern for v32.
    # Connect v32 to exactly 2 vertices in each K4 block.
    # Then from each block, an independent set can include at most 2 vertices
    # (the 2 not connected to v32) if v32 is NOT in the independent set.
    # Wait, in a K4, if we exclude 2 specific vertices, we have 2 left, which form an edge,
    # so we can pick at most 1 from them.
    # So if v32 is not in IS: max 1 per block = 8.
    # If v32 is in IS: can't pick any of its 16 neighbors. From each block, 2 vertices
    # are not neighbors of v32, and they form an edge (in K4), so pick at most 1.
    # Total: 1 + 8 = 9. Still too high!
    
    # Connect v32 to 3 vertices in each block:
    # If v32 not in IS: 1 per block = 8.
    # If v32 in IS: 1 vertex per block not connected to v32, pick 1. Total 1+8=9.
    
    # Connect v32 to 4 vertices in each block (all):
    # Creates C5 as shown.
    
    # So with 8 K4s, we can't add a 33rd vertex while keeping alpha <= 8 and C5-free.
    
    # What if we modify the base structure?
    # Instead of 8 K4s, use a different C5-free structure with alpha=8 and n=32,
    # that has more "room" to add a vertex.
    
    # Or: use 7 K4s (28 vertices, alpha=7) and add 5 vertices with careful connections
    # to keep alpha <= 8 and C5-free.
    
    # Let me try: 7 K4s + 5 new vertices.
    # The 5 new vertices need to be connected such that:
    # - No C5 is formed
    # - Alpha <= 8
    # 
    # If the 5 new vertices form an independent set among themselves,
    # and each is connected to some of the K4 vertices...
    # Max alpha would be 7 (from K4s) + 5 (new vertices) = 12 if new vertices
    # are not connected to K4 vertices. Too high.
    
    # This is getting complex. Let me just stick with n=32 and verify it's valid.
    # The current solution already achieves n=32, which is the known best.
    # Trying to go higher requires a fundamentally different construction.
    
    # Let me verify the current 8 K4 construction is valid and return it.
    # This is the safest approach given the time constraints.
    
    # Quick validation
    def is_valid(n, adj):
        # Check C5
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0: continue
                for v2 in adj[v1]:
                    if v2 in (v0, v1): continue
                    for v3 in adj[v2]:
                        if v3 in (v0, v1, v2): continue
                        for v4 in adj[v3]:
                            if v4 in (v0, v1, v2, v3): continue
                            if v0 in adj[v4]:
                                return False
        # Check alpha: 8 K4s, alpha = 8
        return True

    if not is_valid(n, adj):
        n = 1
        adj = [set()]

    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
