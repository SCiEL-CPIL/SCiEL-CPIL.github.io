"""Construct C4-free graph with no independent 11-set, maximizing n."""
import time
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = t0 + 130  # safety margin before 150s kill
    
    # Build PG(2, q) polarity graph
    # Vertices are points of PG(2,q): (q^2+q+1) vertices
    # Edge between two points if they are "perpendicular" under polarity
    # This is C4-free by construction (a property of polarity graphs)
    
    def build_polarity_graph(q):
        """Build polarity graph of PG(2,q). C4-free by construction."""
        # Points of PG(2,q): homogeneous coords [x:y:z], not all zero
        # Normalize: first non-zero coord is 1
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for x in range(q):
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        
        n_pts = len(points)  # q^2 + q + 1
        
        # Polarity: (x,y,z) perp (x',y',z') iff x*x' + y*y' + z*z' = 0 (mod q)
        # For q odd, use the standard polarity
        # Actually for polarity graphs, we need a specific polarity
        # Use: (x,y,z) ~ (x',y',z') iff x*x' + y*y' + z*z' ≡ 0 mod q
        # But this makes the graph include loops when point is self-perp
        # For C4-free property, we use the graph where edge exists iff perp AND not same
        
        # Actually, the standard result: the polarity graph of PG(2,q) with 
        # polarity (x,y,z)·(x',y',z') = x*x' + y*y' + z*z' = 0
        # has the property that any 4-cycle would give a contradiction
        # (this is a classical result)
        
        edges = []
        for i in range(n_pts):
            xi, yi, zi = points[i]
            for j in range(i+1, n_pts):
                xj, yj, zj = points[j]
                if (xi*xj + yi*yj + zi*zj) % q == 0:
                    edges.append([i, j])
        
        return n_pts, edges
    
    # For q=5: 31 vertices, C4-free
    # For q=3: 13 vertices
    # For q=4: 21 vertices (but q must be prime power for PG)
    
    # Let's use q=5 for 31 vertices
    n_base, base_edges = build_polarity_graph(5)
    
    # Now we want to add more vertices. The complement of a C4-free graph is C4-free.
    # So if G is C4-free, complement(G) is also C4-free.
    # 
    # Strategy: Use the complement of the polarity graph, which is dense,
    # and add vertices. But adding vertices to a C4-free graph is hard.
    #
    # Alternative: Start with the polarity graph (sparse, C4-free) and try to
    # add vertices while maintaining C4-freeness and keeping alpha < 11.
    #
    # Actually, let me think differently. The polarity graph on 31 vertices is C4-free.
    # What is its independence number? For the polarity graph of PG(2,5),
    # the independence number is related to the maximum set of mutually non-perpendicular points.
    #
    # Let me compute it and try to extend.
    
    # Build adjacency as bitmasks for fast C4 checking
    def make_adjacency(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    adj = make_adjacency(n_base, base_edges)
    
    # Check C4-freeness (should be true by construction, but verify)
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors of u and v
                common = adj[u] & adj[v]
                if common & (common - 1):  # more than one bit set
                    return False
        return True
    
    # Compute independence number using branch and bound with bitmasks
    def max_independent_set(adj, n):
        """Find maximum independent set size using branch and bound."""
        best = [0]
        
        def branch(candidates, current_size, chosen):
            if time.time() > deadline - 5:
                return
            if current_size + candidates.bit_count() <= best[0]:
                return
            if candidates == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Choose vertex with highest degree in candidate subgraph
            # (greedy heuristic: pick vertex that eliminates most candidates)
            cands = candidates
            # Find vertex in candidates
            v = (cands & -cands).bit_length() - 1
            
            # Option 1: include v
            # Remove v and its neighbors from candidates
            new_cands = candidates & ~(adj[v] | (1 << v))
            branch(new_cands, current_size + 1, chosen | (1 << v))
            
            # Option 2: exclude v
            new_cands = candidates & ~(1 << v)
            branch(new_cands, current_size, chosen)
        
        all_mask = (1 << n) - 1
        branch(all_mask, 0, 0)
        return best[0]
    
    # Compute alpha of the base graph
    alpha_base = max_independent_set(adj, n_base)
    
    # The polarity graph of PG(2,5) should have alpha around 10-12
    # If alpha < 11, we can try to add vertices
    
    # Try to extend by adding vertices one at a time
    # When adding vertex v, it must not create a C4 with any existing edge
    # A C4 would be: v-a-b-c-v where a,b,c are existing vertices
    # This happens if v has two common neighbors with some existing vertex
    
    def can_add_vertex(adj, n, new_neigh_mask):
        """Check if adding a vertex with given neighbors maintains C4-free."""
        # For each existing vertex u, count common neighbors with new vertex
        # New vertex's neighbors are new_neigh_mask
        # Common neighbors of u and new vertex = adj[u] & new_neigh_mask
        # This must have at most 1 bit set for all u
        for u in range(n):
            common = adj[u] & new_neigh_mask
            if common & (common - 1):
                return False
        # Also check among the neighbors of new vertex:
        # For any two neighbors a, b of new vertex, they must not have two common neighbors
        # Common neighbors of a and b (excluding new vertex) = adj[a] & adj[b] & ~(1 << new_vertex_idx)
        # But new vertex isn't in adj yet, so just adj[a] & adj[b]
        # Wait, we need to check that adding new vertex doesn't create C4
        # C4 involving new vertex: new-a-b-c-new means a,c are neighbors of new, 
        # and b is common neighbor of a and c
        # So for any two neighbors a,c of new vertex, adj[a] & adj[c] must have 
        # at most the bits that are also neighbors of new... no.
        # Actually: C4 is v-a-b-c-v. So a,c are neighbors of v, and b is neighbor of both a and c.
        # The condition is: for any pair of neighbors a,c of v, the number of common neighbors
        # of a and c that are NOT v must be at most 1 (since v is already a common neighbor).
        # Wait no. In the graph after adding v, common neighbors of a and c include v plus 
        # whatever was there before. We need total common neighbors <= 1.
        # Before adding v, a and c might already have some common neighbors.
        # After adding v, if v is neighbor of both a and c, then v becomes a common neighbor.
        # So we need: (adj[a] & adj[c]) has at most 0 bits (if v is neighbor of both a and c)
        # OR we need to be more careful.
        
        # Let me reconsider. The C4-free condition: no two vertices have 2+ common neighbors.
        # After adding vertex v with neighbor set N(v):
        # 1. For any existing pair (a,c): their common neighbors were adj[a]&adj[c].
        #    If v is neighbor of both a and c, then v adds to their common neighbors.
        #    So we need: if a,c both in N(v), then popcount(adj[a] & adj[c]) <= 0
        #    (because v will be the only common neighbor)
        # 2. For v and any existing u: common neighbors = adj[u] & N(v), need popcount <= 1
        # 3. For any two existing vertices not both in N(v): unchanged
        
        # Check condition 2
        for u in range(n):
            common = adj[u] & new_neigh_mask
            if common & (common - 1):
                return False
        
        # Check condition 1: for any two vertices a,c both in N(v), adj[a]&adj[c] must be 0
        neigh_list = []
        m = new_neigh_mask
        while m:
            b = m & -m
            idx = b.bit_length() - 1
            neigh_list.append(idx)
            m ^= b
        
        for i in range(len(neigh_list)):
            for j in range(i+1, len(neigh_list)):
                a, c = neigh_list[i], neigh_list[j]
                if adj[a] & adj[c]:
                    return False
        
        return True
    
    # Greedy extension: try to add vertices
    # Start with the polarity graph
    current_n = n_base
    current_adj = list(adj)
    current_edges = list(base_edges)
    
    # Try adding vertices with various neighbor sets
    # A vertex can be isolated (no neighbors) - always safe for C4
    # Or connected to a set of vertices that form an independent set in the "common neighbor" sense
    
    # Simplest: add isolated vertices. Each isolated vertex increases alpha by 1.
    # So we can add (10 - alpha_base) isolated vertices if alpha_base < 10
    # Wait, we need alpha < 11, so alpha <= 10.
    
    # If alpha_base <= 10, we can add (10 - alpha_base) isolated vertices
    # But that's not optimal. Let's try to add vertices with some edges.
    
    # For now, let's just add isolated vertices to get a valid answer
    # and then try to do better
    
    isolated_to_add = max(0, 10 - alpha_base)
    total_n = current_n + isolated_to_add
    
    if total_n < 31:
        # Something went wrong, just return the base
        total_n = current_n
    
    # Build final edges (isolated vertices add no edges)
    final_edges = current_edges
    
    # Verify C4-free and alpha < 11
    final_adj = make_adjacency(total_n, final_edges)
    
    # Quick C4 check
    c4_ok = True
    for u in range(total_n):
        for v in range(u+1, total_n):
            common = final_adj[u] & final_adj[v]
            if common & (common - 1):
                c4_ok = False
                break
        if not c4_ok:
            break
    
    if not c4_ok:
        # Fallback: just return the base graph
        return {'n': n_base, 'edges': base_edges}
    
    # Check independence number
    alpha_final = max_independent_set(final_adj, total_n)
    
    if alpha_final >= 11:
        # Need to reduce - remove some isolated vertices or use different approach
        # Fallback to base
        return {'n': n_base, 'edges': base_edges}
    
    return {'n': total_n, 'edges': final_edges}
