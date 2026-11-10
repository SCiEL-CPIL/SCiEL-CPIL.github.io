import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 16
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = 0
    best_graph = set()
    
    # Helper to check if adding edge (u, v) creates Q3
    # Uses bitmask adjacency for speed
    def creates_q3(adj, u, v):
        # adj[i] is a bitmask of neighbors of vertex i
        # We need to find a Q3 containing edge (u,v)
        # Q3 structure: two squares sharing edge (u,v)
        # Square 1: u-a-b-v-u
        # Square 2: u-c-d-v-u
        # Plus edges: a-c, b-d, and a-e-c-f-b-d... 
        # Actually Q3 has 8 vertices: let's label them
        # If edge is (u,v), then in Q3 the two squares sharing (u,v) are:
        # u-v-a-b-u and u-v-c-d-u where a,b,c,d are distinct
        # Then we need edges: a-c, b-d, and the remaining 4 vertices form a 4-cycle
        # Wait, Q3 on 8 vertices: if we pick edge (u,v), the other 6 vertices
        # form: two vertices adjacent to both u and v (call them p, q),
        # two adjacent to u and one of {p,q}, etc.
        
        # Let me think more carefully. Q3 = cube. Pick an edge uv.
        # The cube has 8 vertices. Edge uv belongs to exactly 2 faces (squares).
        # Face 1: u-v-x-y-u where x adj to v and u, y adj to u and v... 
        # No. In a cube, each edge is in exactly 2 squares.
        # Square 1: u, v, a, b where a~v, b~u, a~b
        # Square 2: u, v, c, d where c~v, d~u, c~d
        # The remaining 2 vertices: e, f
        # e is adj to a and c, f is adj to b and d, e~f
        # Also: a~e, c~e, b~f, d~f
        
        # So: find all a such that u~a and a~v (a is common neighbor of u and v)
        # Wait no. In square u-v-a-b-u: u~v, v~a, a~b, b~u. So a is adj to v, b is adj to u.
        # And a~b.
        
        # Let's define:
        # N_u = neighbors of u (excluding v since we're adding it, but v might already be there)
        # N_v = neighbors of v (excluding u)
        
        # For the check when adding edge (u,v):
        # We temporarily add u~v, so:
        # Common neighbors of u and v (after adding): let's call this set C
        # But actually we need to be more careful.
        
        # After adding (u,v):
        # u's neighbors: old_N(u) ∪ {v}
        # v's neighbors: old_N(v) ∪ {u}
        
        # A square through (u,v) has form: u-v-a-b-u
        # where a ∈ N(v) \ {u}, b ∈ N(u) \ {v}, and a~b
        # Note: a and b are distinct from u,v and from each other
        
        # So: find all pairs (a,b) where a~v, b~u, a~b, a≠b, a≠u,v, b≠u,v
        # Then for two such pairs (a,b) and (c,d) with {a,b}∩{c,d}=∅:
        # We need e,f such that:
        # e~a, e~c, f~b, f~d, e~f, and e,f not in {u,v,a,b,c,d}
        
        nu = adj[u]
        nv = adj[v]
        
        # Find all (a,b) pairs forming a square through (u,v)
        # a is neighbor of v (not u), b is neighbor of u (not v), a~b
        pairs = []
        for a in range(n):
            if a == u or a == v:
                continue
            if not (nv >> a) & 1:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if not (nu >> b) & 1:
                    continue
                if (adj[a] >> b) & 1:
                    pairs.append((a, b))
        
        if len(pairs) < 2:
            return False
        
        # Try all pairs of distinct pairs
        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                a, b = pairs[i]
                c, d = pairs[j]
                # Check distinctness
                if a == c or a == d or b == c or b == d:
                    continue
                
                used = {u, v, a, b, c, d}
                used_mask = 0
                for x in used:
                    used_mask |= (1 << x)
                
                # Find e: adjacent to a and c, not in used
                candidates_e = []
                for e in range(n):
                    if (used_mask >> e) & 1:
                        continue
                    if (adj[a] >> e) & 1 and (adj[c] >> e) & 1:
                        candidates_e.append(e)
                
                if not candidates_e:
                    continue
                
                for e in candidates_e:
                    # Find f: adjacent to b and d, not in used, not e, and e~f
                    for f in range(n):
                        if (used_mask >> f) & 1:
                            continue
                        if f == e:
                            continue
                        if (adj[b] >> f) & 1 and (adj[d] >> f) & 1:
                            if (adj[e] >> f) & 1:
                                return True
        
        return False

    # Multi-restart greedy with local search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        random.shuffle(all_edges)
        
        # Build adjacency bitmask
        adj = [0] * n
        current_graph = set()
        
        for edge in all_edges:
            u, v = edge
            # Check if adding creates Q3
            # Temporarily add u-v to adj for checking
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if creates_q3(adj, u, v):
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                current_graph.add((u, v))
        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
        
        # Local search: try to add any missing edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for edge in all_edges:
                if edge in current_graph:
                    continue
                u, v = edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not creates_q3(adj, u, v):
                    current_graph.add(edge)
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
        
        # Edge swap local search: remove one edge, try to add two others
        # This helps escape local optima
        if time.time() < deadline:
            # Rebuild adj from current_graph
            adj = [0] * n
            for u, v in current_graph:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Try removing each edge and see if we can add more
            edges_list = list(current_graph)
            random.shuffle(edges_list)
            
            for rem_edge in edges_list:
                if time.time() > deadline:
                    break
                ru, rv = rem_edge
                # Remove
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                current_graph.discard(rem_edge)
                
                # Try to add any missing edge
                added_any = False
                for add_edge in all_edges:
                    if add_edge in current_graph:
                        continue
                    au, av = add_edge
                    adj[au] |= (1 << av)
                    adj[av] |= (1 << au)
                    if not creates_q3(adj, au, av):
                        current_graph.add(add_edge)
                        added_any = True
                    else:
                        adj[au] &= ~(1 << av)
                        adj[av] &= ~(1 << au)
                
                if len(current_graph) > best_edges:
                    best_edges = len(current_graph)
                    best_graph = set(current_graph)
                
                # If we couldn't add more, put the edge back
                if not added_any or len(current_graph) <= best_edges:
                    # Put back
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    current_graph.add(rem_edge)
        
        # Rebuild adj for next iteration
        adj = [0] * n
        for u, v in current_graph:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Safety: limit total restarts to avoid infinite loop
        if num_restarts > 2000:
            break
    
    # Convert to list of lists
    return [[u, v] for u, v in best_graph]
