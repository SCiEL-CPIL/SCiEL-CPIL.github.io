import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with margin
    
    # Build Grötzsch graph on vertices 0..10
    # It's the Mycielski graph of C5
    edges = []
    # C5: 0-1-2-3-4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Mycielski: add vertex 5 (copy of 0), 6 (copy of 1), etc., and vertex 10 (apex)
    # Actually, standard Grötzsch: vertices 0-4 form C5, 5-9 are "copies", 10 is apex
    # Let me use the standard construction
    # C5 vertices: 0,1,2,3,4
    # Copies: 5,6,7,8,9 where i connects to 5+(i+1)%5 and 5+(i-1)%5... 
    # Actually simpler: vertex 5+i is adjacent to vertices i and i+1 (mod 5) in C5
    # And vertex 10 is adjacent to all of 5,6,7,8,9
    
    # Let me rebuild properly
    edges = []
    # C5
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    
    # Mycielski copies: vertex 5+i is adjacent to vertex i and vertex (i+1)%5 in C5
    # Wait, standard Mycielski: for each vertex v in G, add v' adjacent to N(v)
    # For C5, vertex i has neighbors (i-1)%5 and (i+1)%5
    # So vertex 5+i is adjacent to (i-1)%5 and (i+1)%5
    for i in range(5):
        edges.append([5 + i, (i - 1) % 5])
        edges.append([5 + i, (i + 1) % 5])
    
    # Apex vertex 10 adjacent to all Mycielski copies
    for i in range(5):
        edges.append([10, 5 + i])
    
    # Now add vertices 11-15 and try to add edges
    # Current edges count
    initial_edges = len(edges)
    
    def is_triangle_free(edge_list, new_edge):
        """Check if adding new_edge creates a triangle"""
        u, v = new_edge
        # Build adjacency from edge_list
        adj = set()
        for a, b in edge_list:
            adj.add((a, b))
            adj.add((b, a))
        # Check if there's a w such that (u,w) and (w,v) are both in adj
        for w in range(16):
            if w != u and w != v:
                if (u, w) in adj and (w, v) in adj:
                    return False
        return True
    
    def is_not_3_colorable(edge_list):
        """Check if graph is NOT 3-colorable using backtracking"""
        n = 16
        # Build adjacency
        adj = [set() for _ in range(n)]
        for a, b in edge_list:
            adj[a].add(b)
            adj[b].add(a)
        
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if all(colors[u] != c for u in adj[v]):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        
        return not backtrack(0)
    
    # Try to add vertices 11-15 with maximum edges
    # For each new vertex, try all possible connections to existing vertices
    # that don't create triangles and maintain non-3-colorability
    
    best_edges = edges[:]
    best_count = initial_edges
    
    # For each new vertex from 11 to 15, try adding it with various connection patterns
    for new_v in range(11, 16):
        if time.time() > deadline - 5:
            break
        # Try all subsets of existing vertices to connect new_v to
        # This is 2^11 = 2048 possibilities, manageable
        best_for_v = None
        best_for_v_count = -1
        
        existing_vertices = list(range(new_v))
        # Try all subsets - but 2^11 is too many if we check each
        # Instead, try greedy: add edges that don't create triangles
        # Start with no edges, add one by one
        
        # Actually, let's try: connect new_v to all possible vertices (maintaining triangle-free)
        # then check if still not 3-colorable
        
        # First, find which vertices new_v can connect to without creating triangles
        possible_connections = []
        for v in range(new_v):
            # Check if connecting new_v to v creates a triangle
            # Need to check if any common neighbor of v exists
            common_neighbors = []
            for w in range(new_v):
                if w != v:
                    # Check if v-w and w-new_v would create triangle
                    # i.e., is there an edge v-w?
                    has_vw = any((a == v and b == w) or (a == w and b == v) for a, b in best_edges)
                    # And new_v-w would be new
                    if has_vw:
                        common_neighbors.append(w)
            # If new_v connects to v, and there's an edge v-w, then new_v-w can't exist
            # So we need to check: for new_v-v to be valid, for all w with edge v-w, new_v-w must not be an edge
            # This is complex. Let me just check triangle-free when adding.
            possible_connections.append(v)
        
        # Greedy approach: add edges one at a time
        candidate_edges = best_edges[:]
        added_any = False
        
        # Try connecting new_v to each possible vertex
        for v in range(new_v):
            if time.time() > deadline - 5:
                break
            new_edge = [new_v, v]
            # Check triangle-free
            adj = set()
            for a, b in candidate_edges:
                adj.add((a, b))
                adj.add((b, a))
            valid = True
            for w in range(new_v):
                if w != v and w != new_v:
                    if (new_v, w) in adj and (w, v) in adj:
                        valid = False
                        break
            if valid:
                candidate_edges.append(new_edge)
                added_any = True
        
        # Check if still not 3-colorable
        if is_not_3_colorable(candidate_edges):
            if len(candidate_edges) > best_count:
                best_count = len(candidate_edges)
                best_edges = candidate_edges[:]
        else:
            # Try removing some edges to restore 4-chromatic property
            # Or just skip this vertex for now
            pass
    
    # Return the best we found
    return best_edges
