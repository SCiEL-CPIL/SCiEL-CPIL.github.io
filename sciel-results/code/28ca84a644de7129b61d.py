import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Line-graph formulation:
    # Each potential edge in K16 is a node in the line graph.
    # Two nodes are adjacent if the corresponding edges form a triangle.
    # We want the maximum independent set in this line graph that yields a 4-chromatic graph.
    
    # Enumerate all possible edges in K16
    n = 16
    all_edges = []
    edge_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(all_edges)
            all_edges.append((i, j))
            edge_to_idx[(i, j)] = idx
    
    m = len(all_edges)  # 120 edges in K16
    
    # Build conflict graph: edge e1 conflicts with e2 if they share a vertex and form a triangle
    # Actually, two edges conflict if they share exactly one vertex (forming a path of length 2)
    # Wait - two edges form a triangle if they share a vertex AND the third edge exists.
    # For triangle-freeness: no three edges can form a triangle.
    # So we need: for any vertex v, the edges incident to v must form an independent set in the sense
    # that no two neighbors of v are connected by an edge.
    
    # Better approach: build the "triangle conflict" graph on the 120 edges.
    # Edge e1 = (a,b) conflicts with edge e2 = (c,d) if {a,b,c,d} has 3 distinct vertices 
    # and they could form a triangle. Specifically, if they share exactly one vertex, say v,
    # then e1=(v,x), e2=(v,y). They conflict if (x,y) is also an edge.
    # But we're choosing which edges to include, so we need: if we include e1=(v,x) and e2=(v,y),
    # we CANNOT include (x,y).
    
    # This is a 3-uniform hypergraph constraint, not a simple graph.
    # Let me use a different approach: branch-and-bound on adding edges.
    
    # Alternative: Use the fact that a triangle-free graph on n vertices has at most n^2/4 edges (Mantel's theorem).
    # For n=16, max is 64 edges. We need 4-chromatic.
    
    # Let me use a smarter approach: start with a known 4-chromatic triangle-free graph and try to add edges.
    
    # Start with Grötzsch graph (Mycielski of C5) - 11 vertices, 20 edges, 4-chromatic, triangle-free
    def get_groetzsch():
        # C5: 0-1-2-3-4-0
        edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
        # Mycielski: copies 5-9, top vertex 10
        for i in range(5):
            neighbors = [(i+1) % 5, (i-1) % 5]
            for nb in neighbors:
                edges.append([5+i, nb])
        for i in range(5):
            edges.append([10, 5+i])
        return edges
    
    groetzsch_edges = get_groetzsch()
    
    # Build adjacency
    adj = [set() for _ in range(16)]
    for u, v in groetzsch_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Vertices 11-15 are isolated initially
    # Try to add edges involving vertices 11-15 while maintaining triangle-freeness and 4-chromaticity
    
    # For each isolated vertex, find maximum number of edges we can add
    # without creating triangles
    
    # A vertex v can connect to a set S of other vertices if and only if
    # S is an independent set in the graph (no two vertices in S are adjacent)
    
    # For each new vertex, find a large independent set in the remaining graph
    
    def find_max_independent_set(adj, vertices, deadline):
        """Find a large independent set using branch and bound."""
        vertices = list(vertices)
        # Order by degree (ascending) for better pruning
        vertices.sort(key=lambda x: len(adj[x]))
        
        best_set = []
        best_size = 0
        
        def backtrack(current_set, available, depth):
            nonlocal best_set, best_size
            if time.time() > deadline:
                return
            if not available:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return
            
            # Upper bound: current_size + len(available)
            if len(current_set) + len(available) <= best_size:
                return
            
            # Pick the first vertex
            v = available[0]
            remaining = available[1:]
            
            # Option 1: Include v
            current_set.append(v)
            # Remove v and its neighbors from available
            new_available = [u for u in remaining if u not in adj[v]]
            backtrack(current_set, new_available, depth + 1)
            current_set.pop()
            
            # Option 2: Exclude v
            backtrack(current_set, remaining, depth + 1)
        
        backtrack([], vertices, 0)
        return best_set
    
    # For each isolated vertex, find max independent set to connect to
    final_edges = list(groetzsch_edges)
    
    # Process isolated vertices one by one
    isolated = [11, 12, 13, 14, 15]
    
    for v in isolated:
        # Find vertices that v can connect to (independent set in current graph)
        # v can connect to any subset of vertices that forms an independent set
        candidates = list(range(16))
        # Remove v itself
        candidates.remove(v)
        
        # Find max independent set among candidates
        # But we also need to ensure 4-chromaticity is maintained
        # For now, just find a large independent set
        independent_set = find_max_independent_set(adj, candidates, deadline)
        
        # Add edges from v to independent_set
        for u in independent_set:
            final_edges.append([v, u])
            adj[v].add(u)
            adj[u].add(v)
    
    # Verify triangle-free
    adj_final = [set() for _ in range(16)]
    for u, v in final_edges:
        adj_final[u].add(v)
        adj_final[v].add(u)
    
    is_valid = True
    for i in range(16):
        for j in adj_final[i]:
            if i >= j:
                continue
            common = adj_final[i] & adj_final[j]
            if common:
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid:
        return groetzsch_edges
    
    # Check 4-chromatic using backtracking
    def is_not_3_colorable(adj):
        colors = [-1] * 16
        
        def backtrack(v):
            if time.time() > deadline:
                return None  # timeout
            if v == 16:
                return True
            
            # Try colors 0, 1, 2
            for c in range(3):
                # Check if color c is valid for vertex v
                valid = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        # Try all 3-colorings
        return not backtrack(0)
    
    if not is_not_3_colorable(adj_final):
        # Graph is 3-colorable, need to fix
        # Fallback to just the Grötzsch graph
        return groetzsch_edges
    
    return final_edges
