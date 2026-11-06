import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Precompute polarity graph of PG(2,7)
    # q = 7
    # Points are 1D subspaces of F_7^3
    # Hyperplanes are 2D subspaces
    # Incidence: point P is on hyperplane H if P subset H
    
    # Number of points = (7^3 - 1)/(7 - 1) = 343/6 = 57.16... wait
    # (7^3 - 1)/(7 - 1) = (343 - 1)/6 = 342/6 = 57
    # Number of hyperplanes = 57
    
    # Each point is on (7+1) = 8 hyperplanes
    # Each hyperplane contains (7+1) = 8 points
    # Total incidences = 57 * 8 = 456
    
    # Bipartite graph with 57 + 57 = 114 vertices
    # But we can use just the points as vertices, and edges between points
    # if they share a common hyperplane? No, that's not right.
    
    # The polarity graph: vertices are points of PG(2,q).
    # Two points are adjacent if one lies on the polar line of the other.
    # In PG(2,7), this gives a graph on 57 vertices.
    # Each vertex has degree 8 (since each point lies on 8 lines, and each line
    # contains 8 points, but the polarity is a bijection from points to lines).
    
    # Actually, let's construct it properly.
    # A polarity in PG(2,q) maps points to lines and lines to points.
    # The graph has vertices = points, and u~v if u lies on the polar line of v.
    
    # For q=7, we have 57 points.
    # Let's enumerate them.
    
    # We need to be careful about the time limit.
    # Let's just use the known structure: PG(2,7) polarity graph is C4-free with 57 vertices, degree 8.
    # We can take a subset of 47 vertices.
    
    # But we need to check if the induced subgraph on 47 vertices has min degree >= 47-39 = 8.
    # If we remove 10 vertices, the min degree might drop.
    # Each vertex has degree 8. If we remove 10 vertices, a vertex might lose up to 8 neighbors.
    # So we need to remove vertices carefully.
    
    # Alternative: use the complement graph approach.
    # We want a graph G on n vertices with no C4 and min degree >= n-39.
    # Equivalently, the complement G' has no K_{1,39} (i.e., max degree <= 38) and no C4' (complement of C4 is two disjoint edges plus two isolated? No, complement of C4 is C4).
    # Wait, complement of C4 is also C4. So G' must also be C4-free.
    
    # So we need a C4-free graph on n vertices with max degree <= 38.
    # For n=47, we need a C4-free graph with 47 vertices and max degree <= 38.
    # This is easy: take a random graph with low degree.
    # But we also need the original graph to have min degree >= 8.
    # So the complement must have max degree <= 38.
    
    # Let's try to construct a C4-free graph on 47 vertices with max degree 38.
    # We can use a greedy algorithm to add edges while avoiding C4 and keeping degree <= 38.
    
    best_n = 39
    best_edges = []
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg_req = n - 39
        
        # Try to construct a C4-free graph on n vertices with min degree >= min_deg_req
        # using a greedy approach with randomization
        
        for trial in range(200):
            if time.time() > deadline:
                break
                
            adj = [set() for _ in range(n)]
            
            # Create list of all possible edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    edges.append((i, j))
            random.shuffle(edges)
            
            # Greedy addition
            for u, v in edges:
                if time.time() > deadline:
                    break
                
                # Check if adding (u, v) creates a C4
                # C4: u-a-b-v where a in N(u), b in N(v), a~b, a!=b, a!=v, b!=u
                # Check if there's an edge between N(u) and N(v) excluding u and v
                
                found_c4 = False
                for a in adj[u]:
                    if a == v:
                        continue
                    for b in adj[v]:
                        if b == u or b == a:
                            continue
                        if b in adj[a]:
                            found_c4 = True
                            break
                    if found_c4:
                        break
                
                if not found_c4:
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Check min degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_req:
                best_n = n
                best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
                return {'n': n, 'edges': best_edges}
    
    return {'n': best_n, 'edges': best_edges}
