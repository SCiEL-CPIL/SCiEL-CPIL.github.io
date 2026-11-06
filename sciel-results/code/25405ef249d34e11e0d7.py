import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for evaluator kill at 150s

    # Fallback: Andrasfai graph A_5 (n=10) or similar small valid graph
    # Actually, let's use a known valid graph if we can construct it quickly.
    # For now, fallback to a small valid graph to ensure return.
    # A_5 (n=10) is valid.
    # Let's try to find a larger one.
    
    best_n = 10
    best_edges = [[i, j] for i in range(10) for j in range(i + 1, 10) if (j - i) % 10 in [1, 4, 5, 6, 9]] # C10? 
    # C10 is triangle free. Alpha(C10) = 5. Valid.
    # Let's just use a simple C10 as fallback.
    best_edges = [[i, (i+1)%10] for i in range(10)]
    best_n = 10

    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 2000 # Increase attempts due to faster check
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate random maximal triangle-free graph
            adj = [set() for _ in range(n)]
            
            # Random order of vertices to break symmetry
            verts = list(range(n))
            random.shuffle(verts)
            
            # Add edges in random order, checking triangle constraint
            # To make it maximal, we iterate through all pairs.
            # But random order of pairs is better for diversity?
            # Let's just do a random permutation of all edges.
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding (u,v) creates a triangle
                # Triangle exists if there is w in adj[u] AND adj[v]
                common = adj[u] & adj[v]
                if not common:
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Check validity using Greedy Coloring of Complement
            # Complement graph: edge (u,v) exists if NOT (u,v) in adj
            # We want to color the complement.
            # If max color < 10, then alpha < 10.
            
            # Build complement adjacency
            # For speed, we can just check degrees or use a simple greedy.
            # Greedy coloring of complement:
            # Colors assigned to vertices 0..n-1.
            colors = [-1] * n
            max_color = 0
            
            # Order vertices by degree in complement? 
            # Degree in complement = n - 1 - degree in G.
            # High degree in G -> Low degree in complement -> Easy to color?
            # Let's just use natural order or random.
            
            for u in range(n):
                # Find smallest available color
                used_colors = set()
                for v in range(n):
                    if v == u:
                        continue
                    # If (u,v) is NOT an edge in G, it IS an edge in complement
                    if v not in adj[u]:
                        if colors[v] != -1:
                            used_colors.add(colors[v])
                
                c = 0
                while c in used_colors:
                    c += 1
                colors[u] = c
                if c > max_color:
                    max_color = c
                    if max_color >= 10:
                        break
            
            if max_color < 10:
                # Valid graph found
                # Convert adj to edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break

    return {'n': best_n, 'edges': best_edges}
