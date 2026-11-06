import random
import numpy as np

def construct():
    n = 40
    # Initialize with the current known good solution
    # Star in A: 0 connected to 1..10
    # Removed cross edges: (0, 20) to (0, 28)
    
    edges = set()
    
    # Base K20,20
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
            
    # Apply current modifications
    internal_edges = [(0, i) for i in range(1, 11)]
    for i in range(9):
        edges.discard((0, 20 + i))
        
    for u, v in internal_edges:
        edges.add((u, v))
        
    # Convert to adjacency structure for fast evaluation
    # adj[u] = set of v such that (u,v) in edges
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_max_book():
        max_book = 0
        for u, v in edges:
            # Triangles containing (u,v)
            # Common neighbors
            common = adj[u].intersection(adj[v])
            # Note: adj includes u and v? No, adj[u] contains neighbors.
            # If (u,v) is an edge, v is in adj[u] and u is in adj[v].
            # common will contain vertices w such that (u,w) and (v,w) are edges.
            # This counts triangles.
            if len(common) > max_book:
                max_book = len(common)
        return max_book
        
    current_max = get_max_book()
    
    # Local Search
    # Try to improve by swapping edges
    # We must keep exactly 401 edges.
    # Moves:
    # 1. Remove an internal edge, add a cross edge.
    # 2. Remove a cross edge, add an internal edge.
    # 3. Swap two cross edges? (No change in count, might change book)
    # 4. Swap internal and cross?
    
    # Since we are in a local minimum of the star structure,
    # we might need to try random restarts or different structures.
    
    # Let's try a simple random perturbation and hill climbing.
    best_edges = edges.copy()
    best_max = current_max
    
    for _ in range(1000): # Limit iterations
        # Pick a random edge to remove
        edge_list = list(best_edges)
        if not edge_list: break
        e_remove = random.choice(edge_list)
        
        # Pick a random edge to add (not in best_edges)
        # Generate a random non-existent edge
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if (u, v) in best_edges: continue
            if (v, u) in best_edges: continue
            e_add = (u, v)
            break
            
        # Apply move
        new_edges = best_edges - {e_remove, (e_remove[1], e_remove[0])}
        new_edges.add(e_add)
        
        # Evaluate
        adj_new = [set() for _ in range(n)]
        for u, v in new_edges:
            adj_new[u].add(v)
            adj_new[v].add(u)
            
        new_max = 0
        for u, v in new_edges:
            common = adj_new[u].intersection(adj_new[v])
            if len(common) > new_max:
                new_max = len(common)
                
        if new_max < best_max:
            best_max = new_max
            best_edges = new_edges
            
        # Optional: Accept worse moves with low probability (Simulated Annealing)
        # But for now, strict hill climbing.
        
    return [list(e) for e in best_edges]
