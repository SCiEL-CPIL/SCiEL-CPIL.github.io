import time
import random
import numpy as np

def construct():
    deadline = time.time() + 120  # Safety margin: 120s limit, use 120s budget
    n = 40
    m = 401
    
    # Start with K20,20 + 1 edge
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
    edges.add((0, 1))
    
    edge_list = list(edges)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    
    def compute_books():
        books = {}
        for u, v in edge_list:
            books[(u, v)] = len(adj[u] & adj[v])
            books[(v, u)] = books[(u, v)]
        return books
    
    books = compute_books()
    max_book = max(books.values())
    
    best_edges = set(edge_list)
    best_max = max_book
    
    # Local search: try to swap edges to reduce max book
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Find edges with high book
        high_edges = [(e, b) for e, b in books.items() if b >= max_book - 1 and e[0] < e[1]]
        if not high_edges:
            break
        
        # Pick a random high edge
        (u, v), b = random.choice(high_edges)
        
        # Try to find a non-edge to add and this edge to remove
        # such that max book doesn't increase
        found = False
        # Try removing (u,v) and adding a random non-edge
        for _ in range(20):
            # Pick a random non-edge
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                if (x, y) in best_edges or (y, x) in best_edges:
                    continue
                break
            
            # Check if (x,y) is the same as (u,v)
            if (x == u and y == v) or (x == v and y == u):
                continue
            
            # Simulate: remove (u,v), add (x,y)
            new_edges = best_edges - {(u, v), (v, u)}
            new_edges.add((x, y))
            if x > y:
                new_edges.add((y, x))
            else:
                new_edges.add((x, y))
            
            # Check if we have exactly m edges
            if len(new_edges) != m:
                continue
            
            # Compute new max book approximately
            # Only need to check edges affected by the change
            # Edges affected: (u,v) removed, (x,y) added, and edges incident to u,v,x,y
            
            # For efficiency, compute book for (x,y) and check if it's <= max_book
            # Also check that removing (u,v) doesn't increase others (it can only decrease)
            
            # Book of (x,y) = |N(x) ∩ N(y)| in the new graph
            # N(x) in new graph: adj[x] minus {v if v in adj[x]} plus {y if not already}
            # This is getting complex. Let me just compute the full book for the new graph.
            
            # Actually, let me just do a full recomputation since n=40 is small
            new_adj = [set() for _ in range(n)]
            for a, b in new_edges:
                new_adj[a].add(b)
                new_adj[b].add(a)
            
            new_max = 0
            for a, b in new_edges:
                if a < b:
                    bk = len(new_adj[a] & new_adj[b])
                    if bk > new_max:
                        new_max = bk
            
            if new_max < max_book:
                # Accept
                best_edges = new_edges
                edge_list = list(best_edges)
                adj = new_adj
                max_book = new_max
                books = {}
                for a, b in edge_list:
                    if a < b:
                        books[(a, b)] = len(adj[a] & adj[b])
                found = True
                break
            elif new_max == max_book and random.random() < 0.3:
                # Accept equal with some probability to escape local optima
                best_edges = new_edges
                edge_list = list(best_edges)
                adj = new_adj
                books = {}
                for a, b in edge_list:
                    if a < b:
                        books[(a, b)] = len(adj[a] & adj[b])
                found = True
                break
    
    # Return as list of lists
    result = [[u, v] for u, v in best_edges if u < v]
    # Ensure exactly 401 edges
    assert len(result) == 401
    return result
