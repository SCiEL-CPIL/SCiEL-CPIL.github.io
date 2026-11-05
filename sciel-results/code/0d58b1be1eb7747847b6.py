import numpy as np
import time
import random

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    n = 40
    target_edges = 401
    
    # Adjacency matrix representation
    # adj[u,v] = 1 if edge exists
    adj = np.zeros((n, n), dtype=np.int8)
    
    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1
    
    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0
    
    def edge_count():
        return int(np.sum(adj) // 2)
    
    def compute_books():
        """Compute book size for each edge: |N(u) ∩ N(v)| for edge (u,v)."""
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    # Common neighbors
                    common = np.sum(adj[u] & adj[v])
                    books[(u, v)] = int(common)
        return books
    
    def max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    def objective(books):
        """Objective: primarily max book, secondarily sum of squares."""
        if not books:
            return 0
        vals = list(books.values())
        return max(vals) * 1000 + sum(v*v for v in vals)
    
    # Initialize: random graph with 401 edges
    random.seed(42)
    np.random.seed(42)
    
    # Start with a structured initialization: near-bipartite with some mixing
    # Partition into two groups of 20
    left = list(range(20))
    right = list(range(20, 40))
    
    # Start with K20,20
    for u in left:
        for v in right:
            add_edge(u, v)
    
    # Now we have 400 edges. Need 1 more.
    # Instead of just adding one, let's do a smarter initialization
    # Remove some cross edges and add intra edges to create a more spread structure
    
    # Actually, let's start fresh with a better initialization
    # Clear and rebuild
    adj[:] = 0
    
    # Better initialization: use a "blown-up" structure
    # Divide 40 vertices into groups, create a regular-ish graph
    # Target degree: 401*2/40 ≈ 20.05, so degree ~20
    
    # Start with a 20-regular graph on 40 vertices
    # Use a circulant-like construction
    # Vertex i connected to i±1, i±2, ..., i±10 (mod 40) -> that's 20 edges per vertex
    for i in range(n):
        for d in range(1, 11):
            j = (i + d) % n
            if i < j:
                add_edge(i, j)
    
    # Check edge count
    ec = edge_count()
    # 40 vertices, each with degree 20, so 400 edges
    # Need 1 more
    if ec < target_edges:
        # Add a random non-existing edge
        while edge_count() < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not adj[u, v]:
                add_edge(u, v)
    elif ec > target_edges:
        while edge_count() > target_edges:
            # Remove a random edge
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
            u, v = random.choice(edges)
            remove_edge(u, v)
    
    # Compute initial books
    books = compute_books()
    best_books = {k: v for k, v in books.items()}
    best_obj = objective(best_books)
    best_adj = adj.copy()
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    start_time = time.time()
    step = 0
    
    while T > T_min and time.time() < deadline:
        step += 1
        
        # Move: pick a random edge and try to swap it
        # Find all edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
        
        if len(edges) != target_edges:
            # Shouldn't happen, but fix
            break
        
        # Pick a random edge to remove
        u1, v1 = random.choice(edges)
        
        # Try to find a new edge (u2, v2) to add
        # u2, v2 should not form an existing edge, and (u2,v2) != (u1,v1)
        # We want to try swapping: remove (u1,v1), add (u2,v2)
        # For efficiency, pick u2 from neighbors of u1 or v1, and v2 randomly
        
        # Strategy: pick u2 as a random vertex, v2 as a random vertex
        # But ensure (u2,v2) is not an edge and (u2,v2) != (u1,v1)
        
        # Try a few candidates
        found = False
        for _ in range(20):
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) == (u1, v1):
                continue
            if adj[u2, v2]:
                continue
            # This is a valid swap candidate
            found = True
            break
        
        if not found:
            continue
        
        # Compute change in objective
        # Remove (u1,v1), add (u2,v2)
        # This affects books of edges adjacent to u1, v1, u2, v2
        
        # For speed, compute books only for affected edges
        # Affected edges: those incident to u1, v1, u2, v2
        
        affected = set()
        for x in [u1, v1, u2, v2]:
            for y in range(n):
                if adj[x, y] and x < y:
                    affected.add((x, y))
                elif adj[x, y] and x > y:
                    affected.add((y, x))
        # Also (u2,v2) will be added
        affected.add((u2, v2))
        
        # Compute old books for affected edges
        old_books = {}
        for e in affected:
            if e in books:
                old_books[e] = books[e]
        
        # Apply the swap
        remove_edge(u1, v1)
        add_edge(u2, v2)
        
        # Compute new books for affected edges
        new_books = {}
        for e in affected:
            a, b = e
            if adj[a, b]:
                common = int(np.sum(adj[a] & adj[b]))
                new_books[e] = common
        
        # Update books dict
        for e in affected:
            if e in books:
                del books[e]
        for e, val in new_books.items():
            books[e] = val
        
        new_obj = objective(books)
        delta_obj = new_obj - best_obj
        
        # Accept or reject
        if delta_obj < 0 or random.random() < np.exp(-delta_obj / T):
            # Accept
            best_obj = new_obj
            if new_obj < objective(best_books):
                best_books = {k: v for k, v in books.items()}
                best_adj = adj.copy()
        else:
            # Reject: undo the swap
            remove_edge(u2, v2)
            add_edge(u1, v1)
            
            # Restore books
            for e in affected:
                if e in books:
                    del books[e]
            for e, val in old_books.items():
                books[e] = val
        
        # Cool down
        T *= cooling
        
        # Periodic check
        if step % 1000 == 0:
            if time.time() > deadline - 5.0:
                break
    
    # Restore best
    adj = best_adj
    
    # Extract edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                result.append([i, j])
    
    return result
