import numpy as np
import time
import random

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    n = 40
    target_edges = 401
    
    # Bitmask representation for fast intersection
    # adj_mask[i] is a 40-bit integer where bit j is set if edge (i,j) exists
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def edge_count():
        return sum(bin(m).count('1') for m in adj_mask) // 2
    
    def compute_books():
        """Compute book size for each edge using bitmask intersection."""
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    books[(u, v)] = bin(common).count('1')
        return books
    
    def objective(books):
        """Objective: primarily max book, secondarily sum of squares."""
        if not books:
            return 0
        vals = list(books.values())
        return max(vals) * 1000 + sum(v*v for v in vals)
    
    # Initialize: random graph with 401 edges
    random.seed(42)
    np.random.seed(42)
    
    # Start with a 20-regular circulant graph (400 edges)
    # Vertex i connected to i±1, i±2, ..., i±10 (mod 40)
    for i in range(n):
        for d in range(1, 11):
            j = (i + d) % n
            if i < j:
                add_edge(i, j)
    
    # Check edge count
    ec = edge_count()
    if ec < target_edges:
        while edge_count() < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not ((adj_mask[u] >> v) & 1):
                add_edge(u, v)
    elif ec > target_edges:
        while edge_count() > target_edges:
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj_mask[i] >> j) & 1]
            u, v = random.choice(edges)
            remove_edge(u, v)
    
    # Compute initial books
    books = compute_books()
    best_books = {k: v for k, v in books.items()}
    best_obj = objective(best_books)
    best_adj_mask = adj_mask[:]
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    start_time = time.time()
    step = 0
    
    # Precompute all possible edges for random selection
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while T > T_min and time.time() < deadline:
        step += 1
        
        # Move: pick a random edge and try to swap it
        # Find all current edges
        current_edges = []
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    current_edges.append((u, v))
        
        if len(current_edges) != target_edges:
            break
        
        # Pick a random edge to remove
        u1, v1 = random.choice(current_edges)
        
        # Try to find a new edge (u2, v2) to add
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
            if (adj_mask[u2] >> v2) & 1:
                continue
            found = True
            break
        
        if not found:
            continue
        
        # Compute change in objective
        # Remove (u1,v1), add (u2,v2)
        # Affected edges: those incident to u1, v1, u2, v2
        
        affected = set()
        for x in [u1, v1, u2, v2]:
            mask = adj_mask[x]
            while mask:
                y = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if x < y:
                    affected.add((x, y))
                else:
                    affected.add((y, x))
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
            if (adj_mask[a] >> b) & 1:
                common = adj_mask[a] & adj_mask[b]
                new_books[e] = bin(common).count('1')
        
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
                best_adj_mask = adj_mask[:]
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
    adj_mask = best_adj_mask
    
    # Extract edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_mask[i] >> j) & 1:
                result.append([i, j])
    
    return result
