import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 40
    m = 401
    random.seed(42)
    
    # Structural Initialization:
    # K20,20 minus r edges from 0 to B, plus k internal star edges from 0.
    # r + k = 21.
    # To minimize max(r, k), choose r=10, k=11 or r=11, k=10.
    # Let's use r=10, k=11.
    # Remove 10 edges from 0 to B: (0, 20)..(0, 29).
    # Add 11 internal edges: (0, 1)..(0, 11).
    
    set_a = list(range(20))
    set_b = list(range(20, 40))
    
    edges = set()
    adj = [0] * n
    
    # Add all bipartite edges except those removed from 0
    for u in set_a:
        for v in set_b:
            if u == 0 and 20 <= v <= 29:
                continue
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    # Add internal star edges
    for i in range(1, 12):
        edges.add((0, i))
        adj[0] |= (1 << i)
        adj[i] |= (1 << 0)
        
    # Verify count
    assert len(edges) == m, f"Expected {m} edges, got {len(edges)}"
    
    def compute_books():
        books = {}
        for u, v in edges:
            common = adj[u] & adj[v]
            books[(u, v)] = common.bit_count()
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    books = compute_books()
    current_max = get_max_book(books)
    
    # Local Search to try to reduce max book
    # Focus on edges with high book sizes
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    T = 1.0
    T_min = 0.1
    cooling = 0.999
    step = 0
    max_steps = 5000
    
    while step < max_steps and time.time() < deadline - 5.0 and T > T_min:
        # Find edges with high book size
        high_edges = [e for e, b in books.items() if b >= current_max]
        if not high_edges:
            break
            
        # Pick a random high edge to remove
        e_remove = random.choice(high_edges)
        u, v = e_remove
        
        # Try to find a replacement edge that reduces max book
        best_cand = None
        best_new_max = current_max
        best_new_sum_sq = sum(b*b for b in books.values())
        
        # Sample candidates
        for _ in range(50):
            a, b = random.choice(all_possible)
            a, b = min(a, b), max(a, b)
            if (a, b) in edges or (a, b) == e_remove:
                continue
            
            # Apply move temporarily
            edges.remove(e_remove)
            edges.add((a, b))
            
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
            # Recompute books for affected edges
            # Affected vertices: u, v, a, b
            affected = {u, v, a, b}
            new_books = {}
            for e in edges:
                x, y = e
                if x in affected or y in affected:
                    new_books[e] = (adj[x] & adj[y]).bit_count()
            
            # Update global books
            # This is O(E) if done naively, but we only care about max
            # Let's just compute max book for the whole graph for simplicity in this loop
            # Since n=40, E=401, it's fast enough.
            new_max = 0
            new_sum_sq = 0
            for e in edges:
                x, y = e
                b_val = (adj[x] & adj[y]).bit_count()
                if b_val > new_max:
                    new_max = b_val
                new_sum_sq += b_val * b_val
            
            # Revert
            edges.remove((a, b))
            edges.add(e_remove)
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            
            if new_max < best_new_max or (new_max == best_new_max and new_sum_sq < best_new_sum_sq):
                best_new_max = new_max
                best_new_sum_sq = new_sum_sq
                best_cand = (a, b)
                
        if best_cand is None:
            T *= cooling
            step += 1
            continue
            
        a, b = best_cand
        # Apply best move
        edges.remove(e_remove)
        edges.add((a, b))
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        
        # Recompute books fully to update state
        books = compute_books()
        current_max = get_max_book(books)
        
        T *= cooling
        step += 1
        
        if current_max == 7:
            break

    return [list(e) for e in edges]
