import time
import random

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    n = 40
    target_edges = 401
    
    # Initialize with K20,20 + one intra-partition edge
    edges = set()
    adj = [0] * n  # bitmask adjacency
    
    def add_edge(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False
    
    def remove_edge(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            edges.remove((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return True
        return False
    
    def count_common_neighbors(u, v):
        # Number of vertices w such that (u,w) and (v,w) are both edges
        common = adj[u] & adj[v]
        return bin(common).count('1')
    
    def get_books():
        # Compute book size for each edge
        books = {}
        for (u, v) in edges:
            books[(u, v)] = count_common_neighbors(u, v)
        return books
    
    def get_max_book():
        max_b = 0
        for (u, v) in edges:
            b = count_common_neighbors(u, v)
            if b > max_b:
                max_b = b
        return max_b
    
    # Initialize: K20,20 + edge (0,1)
    for a in range(20):
        for b in range(20, 40):
            add_edge(a, b)
    add_edge(0, 1)
    
    current_max = get_max_book()
    best_max = current_max
    best_edges = set(edges)
    
    # Local search: try edge swaps to reduce max book
    # A swap: remove edge e1, add edge e2 (must keep exactly 401 edges)
    # Focus on edges with high book sizes
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        # Get current books
        books = get_books()
        current_max = max(books.values()) if books else 0
        
        if current_max < best_max:
            best_max = current_max
            best_edges = set(edges)
        
        # If we've achieved a good score, try to improve further
        # Find edges with book size == current_max
        high_book_edges = [(u, v) for (u, v), b in books.items() if b == current_max]
        
        if not high_book_edges:
            break
        
        # Pick a random high-book edge to try to reduce
        u, v = rng.choice(high_book_edges)
        
        # Try to find a swap that reduces the max book
        # Remove (u,v) and add some other edge
        # The new edge should not create a larger book
        
        # Candidate edges to add: any non-edge
        # To be efficient, only consider edges that might help
        
        # First, let's try removing (u,v) and see what happens
        # Then add an edge that doesn't create a large book
        
        # Actually, let's try: for each possible new edge, check if the swap improves
        # But that's O(n^2) per iteration. Let's sample.
        
        improved = False
        for _ in range(100):  # Try up to 100 random swaps
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random non-edge to add
            while True:
                ru = rng.randint(0, n-1)
                rv = rng.randint(0, n-1)
                if ru == rv:
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                if (ru, rv) not in edges:
                    break
            
            # Check: if we remove (u,v) and add (ru,rv), does max book decrease?
            # First, temporarily remove (u,v)
            remove_edge(u, v)
            add_edge(ru, rv)
            
            new_max = get_max_book()
            
            if new_max < current_max:
                # Accept the swap
                current_max = new_max
                improved = True
                break
            else:
                # Revert
                remove_edge(ru, rv)
                add_edge(u, v)
        
        if not improved:
            # Try a different approach: remove a cross edge that contributes to a book,
            # and add an intra-partition edge in a low-degree area
            pass
    
    # Return best result
    result = []
    for (u, v) in best_edges:
        result.append([u, v])
    
    # Ensure exactly 401 edges
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    # Ensure all vertices are in range
    for e in result:
        assert 0 <= e[0] <= 39 and 0 <= e[1] <= 39
    
    return result
