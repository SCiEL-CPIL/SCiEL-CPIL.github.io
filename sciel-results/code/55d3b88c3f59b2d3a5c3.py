import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    n = 40
    target_edges = 401
    
    # Adjacency bitmasks
    adj = [0] * n
    
    # Set of edges (u, v) with u < v
    edges = set()
    
    # Dictionary: edge (u,v) -> book_size
    # We maintain this to quickly find max
    book_sizes = {}
    
    def add_edge(u, v):
        if u == v: return False
        if u > v: u, v = v, u
        if (u, v) in edges: return False
        edges.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(u, v):
        if u == v: return False
        if u > v: u, v = v, u
        if (u, v) not in edges: return False
        edges.remove((u, v))
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return True

    def calc_book(u, v):
        # u < v assumed or handled by bitmask symmetry
        common = adj[u] & adj[v]
        return common.bit_count()

    def update_books_for_vertex(v):
        # Recalculate book sizes for all edges incident to v
        # Edges are (v, w) where w in adj[v]
        # Note: book_size for (v,w) depends on common neighbors of v and w
        for w in range(n):
            if (1 << w) & adj[v]:
                u, x = (v, w) if v < w else (w, v)
                book_sizes[(u, x)] = calc_book(u, x)

    def initialize_random():
        # Start with a random graph with target_edges
        # To be efficient, we can start with K20,20 + 1 edge
        # Let's try K20,20 + 1 edge as it's a known good structure
        for i in range(20):
            for j in range(20, 40):
                add_edge(i, j)
        add_edge(0, 1)
        
        # Initialize book sizes
        for u, v in edges:
            book_sizes[(u, v)] = calc_book(u, v)

    def get_max_book():
        if not book_sizes: return 0
        return max(book_sizes.values())

    def get_max_edges():
        m = get_max_book()
        return [(u, v) for (u, v), s in book_sizes.items() if s == m]

    initialize_random()
    
    best_max = get_max_book()
    best_edges = set(edges)
    
    # Simulated Annealing Parameters
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    # Precompute all pairs for random selection
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        if T < T_min:
            T = 5.0 # Restart temperature
            
        current_max = get_max_book()
        
        # Find an edge to remove: preferably one with high book size
        max_edges = get_max_edges()
        if not max_edges:
            break
            
        # Randomly pick one of the max edges to remove
        u, v = random.choice(max_edges)
        
        # Find a candidate edge to add
        # We want an edge (x, y) not in graph such that book_size(x, y) is low
        # Sample random non-edges
        candidate = None
        candidate_book = 0
        
        # Try a few random candidates
        for _ in range(20):
            x, y = random.choice(all_pairs)
            if (x, y) in edges:
                continue
            
            # Calculate potential book size
            # Note: The graph currently has edge (u,v)
            # If we remove (u,v), the book size of (x,y) might change if x or y is u or v
            # For simplicity, let's just estimate based on current graph.
            # If (x,y) is disjoint from (u,v), book size is exact.
            # If they share a vertex, it's an approximation.
            
            # Better approach:
            # 1. Remove (u,v)
            # 2. Calculate book sizes for edges incident to u and v
            # 3. Find best (x,y) to add
            # 4. Add (x,y)
            # 5. Check max
            
            # This is expensive. Let's try a heuristic first.
            # Estimate book size of (x,y) in current graph
            pot_book = calc_book(x, y)
            
            # If pot_book is already > current_max, skip
            if pot_book > current_max:
                continue
            
            # If pot_book is low, it's a good candidate
            if candidate is None or pot_book < candidate_book:
                candidate = (x, y)
                candidate_book = pot_book
                
        if candidate is None:
            # Try harder or skip step
            continue
            
        x, y = candidate
        
        # Perform the move: Remove (u,v), Add (x,y)
        remove_edge(u, v)
        # Update books for u and v
        update_books_for_vertex(u)
        update_books_for_vertex(v)
        
        # Now add (x,y)
        add_edge(x, y)
        # Update books for x and y
        update_books_for_vertex(x)
        update_books_for_vertex(y)
        
        new_max = get_max_book()
        
        # Acceptance criterion
        if new_max < current_max:
            best_max = new_max
            best_edges = set(edges)
            T = max(T_min, T * 0.9) # Cool down
        elif new_max == current_max:
            # Accept sideways moves to explore
            T = max(T_min, T * 0.99)
        else:
            # Worse move
            # Accept with probability exp(-(new_max - current_max) / T)
            delta = new_max - current_max
            if random.random() < pow(2.718281828, -delta / T):
                T = max(T_min, T * 0.99)
            else:
                # Revert
                remove_edge(x, y)
                update_books_for_vertex(x)
                update_books_for_vertex(y)
                
                add_edge(u, v)
                update_books_for_vertex(u)
                update_books_for_vertex(v)
                
                T = max(T_min, T * 0.95)
                
    # Return best edges found
    return [[u, v] for u, v in best_edges]
