import random
import math

def construct():
    n = 40
    target_edges = 401
    
    def compute_books(edge_set):
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        books = {}
        for u, v in edge_set:
            books[(u, v)] = len(adj[u] & adj[v])
        return books

    def get_max_book(books):
        if not books:
            return 0
        return max(books.values())

    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())

    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Initialize with a random graph
    rng = random.Random(42)
    idx = list(range(len(all_possible)))
    rng.shuffle(idx)
    current_edges = set(all_possible[i] for i in idx[:target_edges])
    current_books = compute_books(current_edges)
    current_cost = get_sum_sq_books(current_books)
    
    best_edges = set(current_edges)
    best_cost = current_cost
    best_max_book = get_max_book(current_books)
    
    # Simulated Annealing
    T = 1000.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute adjacency for speed
    def update_books_after_swap(edge_set, remove_edge, add_edge):
        # This is complex to do incrementally correctly without full recompute or careful tracking.
        # Given n=40, full recompute is O(E * N) ~ 400 * 40 = 16000 ops.
        # In Python, 10000 iterations * 16000 ops = 160M ops. Might be slow.
        # Let's stick to full recompute but optimize the inner loop.
        pass

    # Optimized recompute
    def compute_books_fast(edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        books = {}
        for u, v in edge_list:
            books[(u, v)] = len(adj[u] & adj[v])
        return books

    for i in range(50000):
        if T < min_T:
            break
            
        # Pick a random edge to remove
        u, v = next(iter(current_edges)) # Random pick is better
        # Actually, pick random edge from list for uniformity
        edge_list = list(current_edges)
        rem_u, rem_v = edge_list[rng.randrange(len(edge_list))]
        
        # Pick a random edge to add
        add_u = rng.randrange(n)
        add_v = rng.randrange(n)
        if add_u == add_v:
            continue
        add_u, add_v = min(add_u, add_v), max(add_u, add_v)
        if (add_u, add_v) in current_edges:
            continue
            
        new_edges = current_edges.copy()
        new_edges.discard((rem_u, rem_v))
        new_edges.add((add_u, add_v))
        
        new_books = compute_books_fast(list(new_edges))
        new_cost = get_sum_sq_books(new_books)
        new_max_book = get_max_book(new_books)
        
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_edges = new_edges
            current_books = new_books
            current_cost = new_cost
            
            if new_max_book < best_max_book or (new_max_book == best_max_book and new_cost < best_cost):
                best_max_book = new_max_book
                best_cost = new_cost
                best_edges = new_edges
                
                # If we hit the theoretical lower bound, we can stop or keep polishing
                if best_max_book <= 7:
                    # Keep running to minimize cost among 7s, or just break if we want speed
                    # But let's try to find 6 (impossible) or just settle for 7
                    pass
        
        T *= cooling

    # Final local search on best_edges to ensure no easy improvements
    # ... (Optional, SA should handle it)

    return [[u, v] for u, v in best_edges]
