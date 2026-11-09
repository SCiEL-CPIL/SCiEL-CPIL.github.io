import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    # a=9, b=31 is optimal for this family, yielding max book size 9.
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = set()
    
    # 1. Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    needed_in_B = target_edges - (a_size * b_size)  # 122
    
    def try_construct_B(max_deg, seed):
        rng = random.Random(seed)
        B_edges = []
        degree = {v: 0 for v in B}
        adj = {v: set() for v in B}
        
        # Generate pairs and shuffle
        pairs = list(itertools.combinations(B, 2))
        rng.shuffle(pairs)
        
        for u, v in pairs:
            if len(B_edges) == needed_in_B:
                break
            
            # Check triangle constraint: no common neighbor in B
            if adj[u] & adj[v]:
                continue
                
            # Check degree constraint
            if degree[u] < max_deg and degree[v] < max_deg:
                B_edges.append((u, v))
                degree[u] += 1
                degree[v] += 1
                adj[u].add(v)
                adj[v].add(u)
                
        if len(B_edges) == needed_in_B:
            return B_edges
        return None

    def compute_max_book(B_edges):
        """Compute the maximum book size for the given B edges."""
        adj_B = {v: set() for v in B}
        for u, v in B_edges:
            adj_B[u].add(v)
            adj_B[v].add(u)
        
        max_book = 0
        # Cross edges: book = degree in B of the B-endpoint
        for v in B:
            max_book = max(max_book, len(adj_B[v]))
        
        # B-edges: book = a_size + |common neighbors in B|
        for u, v in B_edges:
            common = len(adj_B[u] & adj_B[v])
            max_book = max(max_book, a_size + common)
        
        return max_book

    B_edges = None
    best_B_edges = None
    best_book = float('inf')
    
    # Try to find a solution with max degree 8
    attempts = 0
    max_attempts = 200
    while time.time() < deadline - 15.0 and attempts < max_attempts:
        B_edges = try_construct_B(8, seed=attempts)
        if B_edges:
            book = compute_max_book(B_edges)
            if book < best_book:
                best_book = book
                best_B_edges = B_edges
            if best_book <= 9:
                break
        attempts += 1
    
    # If we found a good solution, use it
    if best_B_edges is not None:
        B_edges = best_B_edges
    else:
        # Fallback: try max degree 9
        B_edges = try_construct_B(9, seed=12345)
        if B_edges:
            best_book = compute_max_book(B_edges)
        
        if not B_edges:
            # Last resort
            pairs = list(itertools.combinations(B, 2))
            random.shuffle(pairs)
            B_edges = pairs[:needed_in_B]
            best_book = compute_max_book(B_edges)

    for u, v in B_edges:
        edges.add((u, v))
        
    result = [[u, v] for u, v in edges]
    
    return result
