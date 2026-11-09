import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = set()
    
    # Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    needed_in_B = target_edges - (a_size * b_size)  # 122
    
    # Initial construction of B edges
    def try_construct_B(max_deg, num_restarts=20):
        best_result = None
        best_max_book = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline - 30.0:
                break
                
            B_edges = []
            degree = {v: 0 for v in B}
            adj = {v: set() for v in B}
            
            pairs = list(itertools.combinations(B, 2))
            random.shuffle(pairs)
            
            for u, v in pairs:
                if len(B_edges) == needed_in_B:
                    break
                
                if adj[u] & adj[v]:
                    continue
                    
                if degree[u] < max_deg and degree[v] < max_deg:
                    B_edges.append((u, v))
                    degree[u] += 1
                    degree[v] += 1
                    adj[u].add(v)
                    adj[v].add(u)
            
            if len(B_edges) == needed_in_B:
                # Calculate max book size for this configuration
                # Book size for B-B edges is 9 + common_neighbors
                # Book size for A-B edges is deg_B(b)
                max_book = 0
                for u, v in B_edges:
                    common = len(adj[u] & adj[v])
                    max_book = max(max_book, 9 + common)
                for b in B:
                    max_book = max(max_book, degree[b])
                    
                if max_book < best_max_book:
                    best_max_book = max_book
                    best_result = B_edges.copy()
                    
        return best_result, best_max_book

    # Try to get max degree 8
    B_edges, current_score = try_construct_B(8, num_restarts=50)
    
    if B_edges is None:
        B_edges, current_score = try_construct_B(9, num_restarts=50)
        
    if B_edges is None:
        B_edges = []
        pairs = list(itertools.combinations(B, 2))
        random.shuffle(pairs)
        for u, v in pairs[:needed_in_B]:
            B_edges.append((u, v))
        current_score = 10 # Estimate

    # Add B edges to main set
    for u, v in B_edges:
        edges.add((u, v))
        
    # Local Search Refinement
    # We try to improve the score by swapping edges.
    # We represent the graph as a set of edges.
    # We calculate the book size for each edge.
    
    def calc_book_sizes(edge_set):
        adj = {v: set() for v in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        book_sizes = {}
        for u, v in edge_set:
            common = len(adj[u] & adj[v])
            book_sizes[(u, v)] = common
            book_sizes[(v, u)] = common
        return book_sizes, adj

    current_books, current_adj = calc_book_sizes(edges)
    max_book = max(current_books.values())
    
    # If max_book is already 7 or 8, we are good.
    # If it is 9, we try to reduce it.
    
    # Pre-compute candidate swaps
    # We will try to swap an edge e1 with an edge e2 not in the set.
    # e1 should be one of the edges with high book size.
    # e2 should be such that adding it doesn't increase the max book size too much.
    
    # For efficiency, we only consider swaps involving B-B edges or A-B edges.
    # Given the structure, swapping A-B edges is risky.
    # Let's try swapping B-B edges first.
    
    b_edges_list = [e for e in edges if e[0] in B and e[1] in B]
    ab_edges_list = [e for e in edges if (e[0] in A and e[1] in B) or (e[0] in B and e[1] in A)]
    
    # We will run a loop of local search steps
    steps = 0
    max_steps = 5000
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        steps += 1
        
        # Pick a random edge with high book size
        high_book_edges = [e for e in edges if current_books[e] >= max_book - 1]
        if not high_book_edges:
            break
            
        e1 = random.choice(high_book_edges)
        u1, v1 = e1
        
        # Generate a random candidate edge e2 not in edges
        # To keep it fast, we pick a random pair
        # We bias towards pairs that might reduce book size
        # For now, just random
        found_e2 = False
        for _ in range(10):
            u2, v2 = random.sample(range(n), 2)
            e2 = (min(u2, v2), max(u2, v2))
            if e2 in edges:
                continue
            if e2 == e1:
                continue
                
            # Check if e2 is a valid swap (doesn't create duplicate)
            # It is not in edges, so it's fine.
            
            # Simulate swap
            new_edges = set(edges)
            new_edges.remove(e1)
            new_edges.add(e2)
            
            # Calculate change in book sizes
            # Only edges incident to u1, v1, u2, v2 are affected.
            # This is complex to do incrementally without full recalc.
            # Given 40 vertices, full recalc is fast.
            
            new_books, _ = calc_book_sizes(new_edges)
            new_max_book = max(new_books.values())
            
            if new_max_book < max_book:
                # Accept
                edges = new_edges
                current_books = new_books
                max_book = new_max_book
                found_e2 = True
                break
            elif new_max_book == max_book:
                # Accept with some probability to explore
                if random.random() < 0.1:
                    edges = new_edges
                    current_books = new_books
                    found_e2 = True
                    break
                    
        if not found_e2:
            # Try a different e1 or break
            pass

    # Final result
    result = [[u, v] for u, v in edges]
    
    return result
