import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Key insight: We need max book size 7 (theoretical lower bound ceil(40/6) = 7)
    # Strategy: Start with a near-regular graph structure that naturally has small books,
    # then use local search to reduce max book to 7.
    
    # Better initial construction: Use a quasi-random approach
    # Start with a random graph with 401 edges, then optimize
    
    def random_graph(n, m):
        """Generate a random graph with n vertices and m edges."""
        edges = set()
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs[:m]:
            edges.add((u, v))
        return edges
    
    def build_adjacency(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_book_sizes(edges, adj, n):
        """Compute book size for each edge."""
        book_sizes = {}
        for u, v in edges:
            common = adj[u] & adj[v]
            book_sizes[(u, v)] = len(common)
        return book_sizes
    
    def get_max_book(book_sizes):
        if not book_sizes:
            return 0, []
        max_b = max(book_sizes.values())
        max_edges = [e for e, b in book_sizes.items() if b == max_b]
        return max_b, max_edges
    
    # Try multiple random restarts with local search
    best_result = None
    best_score = float('inf')
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial graph
        edges = random_graph(n, target_edges)
        adj = build_adjacency(edges, n)
        book_sizes = compute_book_sizes(edges, adj, n)
        current_max_book, current_max_edges = get_max_book(book_sizes)
        
        if current_max_book < best_score:
            best_score = current_max_book
            best_result = set(edges)
        
        if current_max_book <= 7:
            break
        
        # Local search: try to reduce max book size
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline - 2.0:
            if current_max_book <= 7:
                break
            
            if no_improve > max_no_improve:
                # Perturbation: make 3-5 random swaps
                for _ in range(random.randint(3, 5)):
                    # Remove a random edge
                    e = random.choice(list(edges))
                    u, v = e
                    edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                    # Add a random non-edge
                    while True:
                        x = random.randint(0, n - 1)
                        y = random.randint(0, n - 1)
                        if x == y:
                            continue
                        if x > y:
                            x, y = y, x
                        if (x, y) not in edges:
                            break
                    edges.add((x, y))
                    adj[x].add(y)
                    adj[y].add(x)
                
                # Recompute
                book_sizes = compute_book_sizes(edges, adj, n)
                current_max_book, current_max_edges = get_max_book(book_sizes)
                no_improve = 0
                max_no_improve += 50
                continue
            
            # Pick an edge with max book size to remove
            if not current_max_edges:
                break
            u, v = random.choice(current_max_edges)
            
            # Find best non-edge to add: minimize book size of new edge
            # and minimize increase in other edges' book sizes
            min_new_book = float('inf')
            best_pair = None
            best_total_increase = float('inf')
            
            # Sample non-edges for speed
            non_edges = []
            for x in range(n):
                for y in range(x + 1, n):
                    if (x, y) not in edges and (y, x) not in edges:
                        non_edges.append((x, y))
            
            if not non_edges:
                break
            
            # Evaluate a subset of non-edges for speed
            sample_size = min(len(non_edges), 50)
            sampled = random.sample(non_edges, sample_size)
            
            for x, y in sampled:
                new_book = len(adj[x] & adj[y])
                
                if new_book > current_max_book:
                    continue
                
                # Estimate impact: adding (x,y) increases book size of edges (x,z) and (y,z) 
                # for z in common neighbors
                common_xy = adj[x] & adj[y]
                
                # Count how many edges would exceed current_max_book
                violations = 0
                for z in common_xy:
                    # Edge (x,z) book size increases by 1 if y was not already common
                    # Actually, book size of (x,z) = |N(x) ∩ N(z)|
                    # Adding y to N(x) increases this if y ∈ N(z)
                    if y in adj[z]:
                        violations += 1
                    # Edge (y,z) book size increases by 1 if x was not already common
                    if x in adj[z]:
                        violations += 1
                
                # Prefer lower new_book and fewer violations
                score = new_book * 100 + violations
                if score < best_total_increase or (score == best_total_increase and new_book < min_new_book):
                    best_total_increase = score
                    min_new_book = new_book
                    best_pair = (x, y)
            
            if best_pair is None:
                no_improve += 1
                continue
            
            x, y = best_pair
            
            # Perform swap
            edges.remove((u, v))
            adj[u].remove(v)
            adj[v].remove(u)
            
            edges.add((x, y))
            adj[x].add(y)
            adj[y].add(x)
            
            # Recompute book sizes (full recompute for accuracy)
            book_sizes = compute_book_sizes(edges, adj, n)
            new_max, new_max_edges = get_max_book(book_sizes)
            
            # Acceptance
            accepted = False
            if new_max < current_max_book:
                accepted = True
                no_improve = 0
            elif new_max == current_max_book:
                if len(new_max_edges) < len(current_max_edges):
                    accepted = True
                    no_improve = 0
                elif random.random() < 0.1:
                    accepted = True
            else:
                accepted = False
            
            if accepted:
                current_max_book = new_max
                current_max_edges = new_max_edges
                if current_max_book < best_score:
                    best_score = current_max_book
                    best_result = set(edges)
            else:
                # Rollback
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
                edges.remove((x, y))
                adj[x].remove(y)
                adj[y].remove(x)
                
                no_improve += 1
        
        if best_score <= 7:
            break
    
    if best_result is None:
        # Fallback: return any valid graph
        best_result = random_graph(n, target_edges)
    
    return [[u, v] for u, v in best_result]
