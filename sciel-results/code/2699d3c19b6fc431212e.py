import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s limit
    
    n = 40
    target_edges = 401
    
    def build_initial_graph(seed_val):
        """
        Builds a graph with 9-31 partition structure.
        Returns (edges_set, adj_matrix, max_book)
        """
        random.seed(seed_val)
        edges = set()
        
        A = list(range(9))
        B = list(range(9, 40))
        
        # 1. Add all edges between A and B (279 edges)
        for a in A:
            for b in B:
                edges.add((a, b))
                
        # 2. Add 122 edges within B
        B1 = list(range(9, 24))   # 15 vertices
        B2 = list(range(24, 40))  # 16 vertices
        
        needed = 122
        added = 0
        
        deg_b1 = {v: 0 for v in B1}
        deg_b2 = {v: 0 for v in B2}
        
        # Generate all possible pairs between B1 and B2
        pairs = [(u, v) for u in B1 for v in B2]
        random.shuffle(pairs)
        
        # Add edges with degree constraint to keep structure sparse
        max_deg = 9
        
        for u, v in pairs:
            if added == needed:
                break
            if deg_b1[u] < max_deg and deg_b2[v] < max_deg:
                edges.add((u, v))
                deg_b1[u] += 1
                deg_b2[v] += 1
                added += 1
                
        if added < needed:
            # Fallback: fill remaining ignoring constraints
            for u, v in pairs:
                if added == needed:
                    break
                if (u, v) not in edges:
                    edges.add((u, v))
                    added += 1

        # Build adjacency matrix
        adj = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        # Calculate max book
        max_b = 0
        for u, v in edges:
            b = sum(adj[u][w] & adj[v][w] for w in range(n))
            if b > max_b:
                max_b = b
                
        return edges, adj, max_b

    def compute_max_book(edges, adj):
        """Compute the maximum book size."""
        max_b = 0
        max_edges = []
        for u, v in edges:
            b = sum(adj[u][w] & adj[v][w] for w in range(n))
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    def local_search(edges, adj, max_book, max_edges, time_budget):
        """
        Perform local search to reduce max book size.
        Returns (edges, adj, max_book, max_edges)
        """
        current_max_book = max_book
        current_max_edges = max_edges
        no_improve_count = 0
        max_no_improve = 500
        
        # Pre-compute book sizes for all edges
        book_sizes = {}
        for u, v in edges:
            book_sizes[(u, v)] = sum(adj[u][w] & adj[v][w] for w in range(n))
        
        while time.time() < time_budget:
            if current_max_book <= 7:
                # 7 is the theoretical minimum (ceil(40/6) = 7)
                if current_max_book == 7:
                    break
                elif current_max_book <= 6:
                    break

            if not current_max_edges:
                break
            
            # Pick a random edge from the max-book set
            u, v = random.choice(current_max_edges)
            u, v = min(u, v), max(u, v)
            
            # Find a good non-edge to swap with
            # Heuristic: Look for non-edges with low book size potential
            best_swap = None
            best_score = float('inf')
            
            # Try a limited number of random non-edges for speed
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y or x > y:
                    continue
                if adj[x][y] == 1:
                    continue
                attempts += 1
                
                # Estimate the book size if we add (x,y)
                # This is the number of common neighbors of x and y
                common = sum(adj[x][w] & adj[y][w] for w in range(n))
                
                # We want to minimize the resulting max book
                # If common < current_max_book, it's likely to help
                score = common
                if score < best_score:
                    best_score = score
                    best_swap = (x, y)
                    if common <= 5:
                        break  # Good enough
            
            if best_swap is None:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                continue
            
            x, y = best_swap
            
            # Perform the swap
            edges.remove((u, v))
            adj[u][v] = 0
            adj[v][u] = 0
            
            edges.add((x, y))
            adj[x][y] = 1
            adj[y][x] = 1
            
            # Recompute max book (expensive but necessary for accuracy)
            new_max_book, new_max_edges = compute_max_book(edges, adj)
            
            accepted = False
            if new_max_book < current_max_book:
                accepted = True
                no_improve_count = 0
            elif new_max_book == current_max_book:
                if len(new_max_edges) < len(current_max_edges):
                    accepted = True
                    no_improve_count = 0
                else:
                    # Accept with small probability to escape plateaus
                    if random.random() < 0.03:
                        accepted = True
            else:
                accepted = False

            if accepted:
                current_max_book = new_max_book
                current_max_edges = new_max_edges
            else:
                # Rollback
                edges.add((u, v))
                adj[u][v] = 1
                adj[v][u] = 1
                
                edges.remove((x, y))
                adj[x][y] = 0
                adj[y][x] = 0
                
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        return edges, adj, current_max_book, current_max_edges

    # --- Multi-start with local search ---
    best_edges = None
    best_adj = None
    best_max_book = float('inf')
    best_max_edges = []
    
    # Try multiple seeds for the initial graph construction
    seeds = [42, 123, 999, 555, 777, 1234, 5678, 9012, 3456, 7890]
    
    for i, s in enumerate(seeds):
        if time.time() > deadline - 30:  # Leave time for final search
            break
        
        edges, adj, max_book = build_initial_graph(s)
        
        if max_book < best_max_book:
            best_max_book = max_book
            best_edges = edges.copy()
            best_adj = [row[:] for row in adj]
            _, best_max_edges = compute_max_book(edges, adj)
        
        # Perform local search on this instance
        if time.time() > deadline - 20:
            break
        
        search_budget = min(deadline - 10, time.time() + 10)
        edges, adj, new_max_book, new_max_edges = local_search(
            edges, adj, max_book, best_max_edges if max_book == best_max_book else [], 
            search_budget
        )
        
        if new_max_book < best_max_book:
            best_max_book = new_max_book
            best_edges = edges
            best_adj = adj
            best_max_edges = new_max_edges
            
            if best_max_book <= 7:
                # Can't do better than 7 (theoretical minimum)
                break
    
    # Final polish: if we have time, do another round of local search
    if time.time() < deadline - 5 and best_edges is not None:
        _, best_max_edges = compute_max_book(best_edges, best_adj)
        best_edges, best_adj, best_max_book, best_max_edges = local_search(
            best_edges, best_adj, best_max_book, best_max_edges, deadline - 2
        )
    
    # Return the result
    if best_edges is None:
        # Fallback: should never happen
        edges, adj, _ = build_initial_graph(0)
        best_edges = edges
    
    result = [[u, v] for u, v in best_edges]
    return result
