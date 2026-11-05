import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Partition vertices into 3 parts: 14, 13, 13
    part0 = list(range(14))
    part1 = list(range(14, 27))
    part2 = list(range(27, 40))
    
    best_score = float('inf')
    best_edges = None
    
    # Function to calculate max book size
    def get_max_book(edges):
        if not edges:
            return 0
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        max_book = 0
        for u, v in edges:
            # Count common neighbors
            count = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    count += 1
            if count > max_book:
                max_book = count
        return max_book

    # Greedy removal from T3
    def greedy_remove(seed_val):
        random.seed(seed_val)
        
        # Initialize T3 edges
        edges = set()
        for u in part0:
            for v in part1:
                edges.add((u, v))
            for v in part2:
                edges.add((u, v))
        for u in part1:
            for v in part2:
                edges.add((u, v))
                
        edge_list = list(edges)
        
        # Build adjacency matrix for fast lookup
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
            
        # Calculate initial book sizes
        book = [[0]*n for _ in range(n)]
        for u, v in edges:
            count = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    count += 1
            book[u][v] = count
            book[v][u] = count
            
        current_edges = len(edges)
        
        # We need to remove until we have target_edges
        while current_edges > target_edges:
            # Find edge with max book size
            max_b = -1
            candidates = []
            for u, v in edges:
                b = book[u][v]
                if b > max_b:
                    max_b = b
                    candidates = [(u, v)]
                elif b == max_b:
                    candidates.append((u, v))
            
            # Randomly pick one from candidates
            u, v = random.choice(candidates)
            
            # Remove edge (u, v)
            edges.remove((u, v))
            adj[u][v] = False
            adj[v][u] = False
            
            # Update book sizes for neighbors
            # Neighbors of (u, v) are edges (u, w) and (v, w) where w is common neighbor
            # Actually, any edge (u, w) or (v, w) that shares a triangle with (u, v)
            # Triangle (u, v, w) exists if adj[u][w] and adj[v][w]
            # Removing (u, v) reduces book[u][w] and book[v][w] by 1 for all such w
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    # (u, w) and (v, w) are in the graph
                    # Check if they are in edges set
                    if (u, w) in edges or (w, u) in edges:
                         # Find canonical form
                         e1 = (u, w) if u < w else (w, u)
                         if e1 in edges:
                             # Update book for e1
                             # book[e1[0]][e1[1]] -= 1
                             pass # We need to update the matrix correctly
                    # Simpler: just iterate through all edges incident to u or v?
                    # No, just update book[u][w] and book[v][w]
                    # But we need to ensure (u,w) and (v,w) are valid edges
                    if (u, w) in edges:
                        book[u][w] -= 1
                    elif (w, u) in edges:
                        book[w][u] -= 1
                    if (v, w) in edges:
                        book[v][w] -= 1
                    elif (w, v) in edges:
                        book[w][v] -= 1
                            
            current_edges -= 1
            
            # Time check
            if time.time() > deadline - 5.0:
                break
                
        # Convert to list of lists
        res = [[u, v] for u, v in edges]
        return res

    # Run multiple restarts
    iterations = 0
    while time.time() < deadline - 10.0:
        edges = greedy_remove(iterations)
        score = get_max_book(edges)
        if score < best_score:
            best_score = score
            best_edges = edges
            # If we hit the theoretical lower bound, we can stop
            if best_score <= 7:
                break
        iterations += 1
        
    if best_edges is None:
        # Fallback to seed
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return best_edges
