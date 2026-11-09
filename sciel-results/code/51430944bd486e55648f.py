import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    n = 40
    target_edges = 401
    
    # Initialize with K20,20 + (0,1)
    # Vertices 0..19 are A, 20..39 are B
    adj = [set() for _ in range(n)]
    edges = []
    
    # Add K20,20
    for a in range(20):
        for b in range(20, 40):
            adj[a].add(b)
            adj[b].add(a)
            edges.append((a, b))
            
    # Add (0,1)
    adj[0].add(1)
    adj[1].add(0)
    edges.append((0, 1))
    
    # Function to compute book sizes
    # book_size[u, v] = len(adj[u] & adj[v]) for u < v
    # We only care about edges in the graph.
    
    def get_book_sizes():
        books = {}
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Only count if (u,v) is an edge? 
                    # The task says "maximum over edges of the number of triangles containing that edge".
                    # So we only evaluate book sizes for edges that exist.
                    books[(u, v)] = len(adj[u] & adj[v])
        return books

    current_books = get_book_sizes()
    max_book = max(current_books.values())
    best_max_book = max_book
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    
    # Local Search
    # We will try to swap edges to reduce the max book size.
    # A swap: remove (u, x), add (u, y).
    # Constraints:
    # 1. (u, x) must be an edge.
    # 2. (u, y) must NOT be an edge.
    # 3. Graph must remain simple (no self loops, no multi edges).
    # 4. Ideally, we don't disconnect the graph, but connectivity is not strictly required by the prompt, 
    #    but "Graphs with 40 vertices" usually implies simple graphs. 
    #    The prompt doesn't explicitly require connectivity, but it's good practice.
    #    However, minimizing max book size might lead to disconnected components.
    #    Let's assume simple graph is enough.
    
    # To speed up, we identify "hot" edges (high book size) and try to fix them.
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Helper to check validity of swap
    def is_valid_swap(u, x, y):
        if x == y:
            return False
        if (u, x) not in [(a, b) for a in adj[u] if a < b] and (x, u) not in [(a, b) for a in adj[x] if a < b]:
             # (u,x) is not an edge
             return False
        if y in adj[u]:
            return False
        return True

    # More efficient edge check using sets
    def has_edge(u, v):
        return v in adj[u]

    # Main loop
    while time.time() < deadline:
        # Find the edge with the maximum book size
        max_edge = None
        max_val = -1
        for e, b in current_books.items():
            if b > max_val:
                max_val = b
                max_edge = e
        
        if max_val == best_max_book:
            # If we can't beat the best, we might want to try random moves to escape
            pass
        else:
            # Update best
            best_max_book = max_val
            best_edges = edges.copy()
            best_adj = [s.copy() for s in adj]
            
        u, v = max_edge
        
        # Try to reduce book size of (u, v)
        # The book size is |N(u) & N(v)|.
        # We can try to move a neighbor of u away, or a neighbor of v away.
        # Let's try to swap an edge incident to u.
        # Remove (u, x), add (u, y).
        
        # Candidates for x: neighbors of u (excluding v)
        # Candidates for y: non-neighbors of u
        
        # To be efficient, pick x that is in N(u) & N(v) (contributes to the book).
        # If we remove (u, x), x is no longer in N(u), so it won't be in N(u) & N(v) anymore.
        # This reduces the book size of (u, v) by 1 (if x was in the intersection).
        
        common_neighbors_uv = adj[u] & adj[v]
        
        if not common_neighbors_uv:
            # Book size is already 0, try another edge?
            # If max book is 0, we are done.
            if max_val == 0:
                break
            continue

        # Pick a random common neighbor x to remove
        x = random.choice(list(common_neighbors_uv))
        
        # Pick a random non-neighbor y to add
        # Non-neighbors of u: all vertices except u and adj[u]
        non_neighbors = [i for i in range(n) if i != u and i not in adj[u]]
        if not non_neighbors:
            continue
        y = random.choice(non_neighbors)
        
        # Check if swap is valid
        # (u, x) is an edge (by construction)
        # (u, y) is not an edge (by construction)
        # We must ensure we don't create a multi-edge or self-loop (handled by y != u and y not in adj[u])
        
        # Perform swap
        adj[u].remove(x)
        adj[x].remove(u)
        adj[u].add(y)
        adj[y].add(u)
        
        # Update edges list
        # Remove (u, x) or (x, u)
        # Add (u, y) or (y, u)
        # For simplicity, we can just recompute books, but it's expensive.
        # Let's try to update incrementally or just recompute if n is small.
        # n=40, edges ~400. Recomputing books takes 400 intersections of sets of size ~20.
        # 400 * 20 = 8000 ops. Very fast.
        
        # Let's just recompute to be safe and simple.
        current_books = get_book_sizes()
        new_max = max(current_books.values())
        
        if new_max < max_val:
            # Accept
            edges.remove((u, x) if (u, x) in edges else (x, u))
            edges.append((u, y) if u < y else (y, u))
            # Update best if better
            if new_max < best_max_book:
                best_max_book = new_max
                best_edges = edges.copy()
                best_adj = [s.copy() for s in adj]
        else:
            # Reject, undo swap
            adj[u].remove(y)
            adj[y].remove(u)
            adj[u].add(x)
            adj[x].add(u)
            edges.remove((u, y) if (u, y) in edges else (y, u))
            edges.append((u, x) if u < x else (x, u))
            
            # If we are stuck, maybe try a random move (Simulated Annealing)
            # But for now, let's just try another x, y
            # To avoid infinite loop on same bad move, we can shuffle candidates
            
    # Return the best found
    # Convert best_edges to list of lists
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
