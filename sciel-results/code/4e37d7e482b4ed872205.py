import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initialize with a random graph
    edges = set(random.sample(all_edges, target_edges))
    
    def compute_books(edge_set):
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=int)
        for u, v in edge_set:
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Compute common neighbors for all pairs
        # common[u, v] = number of common neighbors of u and v
        # This is the element (u, v) of adj @ adj
        common = adj @ adj
        
        # Book size for edge (u, v) is common[u, v]
        # We only care about edges that exist
        max_book = 0
        edge_books = {}
        for u, v in edge_set:
            b = int(common[u, v])
            edge_books[(u, v)] = b
            if b > max_book:
                max_book = b
        return max_book, edge_books, adj, common

    # Initial evaluation
    max_book, edge_books, adj, common = compute_books(edges)
    
    # Local search
    # We try to reduce the maximum book size
    # Strategy: Find the edge with the max book. Try to break one of its triangles by swapping an edge.
    
    no_improve = 0
    max_iterations = 5000
    
    for _ in range(max_iterations):
        if no_improve > 100:
            break
            
        # Find edge with max book
        current_max = max(edge_books.values())
        
        # Get list of edges with max book
        max_edges = [e for e, b in edge_books.items() if b == current_max]
        
        if not max_edges:
            break
            
        # Pick a random max edge
        u, v = random.choice(max_edges)
        
        # Common neighbors of u and v
        # These are w such that (u,w) and (v,w) are edges
        # We want to remove one of these edges to reduce the book size of (u,v)
        # Candidates for removal: (u, w) for w in common_neighbors(u, v)
        
        # Find common neighbors
        common_neighbors = []
        for w in range(n):
            if w != u and w != v and (u, w) in edges and (v, w) in edges:
                common_neighbors.append(w)
                
        if not common_neighbors:
            continue
            
        # Try removing an edge (u, w) or (v, w)
        # Let's try removing (u, w)
        w = random.choice(common_neighbors)
        
        # We need to add a new edge to maintain the count
        # The new edge should not create a book larger than current_max
        # Ideally, it should be an edge with low book size potential
        
        # Candidate new edges: non-edges (x, y)
        # To be efficient, we can just pick a random non-edge and check
        # But checking all non-edges is expensive. 
        # Let's pick a few random non-edges
        
        # Generate a set of random non-edges to try
        # Total non-edges = 780 - 401 = 379
        # We can just iterate a few random ones
        
        improved = False
        for _ in range(20): # Try 20 random swaps
            # Pick a random non-edge
            while True:
                x, y = random.sample(range(n), 2)
                if (x, y) not in edges and (y, x) not in edges:
                    break
            
            # Proposed new edge set
            new_edges = set(edges)
            new_edges.remove((u, w))
            new_edges.add((x, y))
            
            # Check if (x, y) is valid (distinct, etc.) - yes by construction
            # Evaluate new max book
            # To avoid full recomputation, we can approximate or do full if n is small
            # n=40 is small, full recomputation is fast
            
            new_max, _, _, _ = compute_books(new_edges)
            
            if new_max < current_max:
                edges = new_edges
                max_book = new_max
                no_improve = 0
                improved = True
                break
            elif new_max == current_max:
                # Accept with some probability to escape local minima? 
                # Or just accept if it doesn't get worse?
                # Let's accept if it's the same to explore the space
                if random.random() < 0.1:
                    edges = new_edges
                    # Recompute books for next iteration
                    max_book, edge_books, adj, common = compute_books(edges)
                    no_improve = 0
                    improved = True
                    break
            # If new_max > current_max, reject
            
        if not improved:
            no_improve += 1
        else:
            # Update books
            max_book, edge_books, adj, common = compute_books(edges)
            # If the max book decreased, we are good.
            # If we accepted a same-max swap, we need to update edge_books
            # The 'improved' block above handles the update if we break.
            # But if we didn't break (e.g. found a better one), we updated edges.
            # We need to ensure edge_books is consistent with edges.
            # The 'if new_max < current_max' branch updates edges but not edge_books immediately.
            # The 'if new_max == current_max' branch updates edge_books.
            # Let's just recompute at the end of the loop if edges changed?
            # Actually, the 'improved' flag logic is a bit messy. Let's simplify.
            
            # Re-compute to be safe if edges changed
            if improved:
                 max_book, edge_books, adj, common = compute_books(edges)

    return [list(e) for e in edges]
