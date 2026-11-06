"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n = 40
    target_edges = 401
    
    def compute_books(adj):
        """Compute book size for each edge. adj is n x n 0/1 matrix."""
        # For edge (u,v), book size = number of common neighbors
        # common_neighbors(u,v) = sum_w adj[u,w]*adj[v,w]
        # = (adj @ adj)[u,v] for u != v
        A = adj
        A2 = A @ A
        books = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 1:
                    books[u, v] = int(A2[u, v])
        return books
    
    def get_max_book(adj):
        books = compute_books(adj)
        return int(books.max())
    
    def edge_count(adj):
        return int(adj.sum() // 2)
    
    # Start with K20,20 + 1 edge (the seed)
    adj = np.zeros((n, n), dtype=np.float64)
    for a in range(20):
        for b in range(20, 40):
            adj[a, b] = 1
            adj[b, a] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    current_adj = adj.copy()
    current_max_book = get_max_book(current_adj)
    best_adj = current_adj.copy()
    best_max_book = current_max_book
    
    # Local search: try edge swaps to reduce max book
    def try_improve(adj, max_book, time_limit):
        """Try local search to reduce max book."""
        adj = adj.copy()
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Get current books
            A2 = adj @ adj
            books = np.zeros((n, n), dtype=int)
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v] > 0.5:
                        books[u, v] = int(round(A2[u, v]))
            
            # Find edges with high book size
            high_book_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v] > 0.5 and books[u, v] >= max_book:
                        high_book_edges.append((u, v, books[u, v]))
            
            # Try removing high-book edges and adding low-book edges
            # First, find candidate edges to remove
            candidates_remove = [e for e in high_book_edges if e[2] >= max_book]
            
            # Find candidate edges to add (non-edges)
            candidates_add = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v] < 0.5:
                        # Estimate book size if added
                        est_book = int(round(A2[u, v]))
                        candidates_add.append((u, v, est_book))
            
            # Sort by book size
            candidates_remove.sort(key=lambda x: -x[2])
            candidates_add.sort(key=lambda x: x[2])
            
            # Try swaps
            for ru, rv, rb in candidates_remove[:50]:
                if (time.time() - start_time) > time_limit:
                    break
                for au, av, ab in candidates_add[:100]:
                    # Don't add same edge
                    if (ru, rv) == (au, av) or (ru, rv) == (av, au):
                        continue
                    # Check if adding would keep book size <= current max
                    if ab > max_book:
                        continue
                    
                    # Try the swap
                    adj[ru, rv] = 0
                    adj[rv, ru] = 0
                    adj[au, av] = 1
                    adj[av, au] = 1
                    
                    new_books = compute_books(adj)
                    new_max = int(new_books.max())
                    
                    if new_max < max_book:
                        max_book = new_max
                        improved = True
                        break
                    else:
                        # Revert
                        adj[ru, rv] = 1
                        adj[rv, ru] = 1
                        adj[au, av] = 0
                        adj[av, au] = 0
            
            if improved:
                continue
            
            # If no swap improved, try single removals (if we have extra edges)
            if edge_count(adj) > target_edges:
                for ru, rv, rb in candidates_remove[:30]:
                    if (time.time() - start_time) > time_limit:
                        break
                    adj[ru, rv] = 0
                    adj[rv, ru] = 0
                    new_books = compute_books(adj)
                    new_max = int(new_books.max())
                    if new_max < max_book:
                        max_book = new_max
                        improved = True
                    else:
                        adj[ru, rv] = 1
                        adj[rv, ru] = 1
            elif edge_count(adj) < target_edges:
                for au, av, ab in candidates_add[:50]:
                    if (time.time() - start_time) > time_limit:
                        break
                    adj[au, av] = 1
                    adj[av, au] = 1
                    new_books = compute_books(adj)
                    new_max = int(new_books.max())
                    if new_max <= max_book:
                        improved = True
                    else:
                        adj[au, av] = 0
                        adj[av, au] = 0
            
            if not improved:
                break
        
        return adj, max_book
    
    # Run local search
    current_adj, current_max_book = try_improve(current_adj, current_max_book, deadline - 5)
    
    if current_max_book < best_max_book:
        best_adj = current_adj.copy()
        best_max_book = current_max_book
    
    # Try continuous relaxation approach
    if (time.time() - start_time) < deadline - 10:
        # Use projected gradient descent on continuous relaxation
        # Objective: minimize sum of softmax-weighted book sizes
        W = current_adj.copy()
        
        for iteration in range(2000):
            if (time.time() - start_time) > deadline - 5:
                break
            
            A2 = W @ W
            # Book size for each potential edge
            book_sizes = np.zeros((n, n), dtype=np.float64)
            for u in range(n):
                for v in range(u+1, n):
                    book_sizes[u, v] = A2[u, v]
            
            # Softmax-weighted objective: focus on high book sizes
            temp = 0.5
            exp_books = np.exp(book_sizes / temp)
            weights = exp_books / (exp_books.sum() + 1e-10)
            
            # Gradient: for each edge (u,v), gradient of book_size wrt W[u,v]
            # book_size(u,v) = sum_w W[u,w]*W[v,w]
            # d(book_size)/d(W[u,v]) = W[v,v] + sum_w W[u,w]*d(W[v,w])/d(W[u,v])
            # For symmetric W, this is complex. Use finite differences or approximation.
            
            # Simpler: use coordinate descent
            # Pick random edge and adjust
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                continue
            
            # Try adjusting W[u,v]
            old_val = W[u, v]
            # Gradient approximation
            grad = 0
            for w in range(n):
                if w != u and w != v:
                    grad += W[u, w] * W[v, w]
            
            # Learning rate
            lr = 0.01
            new_val = old_val - lr * grad
            # Project to [0, 1]
            new_val = np.clip(new_val, 0, 1)
            
            W[u, v] = new_val
            W[v, u] = new_val
        
        # Round to binary
        rounded_adj = (W > 0.5).astype(np.float64)
        # Ensure symmetry
        rounded_adj = (rounded_adj + rounded_adj.T) / 2
        rounded_adj = (rounded_adj > 0.5).astype(np.float64)
        
        # Adjust to have exactly 401 edges
        ec = edge_count(rounded_adj)
        if ec > target_edges:
            # Remove edges with highest book size
            books = compute_books(rounded_adj)
            edges_to_remove = []
            for u in range(n):
                for v in range(u+1, n):
                    if rounded_adj[u, v] > 0.5:
                        edges_to_remove.append((u, v, books[u, v]))
            edges_to_remove.sort(key=lambda x: -x[2])
            for u, v, _ in edges_to_remove:
                if ec <= target_edges:
                    break
                rounded_adj[u, v] = 0
                rounded_adj[v, u] = 0
                ec -= 1
        elif ec < target_edges:
            # Add edges with lowest estimated book size
            A2 = rounded_adj @ rounded_adj
            edges_to_add = []
            for u in range(n):
                for v in range(u+1, n):
                    if rounded_adj[u, v] < 0.5:
                        edges_to_add.append((u, v, int(round(A2[u, v]))))
            edges_to_add.sort(key=lambda x: x[2])
            for u, v, _ in edges_to_add:
                if ec >= target_edges:
                    break
                rounded_adj[u, v] = 1
                rounded_adj[v, u] = 1
                ec += 1
        
        # Check if better
        if edge_count(rounded_adj) == target_edges:
            rb = get_max_book(rounded_adj)
            if rb < best_max_book:
                best_adj = rounded_adj.copy()
                best_max_book = rb
                # Continue local search from here
                best_adj, best_max_book = try_improve(best_adj, best_max_book, deadline - 3)
    
    # Final local search polish
    if (time.time() - start_time) < deadline - 1:
        best_adj, best_max_book = try_improve(best_adj, best_max_book, deadline - 0.5)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v] > 0.5:
                edges.append([u, v])
    
    # Verify exactly 401 edges
    assert len(edges) == target_edges, f"Expected {target_edges} edges, got {len(edges)}"
    
    return edges
# EVOLVE-BLOCK-END
