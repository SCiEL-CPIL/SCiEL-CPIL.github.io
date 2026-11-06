"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 40
    target_edges = 401
    total_possible_edges = n * (n - 1) // 2
    
    # Initialize with a random graph
    A = np.zeros((n, n), dtype=bool)
    edges = []
    non_edges = []
    
    # Build initial edge list
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs[:target_edges]:
        A[u, v] = True
        A[v, u] = True
        edges.append((u, v))
    
    for u, v in all_pairs[target_edges:]:
        non_edges.append((u, v))
    
    def compute_all_books(A):
        """Compute book size for every possible edge (including non-edges)."""
        books = {}
        for i in range(n):
            for j in range(i+1, n):
                books[(i, j)] = np.sum(A[i] & A[j])
        return books
    
    def get_max_book_from_books(books, edge_set):
        """Get max book size among edges in edge_set."""
        max_b = 0
        for e in edge_set:
            if e in books:
                b = books[e]
                if b > max_b:
                    max_b = b
        return max_b
    
    def update_books_after_swap(books, A, removed_edge, added_edge):
        """
        Incrementally update book sizes after removing removed_edge and adding added_edge.
        Only update books for edges that are affected.
        """
        u, v = removed_edge
        a, b = added_edge
        
        # Removing edge (u,v) decreases book size of edges (u,x) and (v,y) 
        # where x~v and y~u
        # Adding edge (a,b) increases book size of edges (a,x) and (b,y)
        # where x~b and y~a
        
        # Edges affected by removal of (u,v):
        # For each common neighbor w of u and v, edge (u,w) and (v,w) lose 1
        # Actually, book size of edge (u,w) = common neighbors of u and w
        # Removing (u,v) affects book sizes of edges incident to u or v
        # where the other endpoint was a common neighbor
        
        # Let's just recompute books for edges incident to u, v, a, b
        affected_nodes = set([u, v, a, b])
        
        # Recompute book sizes for all edges incident to these nodes
        for node in affected_nodes:
            for other in range(n):
                if other != node:
                    edge = (min(node, other), max(node, other))
                    books[edge] = np.sum(A[node] & A[other])
        
        return books
    
    # Initial books
    books = compute_all_books(A)
    current_max = get_max_book_from_books(books, set(edges))
    best_A = A.copy()
    best_max = current_max
    best_books = books.copy()
    
    # Targeted local search: repeatedly find worst edge and try to break it
    max_iterations = 50000
    iterations = 0
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Find the worst edge(s)
        max_b = current_max
        worst_edges = [e for e in edges if books.get(e, 0) == max_b]
        
        if not worst_edges:
            break
        
        random.shuffle(worst_edges)
        
        # Try to break the worst edge
        improved = False
        for worst_edge in worst_edges[:3]:  # Try up to 3 worst edges
            if time.time() > deadline:
                break
            
            u, v = worst_edge
            
            # Find common neighbors of u and v (these are the "book" vertices)
            common_neighbors = np.where(A[u] & A[v])[0]
            common_neighbors = [w for w in common_neighbors if w != u and w != v]
            
            if not common_neighbors:
                continue
            
            # For each common neighbor w, try removing edge (u,w) or (v,w)
            # and adding a compensating edge
            candidates = []
            for w in common_neighbors:
                # Option 1: remove (u,w), add some non-edge
                # Option 2: remove (v,w), add some non-edge
                candidates.append(('remove_uw', u, w))
                candidates.append(('remove_vw', v, w))
            
            random.shuffle(candidates)
            
            for move_type, x, y in candidates[:10]:  # Try up to 10 candidates
                if time.time() > deadline:
                    break
                
                # Remove edge (x,y)
                if not A[x, y]:
                    continue
                    
                # Find a non-edge to add
                # Prefer non-edges with low current book size
                non_edge_candidates = [ne for ne in non_edges if books.get(ne, 0) <= max_b - 1]
                
                if not non_edge_candidates:
                    non_edge_candidates = non_edges[:20]
                
                random.shuffle(non_edge_candidates)
                
                for a, b in non_edge_candidates[:15]:
                    if time.time() > deadline:
                        break
                    
                    # Perform swap: remove (x,y), add (a,b)
                    A[x, y] = False
                    A[y, x] = False
                    A[a, b] = True
                    A[b, a] = True
                    
                    # Update edge lists
                    try:
                        edges.remove((x, y))
                    except:
                        pass
                    edges.append((a, b))
                    
                    try:
                        non_edges.remove((a, b))
                    except:
                        pass
                    non_edges.append((x, y))
                    
                    # Update books incrementally
                    books = update_books_after_swap(books, A, (x, y), (a, b))
                    
                    # Compute new max
                    new_max = get_max_book_from_books(books, set(edges))
                    
                    if new_max < current_max:
                        # Accept
                        current_max = new_max
                        improved = True
                        
                        if current_max < best_max:
                            best_max = current_max
                            best_A = A.copy()
                            best_books = books.copy()
                        
                        break
                    else:
                        # Revert
                        A[x, y] = True
                        A[y, x] = True
                        A[a, b] = False
                        A[b, a] = False
                        
                        try:
                            edges.remove((a, b))
                        except:
                            pass
                        edges.append((x, y))
                        
                        try:
                            non_edges.remove((x, y))
                        except:
                            pass
                        non_edges.append((a, b))
                        
                        books = update_books_after_swap(books, A, (a, b), (x, y))
                
                if improved:
                    break
            
            if improved:
                break
        
        # If no improvement, try a random perturbation
        if not improved and iterations % 100 == 0:
            # Perturb: remove 3 random edges, add 3 random non-edges
            for _ in range(3):
                if edges:
                    u, v = random.choice(edges)
                    A[u, v] = False
                    A[v, u] = False
                    edges.remove((u, v))
                    non_edges.append((u, v))
            
            for _ in range(3):
                if non_edges:
                    a, b = random.choice(non_edges)
                    A[a, b] = True
                    A[b, a] = True
                    non_edges.remove((a, b))
                    edges.append((a, b))
            
            # Recompute books
            books = compute_all_books(A)
            current_max = get_max_book_from_books(books, set(edges))
            
            if current_max < best_max:
                best_max = current_max
                best_A = A.copy()
                best_books = books.copy()
    
    # Convert best_A to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                result.append([i, j])
    
    # Verify count
    if len(result) != 401:
        # Fallback: build a valid graph
        result = []
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        result = [list(e) for e in all_pairs[:target_edges]]
    
    return result
# EVOLVE-BLOCK-END
