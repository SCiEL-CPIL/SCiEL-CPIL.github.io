"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n = 40
    target_edges = 401

    def compute_books(A):
        """Compute book sizes for all existing edges."""
        A2 = A @ A
        books = np.zeros((n, n), dtype=np.int32)
        mask = (A == 1)
        books[mask] = A2[mask] - 1
        return books

    def get_max_book(books):
        return int(np.max(books))

    def local_search(A, deadline):
        """
        Local search using edge swaps to minimize max book size.
        """
        A = A.copy()
        books = compute_books(A)
        current_max = get_max_book(books)
        
        best_A = A.copy()
        best_max = current_max
        
        # List of edges and non-edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
        
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline:
            # Find the edge with the maximum book size
            # If there are multiple, pick one randomly? Or the first?
            # Let's pick the one with max book.
            max_val, max_idx = np.unravel_index(np.argmax(books), books.shape)
            u, v = max_idx
            if u > v:
                u, v = v, u
            
            # If max book is 0, we are optimal (theoretically impossible for m > n^2/4)
            if current_max <= 0:
                break
            
            # Try to swap (u, v) with a non-edge (x, y)
            # We want to reduce the max book.
            # Strategy: Pick (x, y) that minimizes the new max book.
            # We sample a few candidates to save time.
            
            K = 20
            if len(non_edges) < K:
                candidates = non_edges
            else:
                candidates = random.sample(non_edges, K)
            
            best_swap = None
            best_new_max = current_max
            
            for x, y in candidates:
                # Simulate swap
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                
                # Recompute books
                # Optimization: Only recompute if necessary?
                # A @ A is fast.
                new_books = compute_books(A)
                new_max = get_max_book(new_books)
                
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_swap = (x, y)
                    if new_max <= 7: # Theoretical lower bound
                        break
            
            if best_swap is None:
                # No improvement found in this sample
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Try a random swap to escape local minimum?
                    # Or just break?
                    # Let's try a random swap with some probability
                    if random.random() < 0.1:
                        x, y = random.choice(non_edges)
                        A[u, v] = 0; A[v, u] = 0
                        A[x, y] = 1; A[y, x] = 1
                        edges.remove((u, v)); edges.append((x, y))
                        non_edges.remove((x, y)); non_edges.append((u, v))
                        books = compute_books(A)
                        current_max = get_max_book(books)
                    else:
                        break
            else:
                # Apply swap
                x, y = best_swap
                A[u, v] = 0; A[v, u] = 0
                A[x, y] = 1; A[y, x] = 1
                
                # Update lists
                edges.remove((u, v))
                edges.append((x, y))
                non_edges.remove((x, y))
                non_edges.append((u, v))
                
                # Update books and max
                books = compute_books(A)
                current_max = get_max_book(books)
                
                if current_max < best_max:
                    best_max = current_max
                    best_A = A.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        return best_A, best_max

    def random_initial_graph():
        A = np.zeros((n, n), dtype=np.int32)
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:target_edges]:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def structured_initial_graph():
        # Start with K20,20
        A = np.zeros((n, n), dtype=np.int32)
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = 1
                A[j, i] = 1
        
        # Add 1 intra edge
        # To minimize book size, we should probably remove some cross edges first?
        # But we need exactly 401 edges.
        # If we just add 1 intra edge, book size is 20.
        # Local search will fix it.
        
        # Pick random intra edge
        u = random.randint(0, 19)
        v = random.randint(0, 19)
        while v == u:
            v = random.randint(0, 19)
        A[u, v] = 1
        A[v, u] = 1
        return A

    random.seed(42)
    
    best_A = None
    best_max = 999
    
    # Try structured initialization first
    A_init = structured_initial_graph()
    A_result, max_result = local_search(A_init, DEADLINE - 10)
    if max_result < best_max:
        best_max = max_result
        best_A = A_result.copy()
    
    # Try random initialization if time permits
    if time.time() < DEADLINE - 20 and best_max > 7:
        A_init = random_initial_graph()
        A_result, max_result = local_search(A_init, DEADLINE - 5)
        if max_result < best_max:
            best_max = max_result
            best_A = A_result.copy()
            
    # If still not optimal, try another random restart
    if time.time() < DEADLINE - 10 and best_max > 7:
        A_init = random_initial_graph()
        A_result, max_result = local_search(A_init, DEADLINE - 2)
        if max_result < best_max:
            best_max = max_result
            best_A = A_result.copy()

    final_A = best_A
    
    # Extract edges
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if final_A[i, j] == 1:
                edge_list.append([i, j])
    
    # Ensure exactly 401 edges
    if len(edge_list) != target_edges:
        # This should not happen if local_search preserves edge count
        # But just in case
        if len(edge_list) < target_edges:
            all_nonedges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                           if final_A[i, j] == 0]
            random.shuffle(all_nonedges)
            for u, v in all_nonedges:
                if len(edge_list) >= target_edges:
                    break
                final_A[u, v] = 1
                final_A[v, u] = 1
                edge_list.append([u, v])
        else:
            edge_list = edge_list[:target_edges]
            final_A = np.zeros((n, n), dtype=np.int32)
            for u, v in edge_list:
                final_A[u, v] = 1
                final_A[v, u] = 1
    
    assert len(edge_list) == target_edges, f"Edge count {len(edge_list)} != {target_edges}"
    
    return edge_list
# EVOLVE-BLOCK-END
