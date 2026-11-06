"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Use NumPy for fast adjacency operations
    # A is a boolean matrix n x n
    A = np.zeros((n, n), dtype=bool)
    
    def get_edges(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    edges.append((i, j))
        return edges

    def count_edges(A):
        return np.sum(A) // 2

    def compute_book_sizes(A):
        """
        Returns a dict: edge (i, j) -> book size.
        Book size of (i, j) is number of common neighbors.
        """
        books = {}
        # Vectorized calculation of common neighbors for all pairs?
        # Triangles containing (i, j) = sum(A[i, :] & A[j, :])
        # We only care about edges.
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    # Common neighbors
                    common = np.sum(A[i] & A[j])
                    books[(i, j)] = common
        return books

    def get_max_book(A):
        """Compute max book size efficiently."""
        max_b = 0
        # Iterate only over edges
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    b = np.sum(A[i] & A[j])
                    if b > max_b:
                        max_b = b
        return max_b

    # 1. Initialize with a random graph of target_edges
    # To ensure connectivity and good structure, we can start with a random regular-ish graph
    # Or just random.
    A = np.zeros((n, n), dtype=bool)
    edges_list = []
    while count_edges(A) < target_edges:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and not A[u, v]:
            A[u, v] = True
            A[v, u] = True
    
    current_max = get_max_book(A)
    best_A = A.copy()
    best_max = current_max
    
    # 2. Local Search: Swap high-book edge with low-book non-edge
    # We will iterate.
    # To speed up, we precompute book sizes for all possible edges (including non-edges)
    # and update them incrementally?
    # For N=40, recomputing is fast enough if we do it smartly.
    
    # Let's try a simple random restart local search.
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # If not first restart, randomize slightly
        if restart > 0:
             # Perturb best_A
             A = best_A.copy()
             # Remove 5 random edges, add 5 random non-edges
             for _ in range(5):
                 edges = []
                 for i in range(n):
                     for j in range(i+1, n):
                         if A[i, j]:
                             edges.append((i, j))
                 if not edges: break
                 u, v = random.choice(edges)
                 A[u, v] = False
                 A[v, u] = False
             
             for _ in range(5):
                 non_edges = []
                 for i in range(n):
                     for j in range(i+1, n):
                         if not A[i, j]:
                             non_edges.append((i, j))
                 if not non_edges: break
                 u, v = random.choice(non_edges)
                 A[u, v] = True
                 A[v, u] = True
             
             current_max = get_max_book(A)
        
        # Local search loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Compute book sizes for all edges
            edge_books = {}
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j]:
                        edge_books[(i, j)] = np.sum(A[i] & A[j])
            
            if not edge_books:
                break
                
            # Find max book edge
            max_b = max(edge_books.values())
            
            # Find all edges with this max book
            max_edges = [e for e, b in edge_books.items() if b == max_b]
            
            # Try to swap one of the max edges
            random.shuffle(max_edges)
            swapped = False
            for u, v in max_edges:
                if time.time() > deadline:
                    break
                # Remove (u, v)
                A[u, v] = False
                A[v, u] = False
                
                # Find a non-edge (a, b) to add that minimizes new max book
                # We can sample some non-edges
                best_add = None
                best_new_max = max_b + 100
                
                # Sample 50 random non-edges
                for _ in range(50):
                    a = random.randint(0, n-1)
                    b = random.randint(0, n-1)
                    if a < b and not A[a, b]:
                        # Calculate book size of (a, b) in current A
                        b_size = np.sum(A[a] & A[b])
                        # Also need to check if this addition increases max book elsewhere?
                        # Adding (a, b) increases book size of edges (a, x) and (b, y) where x~b, y~a.
                        # This is complex to evaluate incrementally.
                        # Just recompute max book after swap?
                        # If we recompute, we don't need best_new_max heuristic.
                        
                        # Let's just pick the non-edge with smallest *current* book size.
                        # This is a greedy heuristic.
                        if b_size < best_new_max:
                            best_new_max = b_size
                            best_add = (a, b)
                
                if best_add:
                    a, b = best_add
                    A[a, b] = True
                    A[b, a] = True
                    
                    new_max = get_max_book(A)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        swapped = True
                        if current_max < best_max:
                            best_max = current_max
                            best_A = A.copy()
                        break
                    else:
                        # Revert
                        A[a, b] = False
                        A[b, a] = False
                
                # Revert removal of (u, v)
                A[u, v] = True
                A[v, u] = True
                
                if swapped:
                    break

    # Convert best_A to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                result.append([i, j])
    
    # Verify count
    if len(result) != 401:
        # Fallback: just return a valid random graph if something went wrong
        # But we should ensure validity.
        pass
        
    return result
# EVOLVE-BLOCK-END
