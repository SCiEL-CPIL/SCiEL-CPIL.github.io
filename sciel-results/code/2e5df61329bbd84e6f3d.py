import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    deadline = time.time() + 115.0

    def compute_books(A):
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        return A2

    def get_largest_book_edges(A2):
        """Return edges sorted by book size (descending)."""
        edge_books = []
        for i in range(n):
            for j in range(i+1, n):
                if A2[i, j] > 0:
                    edge_books.append((A2[i, j], i, j))
        edge_books.sort(reverse=True)
        return edge_books

    def find_best_replacement(A, edges, rem_u, rem_v, A2):
        """Find the best non-edge to add that minimizes max book after swap."""
        best_delta = float('inf')
        best_edge = None
        
        # Try adding edges that are non-edges
        for add_u in range(n):
            for add_v in range(add_u+1, n):
                if (min(add_u, add_v), max(add_u, add_v)) in edges:
                    continue
                if add_u == rem_u or add_v == rem_u or add_u == rem_v or add_v == rem_v:
                    continue
                
                # Estimate: removing (rem_u, rem_v) reduces books of neighbors
                # Adding (add_u, add_v) increases books of common neighbors
                common = np.logical_and(A[rem_u], A[rem_v])
                common_count = common.sum()
                
                common_add = np.logical_and(A[add_u], A[add_v])
                common_add_count = common_add.sum()
                
                # Heuristic: prefer swaps where we remove an edge with high book
                # and add one with low potential book
                est_delta = common_add_count - common_count
                if est_delta < best_delta:
                    best_delta = est_delta
                    best_edge = (add_u, add_v)
        
        return best_edge, best_delta

    # Initialize with a bipartite graph K_20,20 (400 edges) + 1 extra edge
    edges = set()
    A = np.zeros((n, n), dtype=np.int32)
    
    # Partition 0..39 into two sets of 20
    left = list(range(20))
    right = list(range(20, 40))
    
    # Add all edges between left and right
    for u in left:
        for v in right:
            edges.add((u, v))
            A[u, v] = 1
            A[v, u] = 1
            
    # We have 400 edges. Need 1 more.
    # Add a random edge within one partition to start
    u = random.choice(left)
    v = random.choice(left)
    while u == v:
        v = random.choice(left)
    new_edge = (min(u, v), max(u, v))
    edges.add(new_edge)
    A[u, v] = 1
    A[v, u] = 1

    A2 = compute_books(A)
    max_b = np.max(A2)
    best_edges = set(edges)
    best_max_b = max_b

    # Phase 1: Targeted book-breaking - repeatedly find and break the largest book
    while time.time() < deadline - 30:
        # Find edges with largest books
        edge_books = get_largest_book_edges(A2)
        
        if not edge_books or edge_books[0][0] == 0:
            break
        
        # Try to break the top few largest books
        improved = False
        for book_size, rem_u, rem_v in edge_books[:5]:
            if book_size <= best_max_b:
                break
                
            # Find best replacement
            best_edge, best_delta = find_best_replacement(A, edges, rem_u, rem_v, A2)
            
            if best_edge is None:
                continue
            
            add_u, add_v = best_edge
            
            # Apply swap
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A[add_u, add_v] = 1
            A[add_v, add_u] = 1
            
            A2_new = compute_books(A)
            max_new = np.max(A2_new)
            
            if max_new < max_b:
                # Accept
                edges.discard((rem_u, rem_v))
                edges.add((min(add_u, add_v), max(add_u, add_v)))
                A2 = A2_new
                max_b = max_new
                
                if max_b < best_max_b:
                    best_max_b = max_b
                    best_edges = set(edges)
                improved = True
            else:
                # Revert
                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A[add_u, add_v] = 0
                A[add_v, add_u] = 0
        
        if not improved:
            break

    # Phase 2: Simulated annealing for fine-tuning
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline - 2:
        # Pick a random edge to remove, but bias toward edges with larger books
        edge_books = get_largest_book_edges(A2)
        if edge_books:
            # 70% chance pick from top 20%, else random
            if random.random() < 0.7:
                top_idx = min(len(edge_books), max(1, len(edge_books) // 5))
                _, rem_u, rem_v = edge_books[random.randint(0, top_idx - 1)]
            else:
                rem_u, rem_v = random.choice(list(edges))
        else:
            rem_u, rem_v = random.choice(list(edges))
        
        # Pick a random non-edge to add
        added = False
        for _ in range(50):
            add_u = random.randint(0, n - 1)
            add_v = random.randint(0, n - 1)
            if add_u == add_v:
                continue
            add_edge = (min(add_u, add_v), max(add_u, add_v))
            if add_edge not in edges:
                added = True
                break
        
        if not added:
            continue
            
        # Apply swap
        A[rem_u, rem_v] = 0
        A[rem_v, rem_u] = 0
        A[add_u, add_v] = 1
        A[add_v, add_u] = 1
        
        A2_new = compute_books(A)
        max_new = np.max(A2_new)
        
        delta = max_new - max_b
        
        if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
            edges.discard((rem_u, rem_v))
            edges.add(add_edge)
            A2 = A2_new
            max_b = max_new
            
            if max_b < best_max_b:
                best_max_b = max_b
                best_edges = set(edges)
                T = max(T_min, T * 0.95)
            else:
                T = max(T_min, T * cooling)
        else:
            A[rem_u, rem_v] = 1
            A[rem_v, rem_u] = 1
            A[add_u, add_v] = 0
            A[add_v, add_u] = 0
            T = max(T_min, T * cooling)

    return [[u, v] for u, v in best_edges]
