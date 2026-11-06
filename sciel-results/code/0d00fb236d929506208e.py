import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    time_limit = 140
    
    best_edges = None
    best_score = float('inf')
    
    def get_max_book_and_edges(A):
        A2 = A @ A
        # Get indices where A is 1
        # We want the max value in A2 where A is 1
        # And the edges that have this max value
        mask = (A == 1)
        # To avoid double counting, we can just look at upper triangle
        # But A2 is symmetric, so values are same.
        # We just need the max value.
        # And the list of edges having that value.
        max_val = 0
        max_edges = []
        # Iterate over upper triangle
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 1:
                    b = A2[u, v]
                    if b > max_val:
                        max_val = b
                        max_edges = [(u, v)]
                    elif b == max_val:
                        max_edges.append((u, v))
        return max_val, max_edges

    # This is O(n^2) in Python, which is 1600 ops.
    # A @ A is O(n^3) = 64000 ops in C.
    # The Python loop might be slower than the C matrix mult.
    # Let's stick to NumPy for max, but we need the EDGES.
    # We can get the edges from the mask.
    
    def get_max_book_info(A):
        A2 = A @ A
        mask = (A == 1)
        # Set diagonal to 0 to avoid self-loops if any (none here)
        np.fill_diagonal(A2, 0)
        # We want max of A2 * mask
        # But we also need the indices.
        # A2_masked = np.where(mask, A2, 0)
        # max_val = np.max(A2_masked)
        # indices = np.argwhere(A2_masked == max_val)
        
        A2_masked = np.where(mask, A2, 0)
        max_val = int(np.max(A2_masked))
        # Get edges
        indices = np.argwhere(A2_masked == max_val)
        # Convert to list of tuples (u, v) with u < v
        edges = []
        for u, v in indices:
            if u < v:
                edges.append((u, v))
            elif v < u:
                edges.append((v, u))
        return max_val, edges

    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            edges.add((u, v))
        return edges

    def edges_to_matrix(edges):
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def local_search(initial_edges):
        current_edges = set(initial_edges)
        A = edges_to_matrix(current_edges)
        current_score, high_edges = get_max_book_info(A)
        
        no_improve_count = 0
        
        while time.time() < time_limit:
            if not high_edges:
                break
                
            # Pick a random high edge
            u, v = random.choice(high_edges)
            
            # Pick a random non-edge
            x, y = random.randint(0, n-1), random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) in current_edges:
                continue
            
            # Perform swap
            current_edges.remove((u, v))
            current_edges.add((x, y))
            
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            new_score, new_high_edges = get_max_book_info(A)
            
            if new_score < current_score:
                current_score = new_score
                high_edges = new_high_edges
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
                no_improve_count = 0
            else:
                # Revert
                current_edges.remove((x, y))
                current_edges.add((u, v))
                A[x, y] = 0
                A[y, x] = 0
                A[u, v] = 1
                A[v, u] = 1
                no_improve_count += 1
                
            if no_improve_count > 2000:
                # Restart
                current_edges = random_graph()
                A = edges_to_matrix(current_edges)
                current_score, high_edges = get_max_book_info(A)
                no_improve_count = 0
                
        return best_edges, best_score

    initial_edges = random_graph()
    edges, score = local_search(initial_edges)
    
    if edges is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return [[u, v] for u, v in edges]

