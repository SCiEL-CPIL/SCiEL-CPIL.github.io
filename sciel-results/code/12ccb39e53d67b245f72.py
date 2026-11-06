import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Use a numpy matrix for fast triangle counting
    # A is 40x40 boolean/int matrix
    # We want to minimize max over edges (u,v) of (A @ A)[u,v]
    # Note: (A @ A)[u,v] = sum_k A[u,k] A[k,v] = number of common neighbors of u and v
    # This is exactly the book size for edge (u,v).
    
    def count_books(A):
        # A is 0/1 matrix. Diagonal is 0.
        # Triangles containing (u,v) = number of w such that A[u,w]=1 and A[w,v]=1
        # This is the (u,v) entry of A @ A
        # We only care about entries where A[u,v]=1.
        M = A @ A
        # Mask out non-edges
        M = np.where(A == 1, M, 0)
        return int(M.max())
    
    # Initialize: Random graph with 401 edges
    # To be efficient, we can start with a random permutation of edges
    # Or simply random sampling without replacement.
    # 40 vertices -> 780 possible edges.
    # We need 401.
    
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
            
    # Shuffle and pick 401
    np.random.shuffle(all_edges)
    init_edges = all_edges[:target_edges]
    
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in init_edges:
        A[u, v] = 1
        A[v, u] = 1
        
    current_book = count_books(A)
    best_book = current_book
    best_A = A.copy()
    
    current_temp = 10.0
    cooling_rate = 0.995
    min_temp = 0.01
    
    # Precompute list of possible moves?
    # A move is: remove edge (u,v), add edge (x,y) where (x,y) != (u,v) and (x,y) not in E.
    # This is expensive to sample uniformly if we just pick random pairs.
    # Better: Pick a random existing edge to remove, and a random non-existing edge to add.
    
    # To speed up, we can maintain a list of edges and non-edges.
    # But updating lists is O(E) or O(V^2).
    # Given 120s budget, even O(V^2) per step is okay if steps are < 10k.
    # 1600 / 10k = 0.16ms per step. 10k steps in 1.6s. We can do more.
    
    # Let's try to be smarter.
    # We can just iterate.
    
    # Helper to get random edge and non-edge
    # We'll keep a set of edges for O(1) lookup?
    # Or just use the matrix A.
    
    while time.time() < deadline:
        # Pick random edge to remove
        # Random u, v
        while True:
            u = np.random.randint(n)
            v = np.random.randint(n)
            if u != v and A[u, v] == 1:
                break
        
        # Pick random non-edge to add
        while True:
            x = np.random.randint(n)
            y = np.random.randint(n)
            if x != y and A[x, y] == 0 and (x, y) != (u, v) and (y, x) != (u, v):
                # Ensure we don't pick the same edge we just removed
                # Actually if A[x,y]==0, it's not in the graph, so it can't be (u,v) which is in the graph.
                # Wait, we removed (u,v) conceptually? No, we haven't updated A yet.
                # So A[u,v] is still 1. So A[x,y]==0 implies (x,y) != (u,v).
                break
        
        # Apply move
        A[u, v] = 0
        A[v, u] = 0
        A[x, y] = 1
        A[y, x] = 1
        
        new_book = count_books(A)
        
        # Acceptance criterion
        if new_book < current_book or np.random.rand() < np.exp((current_book - new_book) / current_temp):
            current_book = new_book
            if new_book < best_book:
                best_book = new_book
                best_A = A.copy()
        else:
            # Revert
            A[u, v] = 1
            A[v, u] = 1
            A[x, y] = 0
            A[y, x] = 0
            
        current_temp *= cooling_rate
        if current_temp < min_temp:
            current_temp = min_temp

    # Return best found
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    # Verify count just in case
    # assert len(edges) == target_edges
    return edges
