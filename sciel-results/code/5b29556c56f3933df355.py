import numpy as np
import random
import time

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 115.0

    # Initialize A
    # A[i][j] = 1 if edge exists
    
    def get_initial_graph():
        # Start with a random graph of density ~0.5
        A = np.zeros((n, n), dtype=np.int8)
        edges = 0
        while edges < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and A[u, v] == 0:
                A[u, v] = 1
                A[v, u] = 1
                edges += 1
        return A

    def get_max_book(A):
        # A is 0/1 matrix
        # A @ A gives counts of walks of length 2
        # For an edge (u,v), the number of triangles containing it is (A@A)[u,v]
        # because any common neighbor w forms a triangle u-v-w-u
        # Note: (A@A)[u,v] includes paths u-w-v. Since A is symmetric and 0-diagonal, 
        # these are exactly common neighbors.
        A_sq = A @ A
        # Mask out non-edges
        # We only care about max book for edges that exist
        # books[i,j] = A_sq[i,j] if A[i,j]==1 else 0
        books = A_sq * A
        # The max book is the max value in this matrix
        # Note: diagonal will be 0 because A[i,i]=0, so books[i,i]=0
        return np.max(books)

    def random_swap(A):
        # Pick a random edge (u,v) to remove
        # and a random non-edge (x,y) to add
        
        # Get list of edges and non-edges
        # For speed, we can maintain lists, but for N=40, generating lists is cheap
        # Or we can just pick random u,v and check
        
        # Efficient way:
        # Create a flat array of indices?
        # Let's just use lists for clarity first, optimize if needed
        
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        if not edges or not non_edges:
            return None, None, None, None
            
        u, v = random.choice(edges)
        x, y = random.choice(non_edges)
        return u, v, x, y

    def apply_swap(A, u, v, x, y):
        A[u, v] = 0
        A[v, u] = 0
        A[x, y] = 1
        A[y, x] = 1

    def revert_swap(A, u, v, x, y):
        A[u, v] = 1
        A[v, u] = 1
        A[x, y] = 0
        A[y, x] = 0

    # Simulated Annealing
    best_A = None
    best_score = float('inf')
    
    # Multiple restarts to escape local minima
    num_restarts = 20
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        A = get_initial_graph()
        current_score = get_max_book(A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A.copy()
            
        T = 1.0
        cooling = 0.995
        
        # Inner loop for this restart
        # Estimate max steps: 150s / 20 restarts = 7.5s per restart
        # Each step: get_max_book (40x40 matmul) ~ 1-2 us? 
        # Python overhead is higher. Let's say 10k-50k steps per restart.
        
        steps_in_this_restart = 0
        max_steps = 50000 # Cap to prevent infinite loop if time check fails
        
        while steps_in_this_restart < max_steps:
            if time.time() > deadline - 5:
                break
                
            u, v, x, y = random_swap(A)
            if u is None:
                break
                
            apply_swap(A, u, v, x, y)
            new_score = get_max_book(A)
            
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = A.copy()
            else:
                revert_swap(A, u, v, x, y)
                
            T *= cooling
            steps_in_this_restart += 1
            
    # Convert best_A to list of edges
    if best_A is None:
        # Fallback
        A = get_initial_graph()
        best_A = A
        
    result_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                result_edges.append([i, j])
                
    return result_edges
