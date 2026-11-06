import numpy as np
import random
import time

def construct():
    n = 40
    m = 401
    deadline = time.time() + 110.0  # Leave 10s margin
    
    # Helper to calculate max book size
    def calc_max_book(A):
        # A is 40x40 int array
        C = A @ A
        # Get upper triangle indices
        iu, ju = np.triu_indices(n, k=1)
        # Mask for existing edges
        edge_mask = A[iu, ju] == 1
        if not np.any(edge_mask):
            return 0
        # Book sizes for existing edges
        books = C[iu, ju][edge_mask]
        return np.max(books)

    # Initialize random graph
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    # We need exactly m edges
    # Start with random m edges
    A = np.zeros((n, n), dtype=int)
    for u, v in all_edges[:m]:
        A[u, v] = 1
        A[v, u] = 1
        
    current_score = calc_max_book(A)
    best_score = current_score
    best_A = A.copy()
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    # Precompute edge list for faster access?
    # A is small, so iterating over all edges is fine.
    
    while time.time() < deadline:
        # Pick a random edge in graph
        # And a random non-edge
        # To speed up, we can maintain a list of edges and non-edges
        # But for n=40, finding them from A is O(n^2) = 1600, which is fast.
        
        # Get list of current edges
        current_edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    current_edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        if not current_edges or not non_edges:
            break
            
        # Bias: Pick edge with higher book size?
        # For simplicity, random pick.
        # But we can sample multiple times.
        
        u, v = random.choice(current_edges)
        x, y = random.choice(non_edges)
        
        # Perform swap
        A[u, v] = 0
        A[v, u] = 0
        A[x, y] = 1
        A[y, x] = 1
        
        new_score = calc_max_book(A)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = A.copy()
        elif new_score == current_score:
            # Accept with prob 0.5 to explore
            if random.random() < 0.5:
                current_score = new_score
            else:
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
        else:
            # new_score > current_score
            # Accept with prob exp(-(new_score - current_score) / T)
            delta = new_score - current_score
            if random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
                
        T *= cooling
        if T < T_min:
            T = T_min

    # Extract edges from best_A
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    return edges
