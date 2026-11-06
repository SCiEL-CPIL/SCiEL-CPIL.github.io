import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 40
    target_edges = 401
    
    def generate_random_regular_graph(n, d):
        # Start with K_20,20
        A = np.zeros((n, n), dtype=int)
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = 1
                A[j, i] = 1
                
        # Randomize using swaps to break structure and randomize
        # We want a random 20-regular graph
        for _ in range(20000):
            # Pick two edges
            # Efficiently pick edges:
            # We can just pick random u, v and check if edge exists
            # But picking from list is better
            edges = np.argwhere(A == 1)
            edges = edges[edges[:, 0] < edges[:, 1]]
            if len(edges) < 2:
                continue
            e1, e2 = random.sample(edges, 2)
            a, b = e1
            c, d = e2
            # Try to swap to (a,c) and (b,d)
            if a == c or a == d or b == c or b == d:
                continue
            if A[a, c] == 0 and A[b, d] == 0:
                A[a, b] = 0; A[b, a] = 0
                A[c, d] = 0; A[d, c] = 0
                A[a, c] = 1; A[c, a] = 1
                A[b, d] = 1; A[d, b] = 1
            elif A[a, d] == 0 and A[b, c] == 0:
                A[a, b] = 0; A[b, a] = 0
                A[c, d] = 0; A[d, c] = 0
                A[a, d] = 1; A[d, a] = 1
                A[b, c] = 1; A[c, b] = 1
                
        return A

    def get_max_book(A):
        A2 = A @ A
        mask = (A == 1) & (np.eye(n, dtype=int) == 0)
        books = A2 * mask
        return np.max(books)

    def add_best_edge(A):
        A2 = A @ A
        mask = (A == 0) & (np.eye(n, dtype=int) == 0)
        A2_masked = np.where(mask, A2, np.inf)
        min_val = np.min(A2_masked)
        u, v = np.unravel_index(np.argmin(A2_masked), A2_masked.shape)
        A[u, v] = 1
        A[v, u] = 1
        return A, min_val

    best_A = None
    best_score = float('inf')
    
    # Try a few random starts
    for i in range(10):
        if time.time() > deadline:
            break
        A = generate_random_regular_graph(n, 20)
        A, _ = add_best_edge(A)
        score = get_max_book(A)
        
        if score < best_score:
            best_score = score
            best_A = A.copy()
            
        # Local search
        current_A = A.copy()
        current_score = score
        T = 1.0
        # Run local search for a fixed number of steps or time
        steps = 0
        while steps < 20000 and time.time() < deadline:
            steps += 1
            # Pick random edge to remove
            edges = np.argwhere(current_A == 1)
            edges = edges[edges[:, 0] < edges[:, 1]]
            e_rem = random.choice(edges)
            u1, v1 = e_rem
            
            # Pick random non-edge to add
            non_edges = np.argwhere(current_A == 0)
            non_edges = non_edges[(non_edges[:, 0] < non_edges[:, 1]) & (non_edges[:, 0] != non_edges[:, 1])]
            if len(non_edges) == 0:
                continue
            e_add = random.choice(non_edges)
            u2, v2 = e_add
            
            # Apply move
            current_A[u1, v1] = 0; current_A[v1, u1] = 0
            current_A[u2, v2] = 1; current_A[v2, u2] = 1
            
            new_score = get_max_book(current_A)
            
            # Accept if better or with prob exp(-delta/T)
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A.copy()
            else:
                # Revert
                current_A[u1, v1] = 1; current_A[v1, u1] = 1
                current_A[u2, v2] = 0; current_A[v2, u2] = 0
            
            T *= 0.995

    # Convert best_A to list of edges
    if best_A is None:
        # Fallback to seed
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    return edges
