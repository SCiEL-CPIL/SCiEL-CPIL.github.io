import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize random graph
    A = np.zeros((n, n), dtype=np.int8)
    edges = set()
    
    # Generate random edges
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            if u > v: u, v = v, u
            if (u, v) not in edges:
                edges.add((u, v))
                A[u, v] = 1
                A[v, u] = 1
                
    def get_score(A, edges):
        if not edges: return 0
        M = A @ A.T
        max_val = 0
        for u, v in edges:
            val = M[u, v]
            if val > max_val:
                max_val = val
        return int(max_val)

    current_score = get_score(A, edges)
    best_score = current_score
    best_A = A.copy()
    best_edges = set(edges)
    
    # Local Search
    while time.time() < deadline:
        # Pick random edge to remove
        u1, v1 = random.choice(list(edges))
        
        # Pick random non-edge to add
        # To avoid infinite loop picking existing edges, we can just try random pairs
        # Since density is ~50%, random pair has 50% chance of being non-edge.
        added = False
        for _ in range(10): # Try a few times
             u2 = random.randint(0, n-1)
             v2 = random.randint(0, n-1)
             if u2 == v2: continue
             if u2 > v2: u2, v2 = v2, u2
             if (u2, v2) not in edges:
                 added = True
                 break
        if not added:
             continue

        # Perform Swap
        # Remove (u1, v1)
        A[u1, v1] = 0
        A[v1, u1] = 0
        edges.remove((u1, v1))
        
        # Add (u2, v2)
        A[u2, v2] = 1
        A[v2, u2] = 1
        edges.add((u2, v2))
        
        new_score = get_score(A, edges)
        
        # Acceptance Criteria
        if new_score <= current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = A.copy()
                best_edges = set(edges)
        else:
            # Revert
            A[u1, v1] = 1
            A[v1, u1] = 1
            edges.add((u1, v1))
            
            A[u2, v2] = 0
            A[v2, u2] = 0
            edges.remove((u2, v2))
            
    # Return best result
    # Convert best_edges to list of lists
    return [[u, v] for u, v in best_edges]
