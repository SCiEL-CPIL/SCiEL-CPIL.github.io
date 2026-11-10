import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n = 31
    target_edges = 91
    
    def count_c4(A):
        # A is n x n numpy array (0/1)
        M = A @ A
        d = np.sum(A, axis=1)
        sum_M_sq = np.sum(M**2)
        sum_d_sq = np.sum(d**2)
        
        # S2 = sum_{u<v} N(u,v)^2
        # sum_{i != j} M[i,j]^2 = sum_M_sq - sum_d_sq
        # sum_{u<v} N(u,v)^2 = 0.5 * (sum_M_sq - sum_d_sq)
        S2 = 0.5 * (sum_M_sq - sum_d_sq)
        
        # S1 = sum_{u<v} N(u,v)
        # sum_{i != j} M[i,j] = sum_M - trace(M) = sum_d_sq - sum_d
        # Wait, sum(M) = sum_d_sq. trace(M) = sum_d.
        # So sum_{i != j} M[i,j] = sum_d_sq - sum_d.
        # sum_{u<v} N(u,v) = 0.5 * (sum_d_sq - sum_d)
        S1 = 0.5 * (sum_d_sq - np.sum(d))
        
        c4 = 0.5 * (S2 - S1)
        return int(c4)

    def get_edges(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    edges.append([i, j])
        return edges

    # Initialize with a random graph
    def random_graph():
        A = np.zeros((n, n), dtype=np.int8)
        edges = []
        # Generate all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:target_edges]:
            A[u, v] = 1
            A[v, u] = 1
        return A

    best_A = None
    best_c4 = float('inf')
    
    # Try a few random starts
    for run in range(10):
        if time.time() > deadline - 10:
            break
            
        A = random_graph()
        current_c4 = count_c4(A)
        
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_A = A.copy()
            
        # Simulated Annealing
        T = 100.0
        cooling = 0.995
        
        # Number of steps
        steps = 5000
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Propose move: remove one edge, add one non-edge
            # Get list of edges and non-edges
            # To speed up, we can pick random i, j
            # But we need to ensure we remove an edge and add a non-edge.
            
            # Efficient way: 
            # 1. Pick random edge to remove
            # 2. Pick random non-edge to add
            
            # We can maintain lists, but for n=31, scanning is okay?
            # 31*31/2 = 465 edges.
            # Let's just pick random pairs.
            
            # Remove
            while True:
                u1 = random.randint(0, n-1)
                v1 = random.randint(0, n-1)
                if u1 < v1 and A[u1, v1] == 1:
                    break
                if u1 > v1 and A[v1, u1] == 1:
                    u1, v1 = v1, u1
                    break
                if u1 == v1:
                    continue
            
            A[u1, v1] = 0
            A[v1, u1] = 0
            
            # Add
            while True:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2:
                    continue
                if u2 < v2 and A[u2, v2] == 0:
                    break
                if u2 > v2 and A[v2, u2] == 0:
                    u2, v2 = v2, u2
                    break
            
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            new_c4 = count_c4(A)
            delta = new_c4 - current_c4
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_c4 = new_c4
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_A = A.copy()
            else:
                # Revert
                A[u2, v2] = 0
                A[v2, u2] = 0
                A[u1, v1] = 1
                A[v1, u1] = 1
            
            T *= cooling
            
        # Restart SA with lower temperature if time permits? 
        # The loop above does one SA run.
        
    if best_A is None:
        # Fallback to seed
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
        
    return get_edges(best_A)
