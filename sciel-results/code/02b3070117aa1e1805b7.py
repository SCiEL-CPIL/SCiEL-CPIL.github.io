import time
import numpy as np
import random

def construct():
    n = 40
    deadline = 115.0  # Safety margin from 120s limit (evaluator kills at 150s)
    start_time = time.time()
    
    # Helper to compute A^5 for a 40x40 int matrix
    def compute_A5(mat):
        # mat is 40x40 int64
        A2 = mat @ mat
        A3 = A2 @ mat
        A4 = A3 @ mat
        A5 = A4 @ mat
        return A5
    
    # Initialize with 8 disjoint K5s
    def initial_k5():
        A = np.zeros((n, n), dtype=np.int64)
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        return A
    
    best_A = initial_k5()
    best_count = np.sum(best_A) // 2
    
    # Main search loop: greedy add + occasional swap to escape local optima
    while time.time() < deadline:
        A = best_A.copy()
        A5 = compute_A5(A)
        
        # Phase 1: Greedy add
        # Try to add random non-edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle non-edges for diversity
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        non_edges.append((u, v))
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if A[u, v] == 1:
                    continue
                if A5[u, v] == 0:
                    # Safe to add
                    A[u, v] = 1
                    A[v, u] = 1
                    A5 = compute_A5(A)
                    improved = True
                    current_count = np.sum(A) // 2
                    if current_count > best_count:
                        best_count = current_count
                        best_A = A.copy()
        
        # Phase 2: Try edge swaps to escape local optima
        if time.time() > deadline:
            break
        
        # Get list of current edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 1:
                    edges.append((u, v))
        
        if not edges:
            continue
        
        # Try random edge removals and re-additions
        attempts = 0
        max_swap_attempts = 20
        while attempts < max_swap_attempts and time.time() < deadline:
            attempts += 1
            
            # Pick a random edge to remove
            rem_u, rem_v = random.choice(edges)
            
            # Save state
            A_old = A.copy()
            
            # Remove edge
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A5 = compute_A5(A)
            
            # Try to add 2 edges to get net +1
            added = 0
            max_add_attempts = 50
            add_attempts = 0
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        non_edges.append((u, v))
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if add_attempts >= max_add_attempts:
                    break
                if added >= 2:
                    break
                if time.time() > deadline:
                    break
                if A[u, v] == 1:
                    continue
                if A5[u, v] == 0:
                    A[u, v] = 1
                    A[v, u] = 1
                    A5 = compute_A5(A)
                    added += 1
                    add_attempts += 1
            
            current_count = np.sum(A) // 2
            if current_count > best_count:
                best_count = current_count
                best_A = A.copy()
                # Update edges list for next swap
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if A[u, v] == 1:
                            edges.append((u, v))
            else:
                # Revert
                A = A_old
    
    # Return best found
    edges_out = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges_out.append([u, v])
    return edges_out
