import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    
    # Start with K16
    A = np.ones((n, n), dtype=np.int64)
    np.fill_diagonal(A, 0)
    
    def trace_A4(A):
        A2 = A @ A
        return np.sum(A2 * A2)
    
    def edge_reduction(A, i, j):
        """Compute reduction in trace(A^4) when removing edge (i,j)."""
        A2 = A @ A
        # Compute Δ(A^2)
        dA2 = np.zeros((n, n), dtype=np.int64)
        dA2[i, i] = -1
        dA2[j, j] = -1
        for k in range(n):
            if k == i or k == j:
                continue
            dA2[k, i] = -A[k, j]
            dA2[k, j] = -A[k, i]
            dA2[i, k] = -A[j, k]
            dA2[j, k] = -A[i, k]
        # Δ trace = sum 2*A2*dA2 + dA2^2
        delta = np.sum(2 * A2 * dA2 + dA2 * dA2)
        return delta
    
    # Iteratively remove edges
    for iteration in range(200):
        if time.time() > deadline:
            break
        # Find edge with maximum reduction
        best_delta = -np.inf
        best_edge = None
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    d = edge_reduction(A, i, j)
                    if d > best_delta:
                        best_delta = d
                        best_edge = (i, j)
        if best_edge is None:
            break
        i, j = best_edge
        A[i, j] = 0
        A[j, i] = 0
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                edges.append([i, j])
    
    return edges
