"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, 5s buffer
    
    N = 256
    # Squares up to 2*N
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Check if a value is valid for diagonal (2*v not a square)
    def diagonal_ok(v):
        return 2 * v not in squares
    
    # Build conflict: conflict if i+j is a square
    valid_nodes = [v for v in range(1, N + 1) if diagonal_ok(v)]
    M = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build adjacency as sets
    adj = [set() for _ in range(M)]
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    def greedy(order):
        """Greedy independent set given an ordering."""
        result = set()
        for idx in order:
            if not (adj[idx] & result):
                result.add(idx)
        return result
    
    best_set = set()
    best_size = 0
    
    # Try stride-3 order first
    initial = [v for v in range(1, 257, 3) if diagonal_ok(v)]
    initial_set = set()
    for v in initial:
        idx = node_to_idx[v]
        if not (adj[idx] & initial_set):
            initial_set.add(idx)
    
    if len(initial_set) > best_size:
        best_set = initial_set
        best_size = len(initial_set)
    
    # Try random orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        
        order = list(range(M))
        random.shuffle(order)
        current = greedy(order)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Local improvement on best_set: try to add elements
    current = set(best_set)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(M):
            if time.time() > deadline:
                break
            if idx in current:
                continue
            if not (adj[idx] & current):
                current.add(idx)
                changed = True
    
    if len(current) > best_size:
        best_set = current
        best_size = len(current)
    
    # Convert back to values
    result = [valid_nodes[i] for i in sorted(best_set)]
    
    if not result:
        return [1]
    
    return result
# EVOLVE-BLOCK-END
