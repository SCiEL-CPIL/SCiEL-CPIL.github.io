import numpy as np
import time
import random
import math

def construct():
    n = 20
    start_time = time.time()
    timeout = 115
    
    def initialize_greedy():
        """Create a starting graph using a randomized greedy approach."""
        A = np.zeros((n, n), dtype=int)
        K = np.zeros((n, n), dtype=int)
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        edge_count = 0
        for u, v in edges:
            # Check if adding (u, v) is valid
            # 1. u and v common neighbors < 3
            if K[u, v] >= 2:
                continue
            # 2. For all x in N(u), K[x, v] < 2
            valid = True
            for x in range(n):
                if A[u, x] == 1:
                    if K[x, v] >= 2:
                        valid = False
                        break
            if not valid:
                continue
            # 3. For all y in N(v), K[u, y] < 2
            for y in range(n):
                if A[v, y] == 1:
                    if K[u, y] >= 2:
                        valid = False
                        break
            if not valid:
                continue
            
            # Add edge
            A[u, v] = 1
            A[v, u] = 1
            edge_count += 1
            # Update K
            # K[x, v] += 1 for x in N(u)
            for x in range(n):
                if A[u, x] == 1 and x != v:
                    K[x, v] += 1
                    K[v, x] += 1
            # K[u, y] += 1 for y in N(v)
            for y in range(n):
                if A[v, y] == 1 and y != u:
                    K[u, y] += 1
                    K[y, u] += 1
        return A, K, edge_count

    def can_add(A, K, u, v):
        if A[u, v] == 1:
            return False
        if K[u, v] >= 2:
            return False
        for x in range(n):
            if A[u, x] == 1:
                if K[x, v] >= 2:
                    return False
        for y in range(n):
            if A[v, y] == 1:
                if K[u, y] >= 2:
                    return False
        return True

    def add_edge(A, K, u, v):
        A[u, v] = 1
        A[v, u] = 1
        for x in range(n):
            if A[u, x] == 1 and x != v:
                K[x, v] += 1
                K[v, x] += 1
        for y in range(n):
            if A[v, y] == 1 and y != u:
                K[u, y] += 1
                K[y, u] += 1

    def remove_edge(A, K, u, v):
        A[u, v] = 0
        A[v, u] = 0
        for x in range(n):
            if A[u, x] == 1 and x != v:
                K[x, v] -= 1
                K[v, x] -= 1
        for y in range(n):
            if A[v, y] == 1 and y != u:
                K[u, y] -= 1
                K[y, u] -= 1

    # Run multiple SA instances
    best_edges = 0
    best_A = None
    best_K = None
    
    num_sa_runs = 50
    
    for run in range(num_sa_runs):
        # Initialize
        A, K, edge_count = initialize_greedy()
        current_edges = edge_count
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_A = A.copy()
            best_K = K.copy()
        
        if time.time() - start_time > timeout:
            break
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while T > T_min and time.time() - start_time < timeout:
            # Generate a random move
            # Choose a random pair (u, v)
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            is_add = (A[u, v] == 0)
            
            if is_add:
                if can_add(A, K, u, v):
                    # Accept if valid (always improves or equal, but we are maximizing edges, so add is good)
                    add_edge(A, K, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_A = A.copy()
                        best_K = K.copy()
                else:
                    # Reject
                    pass
            else:
                # Remove edge
                # Always valid
                remove_edge(A, K, u, v)
                current_edges -= 1
                # In SA, we might accept worse moves to escape local optima
                # But here, removing an edge is always "worse" in terms of objective.
                # We accept it with probability exp(-delta / T)
                # delta = -1 (change in edges)
                # prob = exp(1/T) > 1? No, cost function is -edges.
                # Let's define cost = -current_edges.
                # New cost = -(current_edges - 1) = current_edges.
                # Old cost = -current_edges.
                # Delta = New - Old = 1.
                # Accept if random < exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    # Keep removal
                    pass
                else:
                    # Reject removal, put it back
                    add_edge(A, K, u, v)
                    current_edges += 1
            
            T *= cooling
            
            # If we are stuck, we might need to cool faster or just continue
            # The above logic accepts removals probabilistically.
            # This allows the graph to shrink to find a better configuration.

    # Extract edges from best_A
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
                
    return edges
