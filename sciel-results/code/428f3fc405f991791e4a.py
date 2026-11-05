import random
import time
import sys
import numpy as np

def construct():
    n = 24
    # Generate and shuffle edges with pinned seed
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    r = random.Random(614)
    r.shuffle(edges)
    
    # Build edge arrival index: edge_index[(u,v)] = 0-based index in shuffled list
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    def get_arrival(u, v):
        return edge_index[(u, v)]
    
    # Precompute a good initial cycle using greedy + local search on the full graph
    # We want to find a cycle with minimal max edge arrival index.
    
    # Strategy: 
    # 1. Use simulated annealing / local search on permutations to minimize max edge weight
    # 2. Edge weight = arrival index in the shuffled list
    # 3. The objective is min over Hamilton cycles of max(arrival(e) for e in cycle)
    
    # Start with a random permutation
    best_cycle = None
    best_score = float('inf')
    
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Generate initial solution: use a greedy approach
    # For each vertex, prefer edges with low arrival index
    def greedy_cycle():
        """Greedy construction: start at 0, at each step pick the unvisited neighbor with lowest arrival index."""
        visited = [False] * n
        visited[0] = True
        path = [0]
        for _ in range(n - 1):
            cur = path[-1]
            best_next = -1
            best_arr = float('inf')
            for v in range(n):
                if not visited[v]:
                    arr = edge_index[(cur, v)]
                    if arr < best_arr:
                        best_arr = arr
                        best_next = v
            if best_next == -1:
                return None
            visited[best_next] = True
            path.append(best_next)
        # Check closing edge
        if not (edge_index.get((path[-1], 0)) is not None):
            return None
        return path
    
    # Try greedy from different starting points
    for start in range(n):
        if time.time() > deadline:
            break
        # Rotate and try
        visited = [False] * n
        visited[start] = True
        path = [start]
        for _ in range(n - 1):
            cur = path[-1]
            best_next = -1
            best_arr = float('inf')
            for v in range(n):
                if not visited[v]:
                    arr = edge_index[(cur, v)]
                    if arr < best_arr:
                        best_arr = arr
                        best_next = v
            if best_next == -1:
                break
            visited[best_next] = True
            path.append(best_next)
        if len(path) == n and edge_index.get((path[-1], start)) is not None:
            # This is a valid cycle
            max_arr = max(edge_index[(path[i], path[(i+1) % n])] for i in range(n))
            if max_arr < best_score:
                best_score = max_arr
                best_cycle = path[:]
    
    # If no greedy solution found, use a simple cycle
    if best_cycle is None:
        best_cycle = list(range(n))
        best_score = max(edge_index[(i, (i+1) % n)] for i in range(n))
    
    # Local search: 2-opt style moves on the cycle
    # A move: reverse a segment of the cycle
    # Or swap two non-adjacent edges
    
    def cycle_score(cycle):
        """Compute max arrival index of edges in the cycle."""
        max_arr = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            arr = edge_index[(u, v)]
            if arr > max_arr:
                max_arr = arr
        return max_arr
    
    # 2-opt: try reversing a segment
    def two_opt(cycle, max_iters=1000):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for i in range(n):
                if time.time() > deadline:
                    return cycle
                for j in range(i + 2, n):
                    if j == (i + n - 1) % n:
                        continue
                    # Try reversing segment from i to j
                    new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                    new_score = cycle_score(new_cycle)
                    if new_score < best_score:
                        best_score = new_score
                        cycle = new_cycle
                        improved = True
                        break
                if improved:
                    break
        return cycle
    
    # Or-opt: move a segment of 2-3 vertices
    def or_opt(cycle, max_iters=1000):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for seg_len in [2, 3]:
                if time.time() > deadline:
                    return cycle
                for i in range(n):
                    # Extract segment of length seg_len starting at i
                    seg = [cycle[(i + k) % n] for k in range(seg_len)]
                    # Remove segment
                    remaining = [cycle[k] for k in range(n) if k != (i + k) % n]
                    # Wait, this is wrong. Let me redo.
                    # Actually, let's just try all possible insertions
                    pass
        return cycle
    
    # Simpler: use simulated annealing on permutations
    import numpy as np
    
    def simulated_annealing(cycle, max_iters=50000):
        nonlocal best_score, best_cycle
        current = cycle[:]
        current_score = cycle_score(current)
        T = 100.0
        cooling = 0.9999
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if it % 10000 == 0 and T < 1:
                T = 1.0
            
            # Random move: swap two non-adjacent positions
            i, j = np.random.randint(0, n, 2)
            if abs(i - j) <= 1 or abs(i - j) == n - 1:
                continue
            
            new_cycle = current[:]
            new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]
            new_score = cycle_score(new_cycle)
            
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current = new_cycle
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_cycle = current[:]
            
            T *= cooling
        
        return current
    
    # Run local search
    best_cycle = two_opt(best_cycle, max_iters=200)
    
    # Run simulated annealing
    if time.time() < deadline:
        sa_cycle = simulated_annealing(best_cycle, max_iters=100000)
    
    # Final check
    if best_cycle is not None:
        return best_cycle
    
    # Fallback
    return list(range(n))
