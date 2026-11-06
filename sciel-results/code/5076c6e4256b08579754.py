import random
import numpy as np

def construct():
    n = 16
    # Start with a known valid graph: Mycielski of C5 (11 vertices) + 5 isolated
    # But we want to maximize edges.
    # Let's start with a random triangle-free graph or the seed's graph.
    
    # Helper: Check if graph is triangle-free
    def is_triangle_free(edges_set):
        # Build adjacency list for speed if needed, or use set
        adj = {i: set() for i in range(n)}
        for u, v in edges_set:
            if u > v: u, v = v, u
            adj[u].add(v)
            adj[v].add(u)
        
        for u in range(n):
            for v in adj[u]:
                if v <= u: continue
                # Check common neighbors of u and v
                if adj[u].intersection(adj[v]):
                    return False
        return True

    # Helper: Check if graph is 3-colorable
    def is_3_colorable(edges_set):
        adj = {i: set() for i in range(n)}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            for c in range(3):
                if all(colors[nb] != c for nb in adj[idx]):
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
            return False
        
        return backtrack(0)

    # Initial State: The seed program's graph
    # Re-construct seed graph to ensure valid start
    edges_grotzsch = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 6], [1, 5],
        [1, 7], [2, 6],
        [2, 8], [3, 7],
        [3, 9], [4, 8],
        [4, 5], [0, 9],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ]
    # Add vertices 11-15 as isolated initially
    current_edges = set(tuple(sorted(e)) for e in edges_grotzsch)
    
    # We will use Simulated Annealing
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.999
    T = T_start
    
    best_edges = current_edges
    best_score = len(current_edges)
    # Score: number of edges if valid, else -infinity (or large penalty)
    
    def get_score(edges_set):
        if not is_triangle_free(edges_set):
            return -1000
        if is_3_colorable(edges_set):
            return -1000
        return len(edges_set)

    current_score = get_score(current_edges)
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # To make SA efficient, we need fast constraint checks.
    # is_triangle_free on 16 vertices is O(E * (deg(u)+deg(v))) which is small.
    # is_3_colorable is the bottleneck.
    
    # Optimization: Only check 3-colorability if triangle-free.
    # Also, 3-colorability check can be cached? No, state changes.
    
    # Let's run SA
    import time
    start_time = time.time()
    max_time = 100.0 # Leave buffer for 120s limit
    
    while time.time() - start_time < max_time:
        # Generate neighbor
        # 1. Add random edge
        # 2. Remove random edge
        # 3. Swap edge
        
        r = random.random()
        if r < 0.6:
            # Add edge
            e = random.choice(all_edges)
            if e not in current_edges:
                new_edges = current_edges | {e}
            else:
                new_edges = current_edges
        elif r < 0.8:
            # Remove edge
            if current_edges:
                e = random.choice(list(current_edges))
                new_edges = current_edges - {e}
            else:
                new_edges = current_edges
        else:
            # Swap: remove one, add one
            if current_edges:
                e_rem = random.choice(list(current_edges))
                e_add = random.choice(all_edges)
                if e_add in current_edges and e_add != e_rem:
                    # Try to find a different add
                    e_add = random.choice([x for x in all_edges if x not in current_edges] or [e_rem])
                new_edges = (current_edges - {e_rem}) | {e_add}
            else:
                new_edges = current_edges

        new_score = get_score(new_edges)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0:
            current_edges = new_edges
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_edges = current_edges
        else:
            if T > T_end:
                accept_prob = np.exp(delta / T)
                if random.random() < accept_prob:
                    current_edges = new_edges
                    current_score = new_score
        
        T *= cooling_rate
        if T < T_end:
            T = T_end

    return [list(e) for e in best_edges]
