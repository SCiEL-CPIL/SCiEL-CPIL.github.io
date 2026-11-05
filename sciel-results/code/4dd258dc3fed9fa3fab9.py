import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)

    def is_c4_free(n, edges):
        """Check if graph has no C4 using adjacency sets."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & adj[j]:
                    common = adj[i] & adj[j]
                    if len(common) >= 2:
                        return False
        return True

    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg) if n > 0 else 0

    def validate(n, edges):
        if not is_c4_free(n, edges):
            return False
        return min_degree(n, edges) >= n - 39

    def fast_greedy_c4_free_min_deg(n, min_deg_target, seed_val=42):
        """Fast greedy construction prioritizing minimum degree."""
        rng = random.Random(seed_val)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Precompute all edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        def would_create_c4(u, v):
            # Check if adding (u,v) creates a C4
            # C4: u-a-v-b-u where a in adj[u], b in adj[v], a!=b, a!=v, b!=u, and a~b
            # Optimization: 
            # 1. If |N(u) cap N(v)| >= 2, C4 exists (u-a-v-b-u with a,b common)
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return True
            
            # 2. Check for a in N(u), b in N(v) such that a!=b, a!=v, b!=u, a~b
            # Since n is small, nested loop is fine
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[v]:
                    if b == u or b == a:
                        continue
                    if b in adj[a]:
                        return True
            return False

        # Greedy loop: always try to add an edge incident to the lowest degree vertex
        # If no such edge, try any edge
        for _ in range(len(all_edges) * 2): # Safety bound
            if time.time() > deadline - 10:
                break
            
            # Find min degree vertex
            degs = [len(adj[i]) for i in range(n)]
            min_deg = min(degs)
            if min_deg >= min_deg_target:
                return [[u, v] for u, v in edges]
            
            min_v = degs.index(min_deg)
            
            # Find valid neighbors for min_v
            candidates = []
            for v in range(n):
                if v == min_v or v in adj[min_v]:
                    continue
                if not would_create_c4(min_v, v):
                    candidates.append(v)
            
            if candidates:
                # Pick one, maybe the one with highest degree to balance?
                # Or random?
                v = rng.choice(candidates)
                adj[min_v].add(v)
                adj[v].add(min_v)
                edges.append((min_v, v))
            else:
                # No edge can be added to min_v
                # Try to add any edge between other vertices?
                # Or break?
                # If we can't add to min_v, we might be stuck.
                # Try to add a random valid edge elsewhere to change structure?
                # For now, break to try a new seed
                break
        
        degs = [len(adj[i]) for i in range(n)]
        if min(degs) >= min_deg_target:
            return [[u, v] for u, v in edges]
        return None

    # Try deterministic constructions first
    for n in [47, 46, 45]:
        if time.time() > deadline - 20:
            break
        
        min_deg_target = n - 39
        
        # Try multiple seeds for greedy
        found = None
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            edges = fast_greedy_c4_free_min_deg(n, min_deg_target, seed_val=seed)
            if edges is not None:
                if validate(n, edges):
                    found = edges
                    break
        
        if found is not None:
            return {'n': n, 'edges': found}

    # Fallback: try smaller n
    for n in range(44, 38, -1):
        if time.time() > deadline - 5:
            break
        min_deg_target = n - 39
        if min_deg_target < 1:
            return {'n': n, 'edges': []}
        
        for seed in range(10):
            if time.time() > deadline - 3:
                break
            edges = fast_greedy_c4_free_min_deg(n, min_deg_target, seed_val=seed)
            if edges is not None:
                if validate(n, edges):
                    return {'n': n, 'edges': edges}

    # Absolute fallback
    return {'n': 39, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
