"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    
    def check_c4(adj, n):
        """Check if graph has a C4. Returns True if C4-free."""
        for i in range(n):
            ni = adj[i]
            for j in ni:
                if j <= i:
                    continue
                # Check common neighbors of i and j
                common = set(ni) & set(adj[j])
                if len(common) >= 2:
                    return False
        return True
    
    def try_greedy(n, seed):
        """Try to build a C4-free graph on n vertices with min degree >= n-39."""
        rng = random.Random(seed)
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
        
        # Start with a random ordering
        order = list(range(n))
        rng.shuffle(order)
        
        adj = [set() for _ in range(n)]
        
        # Try to add edges greedily
        # For each vertex, try to connect to as many other vertices as possible
        # without creating a C4
        
        for u in order:
            # Get potential neighbors
            candidates = [v for v in range(n) if v != u and v not in adj[u]]
            rng.shuffle(candidates)
            
            for v in candidates:
                if len(adj[u]) >= n - 1:
                    break
                # Check if adding u-v creates a C4
                # A C4 is created if u and v have at least 2 common neighbors
                common = adj[u] & adj[v]
                if len(common) < 2:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Check minimum degree
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= target_deg and check_c4(adj, n):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        return None
    
    # Try n=47 first, then 46
    for n in [47, 46]:
        target_deg = n - 39
        max_attempts = 200 if n == 47 else 500
        
        for seed in range(max_attempts):
            if time.time() > deadline:
                break
            
            result = try_greedy(n, seed)
            if result is not None:
                return result
    
    # Fallback: return the trivial 39-vertex graph
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
