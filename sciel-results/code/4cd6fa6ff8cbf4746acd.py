"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _creates_c4(adj, u, v):
    """Check if adding edge (u, v) creates a C4."""
    # A C4 is u-x-y-v-u.
    # This exists if there is an edge (x, y) such that x in adj[u] and y in adj[v].
    nu = adj[u]
    nv = adj[v]
    for x in nu:
        if x == v: continue
        for y in nv:
            if y == u: continue
            if x == y: continue
            if y in adj[x]:
                return True
    return False

def _greedy_construct(n, min_deg_required, time_limit=5):
    start_time = time.time()
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        adj = [set() for _ in range(n)]
        
        # Shuffle pairs to introduce randomness in tie-breaking
        pairs = all_pairs[:]
        random.shuffle(pairs)
        
        while True:
            if time.time() - start_time > time_limit:
                return None
                
            best_edge = None
            best_score = float('inf')
            best_u = -1
            best_v = -1
            
            for u, v in pairs:
                if v in adj[u]:
                    continue
                
                if _creates_c4(adj, u, v):
                    continue
                
                du = len(adj[u])
                dv = len(adj[v])
                score = min(du, dv)
                
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
                    best_u = u
                    best_v = v
                elif score == best_score:
                    # Randomly replace to diversify
                    if random.random() < 0.5:
                        best_edge = (u, v)
                        best_u = u
                        best_v = v
            
            if best_edge is None:
                break
            
            adj[best_u].add(best_v)
            adj[best_v].add(best_u)
            
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= min_deg_required:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
    return None

def construct():
    start_time = time.time()
    deadline = 110
    
    best_result = {'n': 39, 'edges': []}
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 5:
            break
            
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        result = _greedy_construct(n, min_deg_req, time_limit=remaining - 2)
        if result is not None:
            return result
            
    return best_result
# EVOLVE-BLOCK-END
