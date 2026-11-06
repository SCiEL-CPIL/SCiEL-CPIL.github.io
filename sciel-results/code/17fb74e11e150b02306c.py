"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 40
    best_edges = None
    best_count = 0
    
    def has_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6.
        Need a path u-a-b-c-d-v with all 6 vertices distinct.
        This means: exists a in N(u), d in N(v), a != d,
        and exists x in N(a) ∩ N(d) with x not in {u, v, a, d}.
        """
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        # Get list of neighbors
        a_list = [i for i in range(n) if (nu >> i) & 1]
        d_list = [i for i in range(n) if (nv >> i) & 1]
        
        for a in a_list:
            for d in d_list:
                if a == d:
                    continue
                common = adj[a] & adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if common != 0:
                    return True
        return False
    
    def build_greedy_random(seed_val):
        rng = random.Random(seed_val)
        adj = [0] * n
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_edges)
        
        edge_list = []
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not has_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_list.append([u, v])
        
        return edge_list
    
    # Try multiple random orderings
    for seed_val in range(1000):
        if time.time() > deadline - 5:
            break
        edges = build_greedy_random(seed_val)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            # Early termination if we hit the known best
            if best_count >= 105:
                break
    
    if best_edges is None:
        # Fallback: seed construction (8 copies of K5 = 80 edges)
        best_edges = []
        for k in range(8):
            best_edges += [[5*k + a, 5*k + b] for a in range(5) for b in range(a+1, 5)]
    
    return best_edges
# EVOLVE-BLOCK-END
