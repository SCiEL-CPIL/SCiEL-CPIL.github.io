"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    n = 16
    target_edges = 65
    
    # Strategy: Simulated Annealing minimizing sum of squared triangle degree sums.
    # This acts as a QUBO-like penalty function to enforce uniform low degrees in triangles.
    
    def get_triangles_and_sums(adj):
        """
        Efficiently compute all triangles and their degree sums.
        adj: numpy array n x n, 0/1 symmetric.
        Returns: list of (u, v, w) tuples and list of degree sums.
        """
        triangles = []
        sums = []
        # Use bitmasks for speed? n=16 fits in 64-bit int.
        # But numpy is fine for n=16.
        # To be faster, we can use integer bitmasks.
        # Let's stick to numpy for clarity but optimize loops.
        
        # Precompute degrees
        degrees = adj.sum(axis=1)
        
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    # Common neighbors w > v
                    # adj[u, v+1:] & adj[v, v+1:]
                    # We need w such that adj[u,w]=1 and adj[v,w]=1
                    # w must be > v to avoid duplicates
                    # Let's use numpy intersection
                    # neighbors_u = np.where(adj[u, v+1:] == 1)[0] + v + 1
                    # neighbors_v = np.where(adj[v, v+1:] == 1)[0] + v + 1
                    # common = np.intersect1d(neighbors_u, neighbors_v)
                    
                    # Simpler loop for small n
                    for w in range(v+1, n):
                        if adj[u, w] and adj[v, w]:
                            triangles.append((u, v, w))
                            s = degrees[u] + degrees[v] + degrees[w]
                            sums.append(s)
        return triangles, sums

    def get_objective(adj):
        """
        Objective: Minimize sum of squares of triangle degree sums.
        Also track max for final reporting.
        """
        triangles, sums = get_triangles_and_sums(adj)
        if not sums:
            return 0, 0
        sum_sq = sum(s*s for s in sums)
        max_sum = max(sums)
        return sum_sq, max_sum

    # Initial state: K_{8,8} + edge (0,1)
    adj = np.zeros((n, n), dtype=int)
    # K_{8,8} between 0-7 and 8-15
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    # Add edge (0,1)
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    current_edges = int(adj.sum() // 2)
    assert current_edges == 65, f"Initial edges {current_edges} != 65"
    
    current_obj, current_max = get_objective(adj)
    best_obj = current_obj
    best_max = current_max
    best_adj = adj.copy()
    
    # SA Parameters
    # We want to run for a reasonable amount of time within 120s limit.
    # n=16 is small. Let's try 20000 steps.
    T = 20000
    t_start = 100.0
    t_end = 1.0
    cooling = (t_end / t_start) ** (1.0 / T)
    
    # Pre-compute list of all possible edges for faster random selection?
    # There are 16*15/2 = 120 edges.
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for step in range(T):
        t = t_start * (cooling ** step)
        
        # Move: Swap an existing edge with a non-existing edge
        # 1. Pick random existing edge (a,b)
        # 2. Pick random non-existing edge (c,d)
        
        # To pick random existing edge efficiently:
        # We can maintain a list of edges, but updating it is O(E).
        # Or just pick random u,v until we find an edge?
        # With 65/120 edges, probability of hitting an edge is ~0.54.
        # Probability of hitting non-edge is ~0.46.
        
        # Let's pick random pairs.
        attempts = 0
        while attempts < 10:
            u1 = random.randint(0, n-1)
            v1 = random.randint(0, n-1)
            if u1 == v1:
                continue
            if u1 > v1:
                u1, v1 = v1, u1
            if adj[u1, v1] == 1:
                break
            attempts += 1
        else:
            continue # Failed to find an edge, skip step
            
        # Found edge (u1, v1) to remove
        
        attempts = 0
        while attempts < 10:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if adj[u2, v2] == 0:
                # Ensure (u2,v2) is not (u1,v1) - guaranteed by one being 1 and other 0
                break
            attempts += 1
        else:
            continue # Failed to find a non-edge, skip step
            
        # Found non-edge (u2, v2) to add
        
        # Apply move
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        new_obj, new_max = get_objective(adj)
        
        # Acceptance criterion based on objective (sum of squares)
        delta = new_obj - current_obj
        accepted = False
        
        if delta < 0:
            accepted = True
        else:
            # Metropolis criterion
            # Delta is integer (sum of squares of integers)
            if delta == 0:
                accepted = True
            else:
                # Avoid overflow in exp for large delta/t
                if delta / t > 50:
                    prob = 0.0
                else:
                    prob = np.exp(-delta / t)
                if random.random() < prob:
                    accepted = True
        
        if accepted:
            current_obj = new_obj
            current_max = new_max
            if new_obj < best_obj or (new_obj == best_obj and new_max < best_max):
                best_obj = new_obj
                best_max = new_max
                best_adj = adj.copy()
        else:
            # Revert move
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v] == 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
