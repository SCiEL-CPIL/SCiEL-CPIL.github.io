import time
import random
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def construct():
    """Construct a K3,3-free graph on 20 vertices maximizing edges."""
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin within 120s limit
    
    n = 20
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 190
    
    # Precompute: for each pair of vertices, which edges have them as endpoints
    # For K3,3 detection: we need to check if there exist 3 vertices A and 3 vertices B
    # such that all 9 edges between A and B are present.
    
    # Key insight: A K3,3 subgraph exists iff there are 3 vertices whose common neighbors
    # include 3 vertices that are mutually connected in the bipartite sense... 
    # Actually, K3,3 means: there exist sets A={a1,a2,a3} and B={b1,b2,b3} such that
    # all edges (ai, bj) for i,j in {1,2,3} are present.
    
    # Efficient check: For each pair of vertices (u,v), compute common neighbors.
    # If any 3 vertices have >= 3 common neighbors, AND those 3 common neighbors are 
    # all connected to all 3 of the original vertices... wait, that's automatically true
    # if they're common neighbors.
    
    # So: K3,3 exists iff there exist 3 vertices {u,v,w} such that |N(u) ∩ N(v) ∩ N(w)| >= 3.
    
    # Strategy: Simulated annealing with edge additions/removals and swaps.
    
    def is_valid(adj):
        """Check if the graph represented by adjacency bitmasks is K3,3-free.
        adj[v] is a bitmask of neighbors of vertex v.
        """
        # For each triple of vertices, check if they have >= 3 common neighbors
        # This is O(n^3 * n/word_size) which for n=20 is fine
        for u in range(n):
            for v in range(u+1, n):
                common_uv = adj[u] & adj[v]
                for w in range(v+1, n):
                    common_uvw = common_uv & adj[w]
                    # Count bits
                    if common_uvw & (common_uvw - 1) and (common_uvw & (common_uvw - 1)) & (common_uvw - 1):
                        # At least 3 bits set
                        return False
        return True
    
    def count_edges(adj):
        """Count edges in adjacency bitmask representation."""
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2
    
    def add_edge(adj, u, v):
        """Add edge (u,v), return new adj or None if already present."""
        if adj[u] & (1 << v):
            return None
        new_adj = adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        return new_adj
    
    def remove_edge(adj, u, v):
        """Remove edge (u,v), return new adj or None if not present."""
        if not (adj[u] & (1 << v)):
            return None
        new_adj = adj[:]
        new_adj[u] &= ~(1 << v)
        new_adj[v] &= ~(1 << u)
        return new_adj
    
    # Greedy initialization: add edges in random order if valid
    def greedy_init(seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = all_edges[:]
        rng.shuffle(edges)
        for (u, v) in edges:
            new_adj = add_edge(adj, u, v)
            if new_adj is not None and is_valid(new_adj):
                adj = new_adj
        return adj
    
    # Try multiple greedy initializations
    best_adj = None
    best_count = 0
    
    for seed in range(50):
        if time.time() > deadline - 60:
            break
        adj = greedy_init(seed)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Simulated annealing refinement
    if best_adj is not None:
        adj = best_adj[:]
        cur_count = best_count
        best_sa_adj = adj[:]
        best_sa_count = cur_count
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        n_iter = 50000
        
        for it in range(n_iter):
            if time.time() > deadline - 5:
                break
            
            T = T_start * (T_end / T_start) ** (it / n_iter)
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.4:
                # Try adding a random edge
                u, v = random.choice(all_edges)
                new_adj = add_edge(adj, u, v)
                if new_adj is None:
                    continue
                if is_valid(new_adj):
                    adj = new_adj
                    cur_count += 1
                    if cur_count > best_sa_count:
                        best_sa_count = cur_count
                        best_sa_adj = adj[:]
                else:
                    # Reject (adding an invalid edge)
                    pass
            elif move_type < 0.7:
                # Try removing a random edge
                u, v = random.choice(all_edges)
                new_adj = remove_edge(adj, u, v)
                if new_adj is None:
                    continue
                # Always accept removal (or accept with some probability)
                delta = -1
                if delta < 0 and random.random() > exp(-delta / T):
                    adj = new_adj
                    cur_count -= 1
            else:
                # Swap: remove one edge, add another
                u1, v1 = random.choice(all_edges)
                u2, v2 = random.choice(all_edges)
                new_adj = remove_edge(adj, u1, v1)
                if new_adj is None:
                    continue
                new_adj2 = add_edge(new_adj, u2, v2)
                if new_adj2 is None:
                    continue
                if is_valid(new_adj2):
                    # Net change is 0 edges, but structure changes
                    # Accept with some probability to escape local optima
                    if random.random() < 0.5 or random.random() < exp(0/T):
                        adj = new_adj2
                        # cur_count unchanged
                else:
                    pass
        
        if best_sa_count > best_count:
            best_count = best_sa_count
            best_adj = best_sa_adj[:]
    
    # Final local search: try all edge additions
    if best_adj is not None:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for (u, v) in all_edges:
                if best_adj[u] & (1 << v):
                    continue
                new_adj = add_edge(best_adj, u, v)
                if new_adj is not None and is_valid(new_adj):
                    best_adj = new_adj
                    best_count += 1
                    improved = True
                    break
    
    # Convert adjacency bitmasks to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges

def exp(x):
    """Safe exponential for simulated annealing."""
    if x < -50:
        return 0.0
    if x > 50:
        return 1e21
    import math
    return math.exp(x)

# EVOLVE-BLOCK-END
