import time
import random
import math

def construct():
    n = 16
    m = 65
    start_time = time.time()
    deadline = start_time + 105  # Safety margin
    
    # Bitmask-based representation for fast triangle detection
    # adj[u] is a bitmask of neighbors of u
    # deg[u] is degree of u
    
    def build_adj_deg(edges_list):
        adj = [0] * n
        deg = [0] * n
        for u, v in edges_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def calc_max_tri_sum(adj, deg):
        """Calculate max degree sum over all triangles using bitmasks."""
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Common neighbors of u and v with index > v
                    common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                    while common:
                        lb = common & (-common)
                        w = lb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lb
        return max_sum
    
    def calc_qubo_score(adj, deg):
        """
        QUBO-inspired smooth surrogate: minimize sum of squared triangle degree sums.
        This encourages uniform low-degree distribution across all triangles.
        Also returns the max for constraint checking.
        """
        total_sq = 0
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                    while common:
                        lb = common & (-common)
                        w = lb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        total_sq += s * s
                        if s > max_sum:
                            max_sum = s
                        common ^= lb
        return max_sum, total_sq
    
    def add_edge(adj, deg, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
    
    def remove_edge(adj, deg, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1
    
    # Build initial graph: K_{8,8} + edge (0,1)
    adj = [0] * n
    deg = [0] * n
    for u in range(8):
        for v in range(8, 16):
            add_edge(adj, deg, u, v)
    add_edge(adj, deg, 0, 1)
    
    cur_max, cur_sq = calc_qubo_score(adj, deg)
    best_max = cur_max
    best_adj = adj[:]
    best_deg = deg[:]
    
    # Simulated annealing with QUBO-inspired objective
    # Primary: minimize max triangle sum
    # Secondary: minimize sum of squared triangle sums (smooth surrogate)
    T = 5.0
    cool_rate = 0.9995
    min_T = 0.05
    
    while time.time() < deadline:
        # Pick random edge to remove
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if not (adj[u] & (1 << v)):
            continue
        
        # Pick random non-edge to add
        x = random.randint(0, n - 1)
        y = random.randint(0, n - 1)
        if x == y:
            continue
        if adj[x] & (1 << y):
            continue
        
        # Perform move
        remove_edge(adj, deg, u, v)
        add_edge(adj, deg, x, y)
        
        new_max, new_sq = calc_qubo_score(adj, deg)
        
        # Acceptance: prefer lower max, then lower sum of squares
        if new_max < cur_max:
            accept = True
        elif new_max == cur_max:
            if new_sq < cur_sq:
                accept = True
            else:
                # Accept with probability based on combined penalty
                delta = (new_max - cur_max) * 10 + (new_sq - cur_sq) / 100.0
                accept = random.random() < math.exp(-delta / T)
        else:
            # new_max > cur_max: accept with small probability
            delta = (new_max - cur_max) * 10 + (new_sq - cur_sq) / 100.0
            accept = random.random() < math.exp(-delta / T)
        
        if accept:
            cur_max = new_max
            cur_sq = new_sq
            if cur_max < best_max:
                best_max = cur_max
                best_adj = adj[:]
                best_deg = deg[:]
        else:
            # Revert
            remove_edge(adj, deg, x, y)
            add_edge(adj, deg, u, v)
        
        T *= cool_rate
        if T < min_T:
            T = min_T
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            mask ^= lb
    
    return {'n': n, 'edges': edges}


