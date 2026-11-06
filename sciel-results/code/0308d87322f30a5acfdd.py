import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def check_c4_bitmask(adj, n, u, v):
        """Check if adding edge (u,v) creates a C4 using bitmasks."""
        # A C4 u-x-y-v-u exists iff N(u) ∩ N(v) has size >= 2
        # OR there exist x in N(u), y in N(v) with x!=y and x~y
        # Actually simpler: after adding (u,v), check if any pair of common neighbors exists
        # But we can check: does there exist x in N(u) \ {v} and y in N(v) \ {u} such that x~y and x!=y?
        # This is equivalent to: (N(u) ∩ N(v)) has size >= 2, OR 
        # there exists x in N(u), y in N(v) with x!=y, x~y, and x not in N(v), y not in N(u)
        
        # More efficient: iterate over smaller neighbor set
        nu = adj[u]
        nv = adj[v]
        
        if nu == 0 and nv == 0:
            return False
            
        # Check if any x in N(u) has a neighbor y in N(v) where y != x
        # For each x in N(u), check if N(x) & (N(v) \ {x}) is non-empty
        if nu < nv:
            # Iterate over bits in nu
            x_bits = nu
            while x_bits:
                lb = x_bits & (-x_bits)
                x = lb.bit_length() - 1
                x_bits ^= lb
                # Check if N(x) intersects (N(v) \ {x})
                if adj[x] & (nv & ~(1 << x)):
                    return True
        else:
            # Iterate over bits in nv
            y_bits = nv
            while y_bits:
                lb = y_bits & (-y_bits)
                y = lb.bit_length() - 1
                y_bits ^= lb
                # Check if N(y) intersects (N(u) \ {y})
                if adj[y] & (nu & ~(1 << y)):
                    return True
        return False
    
    def try_build(n, min_deg_req, time_limit, seed):
        rng = random.Random(seed)
        adj = [0] * n  # adj[i] is bitmask of neighbors of i
        
        def add_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        def remove_edge(u, v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        def get_min_deg():
            return min(bin(adj[i]).count('1') for i in range(n))
            
        def get_deg(i):
            return bin(adj[i]).count('1')
        
        # Phase 1: Greedy addition - try to add edges that don't create C4
        # Start with random ordering
        vertices = list(range(n))
        rng.shuffle(vertices)
        
        for u in vertices:
            # Try all possible v > u in random order
            candidates = [v for v in range(u+1, n)]
            rng.shuffle(candidates)
            for v in candidates:
                if not check_c4_bitmask(adj, n, u, v):
                    add_edge(u, v)
        
        # Phase 2: Local search to maximize minimum degree
        t0 = time.time()
        stuck_counter = 0
        last_min_deg = get_min_deg()
        
        while time.time() - t0 < time_limit:
            if get_min_deg() >= min_deg_req:
                return adj
                
            # Find vertex with minimum degree
            min_deg = n
            target = 0
            for i in range(n):
                d = bin(adj[i]).count('1')
                if d < min_deg:
                    min_deg = d
                    target = i
            
            # Try to add an edge to target
            # Get non-neighbors of target
            non_neighbors = []
            for v in range(n):
                if v != target and not (adj[target] & (1 << v)):
                    non_neighbors.append(v)
            rng.shuffle(non_neighbors)
            
            added = False
            for v in non_neighbors:
                if not check_c4_bitmask(adj, n, target, v):
                    add_edge(target, v)
                    added = True
                    stuck_counter = 0
                    break
            
            if not added:
                stuck_counter += 1
                if stuck_counter > 30:
                    # Try to remove an edge to unblock
                    # Prefer edges not incident to target
                    edges_to_remove = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if (adj[u] & (1 << v)) and u != target and v != target:
                                edges_to_remove.append((u, v))
                    
                    if edges_to_remove:
                        u, v = rng.choice(edges_to_remove)
                        remove_edge(u, v)
                        stuck_counter = 0
                    else:
                        # Try removing an edge incident to target
                        neighbors = []
                        for v in range(n):
                            if v != target and (adj[target] & (1 << v)):
                                neighbors.append(v)
                        if neighbors:
                            v = rng.choice(neighbors)
                            remove_edge(target, v)
                            stuck_counter = 0
                        else:
                            break
                        
        return None
    
    def verify_graph(n, adj):
        """Verify the graph is C4-free and has sufficient minimum degree."""
        # Check minimum degree
        for i in range(n):
            if bin(adj[i]).count('1') < n - 39:
                return False
        
        # Check C4-free: no two vertices have 2+ common neighbors
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] & (1 << v)) and (adj[v] & (1 << u)):
                    common = adj[u] & adj[v]
                    if bin(common).count('1') >= 2:
                        return False
        return True
    
    best_result = None
    
    # Try n from 47 down
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate time based on n
        time_budget = 18.0
        if n <= 45:
            time_budget = 8.0
            
        attempts = 6
        success = False
        
        for i in range(attempts):
            if time.time() - start_time > deadline:
                break
            seed = 1000 + n * 100 + i
            t_limit = time_budget / attempts
            res = try_build(n, min_deg_req, t_limit, seed)
            if res is not None:
                # Verify
                if verify_graph(n, res):
                    edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if res[u] & (1 << v):
                                edges.append([u, v])
                    best_result = {'n': n, 'edges': edges}
                    success = True
                    break
        if success:
            return best_result
            
    # Fallback
    if best_result is None:
        return {'n': 3, 'edges': [[0,1],[1,2],[0,2]]}
        
    return best_result
