import random
import time
import sys

def construct():
    deadline = time.time() + 140  # 10s margin before 150s kill

    def find_c4(adj, n):
        """Find a C4 in the graph. Returns (u,v,w,x) or None."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    if bin(common).count('1') >= 2:
                        bits = []
                        c = common
                        while c:
                            lsb = c & -c
                            idx = lsb.bit_length() - 1
                            bits.append(idx)
                            c -= lsb
                            if len(bits) == 2:
                                break
                        w, x = bits
                        return (u, v, w, x)
        return None

    def can_add_edge(adj, u, v, n):
        """Check if adding edge (u,v) creates a C4."""
        N_u = adj[u]
        N_v = adj[v]
        # Check if any neighbor of u is connected to any neighbor of v
        c = N_u
        while c:
            lsb = c & -c
            a = lsb.bit_length() - 1
            c -= lsb
            if adj[a] & N_v:
                return False
        return True

    def is_c4_free(adj, n):
        """Check if graph is C4-free."""
        return find_c4(adj, n) is None

    def compute_deg(adj, n):
        """Compute degree of each vertex."""
        return [bin(adj[i]).count('1') for i in range(n)]

    def try_construct(n, min_deg, seed=42, max_restarts=200):
        """Try to construct a C4-free graph on n vertices with min degree >= min_deg."""
        random.seed(seed)
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with random sparse graph
            adj = [0] * n
            # Use a density that gives reasonable initial structure
            # For C4-free, we want sparse but connected enough
            p = 0.12
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Step 1: Remove C4s
            changed = True
            while changed and time.time() < deadline:
                changed = False
                c4 = find_c4(adj, n)
                if c4:
                    u, v, w, x = c4
                    # Remove a random edge from the C4
                    edges_to_remove = [(u, w), (w, v), (v, x), (x, u)]
                    e_u, e_v = random.choice(edges_to_remove)
                    adj[e_u] &= ~(1 << e_v)
                    adj[e_v] &= ~(1 << e_u)
                    changed = True
            
            if not is_c4_free(adj, n):
                continue
                
            deg = compute_deg(adj, n)
            
            # Step 2: Boost degree using local search with kicks
            success = False
            for _ in range(200):  # Max iterations
                if time.time() > deadline:
                    break
                    
                if min(deg) >= min_deg:
                    success = True
                    break
                    
                # Find vertex with minimum degree
                v_min = deg.index(min(deg))
                
                # Try to add an edge from v_min to some u
                candidates = []
                for u in range(n):
                    if u != v_min and not (adj[v_min] >> u) & 1:
                        if can_add_edge(adj, v_min, u, n):
                            candidates.append(u)
                
                if candidates:
                    # Choose candidate that maximizes degree balance
                    # Prefer adding edge to a vertex with lower degree
                    best_u = min(candidates, key=lambda u: deg[u])
                    # Add edge
                    adj[v_min] |= (1 << best_u)
                    adj[best_u] |= (1 << v_min)
                    deg[v_min] += 1
                    deg[best_u] += 1
                else:
                    # Stuck: use kick strategy
                    # Remove a random edge incident to a high-degree vertex to create space
                    # Then try to restructure
                    high_deg_vertices = [v for v in range(n) if deg[v] > min_deg]
                    if not high_deg_vertices:
                        # No high degree vertices, try removing any edge
                        high_deg_vertices = list(range(n))
                    
                    v_kick = random.choice(high_deg_vertices)
                    # Find a neighbor to remove edge to
                    neighbors = []
                    c = adj[v_kick]
                    while c:
                        lsb = c & -c
                        idx = lsb.bit_length() - 1
                        neighbors.append(idx)
                        c -= lsb
                    
                    if neighbors:
                        u_remove = random.choice(neighbors)
                        adj[v_kick] &= ~(1 << u_remove)
                        adj[u_remove] &= ~(1 << v_kick)
                        deg[v_kick] -= 1
                        deg[u_remove] -= 1
                    else:
                        # Isolated vertex, try different restart
                        break
            
            if success and min(deg) >= min_deg:
                # Construct edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
        return None

    # Try n=47 first (theoretical max)
    res = try_construct(47, 8, seed=42, max_restarts=100)
    if res:
        return res
    
    # Try n=46
    res = try_construct(46, 7, seed=42, max_restarts=100)
    if res:
        return res
    
    # Try n=45
    res = try_construct(45, 6, seed=42, max_restarts=100)
    if res:
        return res
    
    # Fallback: construct a valid small graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}
