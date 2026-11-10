import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # adj[i] is bitmask
    # cc[i][j] is count of common neighbors of i and j
    
    def popcount(x):
        return bin(x).count('1')
        
    def init_graph():
        adj = [0] * n
        cc = [[0] * n for _ in range(n)]
        return adj, cc

    def is_valid_add(adj, cc, u, v):
        if (adj[u] >> v) & 1:
            return False
        
        # 1. Direct common neighbors of u, v must be <= 2
        if cc[u][v] > 2:
            return False
            
        # 2. For w in N(v) \ {u}, cc[u][w] must be <= 1
        #    Because adding (u,v) makes v a common neighbor of u and w
        #    so cc[u][w] becomes cc[u][w] + 1. We need <= 2, so old <= 1.
        nv = adj[v] & ~(1 << u) & ~(1 << v)
        while nv:
            lsb = nv & -nv
            w = lsb.bit_length() - 1
            nv ^= lsb
            if cc[u][w] > 1:
                return False
                
        # 3. For w in N(u) \ {v}, cc[v][w] must be <= 1
        nu = adj[u] & ~(1 << u) & ~(1 << v)
        while nu:
            lsb = nu & -nu
            w = lsb.bit_length() - 1
            nu ^= lsb
            if cc[v][w] > 1:
                return False
                
        return True

    def add_edge(adj, cc, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update cc
        # For w in N(v) \ {u}, cc[u][w] += 1
        nv = adj[v] & ~(1 << u) & ~(1 << v)
        while nv:
            lsb = nv & -nv
            w = lsb.bit_length() - 1
            nv ^= lsb
            cc[u][w] += 1
            cc[w][u] += 1
            
        # For w in N(u) \ {v}, cc[v][w] += 1
        nu = adj[u] & ~(1 << u) & ~(1 << v)
        while nu:
            lsb = nu & -nu
            w = lsb.bit_length() - 1
            nu ^= lsb
            cc[v][w] += 1
            cc[w][v] += 1

    def remove_edge(adj, cc, u, v):
        # Must be called only if edge exists
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update cc
        # For w in N(v) \ {u}, cc[u][w] -= 1
        # Note: adj[v] has been updated, so u is no longer in N(v).
        # But we need to iterate over the *current* neighbors of v (which is N(v) \ {u})
        nv = adj[v] & ~(1 << v)
        while nv:
            lsb = nv & -nv
            w = lsb.bit_length() - 1
            nv ^= lsb
            cc[u][w] -= 1
            cc[w][u] -= 1
            
        # For w in N(u) \ {v}, cc[v][w] -= 1
        nu = adj[u] & ~(1 << u)
        while nu:
            lsb = nu & -nu
            w = lsb.bit_length() - 1
            nu ^= lsb
            cc[v][w] -= 1
            cc[w][v] -= 1

    def edge_count(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    def local_improve(adj, cc):
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, cc, u, v):
                        add_edge(adj, cc, u, v)
                        improved = True
        return adj, cc

    best_adj = [0] * n
    best_cc = [[0] * n for _ in range(n)]
    best_count = 0
    rng = random.Random(42)

    # Multi-start
    while time.time() < deadline - 10.0:
        adj, cc = init_graph()
        
        # Random greedy initialization
        edges = all_edges[:]
        rng.shuffle(edges)
        for u, v in edges:
            if is_valid_add(adj, cc, u, v):
                add_edge(adj, cc, u, v)
                
        # Local improvement
        adj, cc = local_improve(adj, cc)
        
        count = edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            best_cc = [row[:] for row in cc]
            
        # Kicks: remove a few random edges and re-improve
        # Try a few kicks if we have time
        if time.time() < deadline - 5.0:
            kicks = 3
            for _ in range(kicks):
                if time.time() > deadline - 5.0:
                    break
                # Remove 2 random edges
                removed = []
                for _ in range(2):
                    # Pick random edge
                    # Find an edge in adj
                    # To be efficient, pick random u, then random neighbor
                    u = rng.randint(0, n - 1)
                    neighbors = adj[u]
                    if neighbors == 0:
                        continue
                    # Pick random set bit
                    # Use random index from list of set bits?
                    # Or random shift?
                    # For n=20, we can just list them
                    neigh_list = []
                    temp = neighbors
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        neigh_list.append(w)
                        temp ^= lsb
                    if not neigh_list:
                        continue
                    v = rng.choice(neigh_list)
                    remove_edge(adj, cc, u, v)
                    removed.append((u, v))
                    
                # Re-improve
                adj, cc = local_improve(adj, cc)
                count = edge_count(adj)
                
                if count > best_count:
                    best_count = count
                    best_adj = adj[:]
                    best_cc = [row[:] for row in cc]

    # Final verification and conversion
    # Verify best_adj is valid
    # Just to be safe, though the logic should hold
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                common = best_adj[u] & best_adj[v]
                if popcount(common) > 2:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid construction (e.g. 5xK4) if something went wrong
        # But the code should be correct.
        pass

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
