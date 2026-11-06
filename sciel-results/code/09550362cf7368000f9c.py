import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 138  # safety margin for 150s kill
    
    best_n = 39
    best_edges = []
    
    def make_bitset(n):
        # return list of n integers, each a bitmask of neighbors
        return [0] * n
    
    def c4_free_check(adj_bit, n):
        """Check if graph has no C4 using bitmasks."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj_bit[i] & (1 << j):
                    common = adj_bit[i] & adj_bit[j]
                    if common & (common - 1):  # more than one bit set
                        return False
        return True
    
    def demand_greedy(n, min_deg_req, adj_init=None):
        """
        Demand-driven greedy: repeatedly pick the lowest-degree vertex,
        connect it to the lowest-degree safe neighbor.
        Returns (adj_bit, deg, success).
        """
        if adj_init is not None:
            adj = list(adj_init)
        else:
            adj = [0] * n
        deg = [bin(adj[i]).count('1') for i in range(n)]
        
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline:
                return adj, deg, False
            
            # Find vertex with minimum degree
            u = 0
            for i in range(1, n):
                if deg[i] < deg[u]:
                    u = i
            
            if deg[u] >= min_deg_req:
                return adj, deg, True
            
            # Find safe candidates for u
            best_v = -1
            best_v_deg = n + 1
            for v in range(n):
                if v == u:
                    continue
                if adj[u] & (1 << v):
                    continue
                # Check C4: for each a in N(u), check if any b in N(v) is adjacent to a
                safe = True
                for a in range(n):
                    if not (adj[u] & (1 << a)):
                        continue
                    # Check if any neighbor of v is also neighbor of a
                    if adj[v] & adj[a]:
                        safe = False
                        break
                if safe:
                    if deg[v] < best_v_deg:
                        best_v_deg = deg[v]
                        best_v = v
            
            if best_v == -1:
                return adj, deg, False
            
            # Add edge (u, best_v)
            adj[u] |= (1 << best_v)
            adj[best_v] |= (1 << u)
            deg[u] += 1
            deg[best_v] += 1
        
        return adj, deg, min(deg) >= min_deg_req
    
    def random_init(n, p):
        """Create random graph G(n,p)."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def er_init_then_fix(n, p, min_deg_req):
        """Initialize with G(n,p), then use demand greedy to fix degrees."""
        adj = random_init(n, p)
        # First, remove C4s by deleting edges
        # Then use demand greedy to add edges
        # Actually, let's just use demand greedy from the init
        adj, deg, success = demand_greedy(n, min_deg_req, adj_init=adj)
        return adj, deg, success
    
    # Try n=47 first (theoretical max), then 46, 45, ...
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
        
        min_deg_req = n - 39
        
        # Strategy 1: Pure demand-driven greedy with random restarts
        success = False
        trials = 0
        max_trials = 5000
        
        while not success and trials < max_trials:
            if time.time() > deadline:
                break
            trials += 1
            
            adj, deg, ok = demand_greedy(n, min_deg_req)
            if ok:
                # Verify C4-free
                if c4_free_check(adj, n):
                    success = True
                    break
        
        if not success:
            # Strategy 2: ER init + demand greedy
            for p in [0.15, 0.2, 0.25, 0.1, 0.3]:
                if time.time() > deadline:
                    break
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    adj, deg, ok = er_init_then_fix(n, p, min_deg_req)
                    if ok:
                        if c4_free_check(adj, n):
                            success = True
                            break
                if success:
                    break
        
        if success:
            # Extract edges
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            
            # Final verification
            if c4_free_check(adj, n) and min(deg) >= min_deg_req:
                best_n = n
                best_edges = edges
                return {'n': n, 'edges': best_edges}
    
    # Fallback: return best found so far
    return {'n': best_n, 'edges': best_edges}
