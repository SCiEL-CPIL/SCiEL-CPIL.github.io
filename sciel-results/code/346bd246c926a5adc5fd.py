import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def popcount(x):
        return bin(x).count('1')
        
    def has_k4(adj):
        n = len(adj)
        # adj is list of bitmasks
        # Count K4s
        # For each edge (u, v), count edges in N(u) & N(v)
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Count edges in common
                    # Iterate through bits in common
                    c = common
                    edges_in_common = 0
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        idx = (lsb.bit_length() - 1)
                        # Check if idx is connected to other nodes in common
                        # adj[idx] & common
                        # But we must not count the edge (idx, idx) which is 0
                        # And we count each edge twice
                        edges_in_common += popcount(adj[idx] & common)
                        c ^= lsb
                    k4_count += edges_in_common // 2
        return k4_count > 0

    def has_indep_6(adj):
        n = len(adj)
        # Backtracking for independent set of size 6
        # adj[i] is bitmask of neighbors of i
        
        # Precompute degrees to order vertices? 
        # Simple recursion with bitmask
        sys.setrecursionlimit(10000)
        
        def exists_indep_k(available, k):
            if k == 0:
                return True
            if popcount(available) < k:
                return False
            
            # Heuristic: Pick a vertex with highest degree in the induced subgraph?
            # Or just pick the first one?
            # Let's pick the first available vertex
            # To improve pruning, we can try to pick a vertex that removes many candidates
            # i.e., high degree in the remaining graph.
            
            # Get list of available vertices
            # This might be slow if done every step.
            # Alternative: just iterate
            v = -1
            best_v = -1
            max_deg = -1
            c = available
            while c:
                lsb = c & -c
                idx = (lsb.bit_length() - 1)
                # Degree of idx in available is popcount(adj[idx] & available)
                # But we want to maximize removal, so maximize degree
                deg = popcount(adj[idx] & available)
                if deg > max_deg:
                    max_deg = deg
                    best_v = idx
                c ^= lsb
            
            if best_v == -1:
                return False
                
            # Branch 1: Include best_v
            # New available: remove best_v and its neighbors
            new_avail = available & ~(adj[best_v] | (1 << best_v))
            if exists_indep_k(new_avail, k - 1):
                return True
            
            # Branch 2: Exclude best_v
            new_avail = available & ~(1 << best_v)
            if exists_indep_k(new_avail, k):
                return True
                
            return False
            
        return exists_indep_k((1 << n) - 1, 6)

    def count_violations(adj):
        # Returns (k4_count, has_indep_6)
        # We need counts for cost function
        # k4_count is as above
        # For indep 6, we just need True/False for boolean constraint, 
        # but for cost, we might want a soft count or just 0/1.
        # Let's use 0/1 for indep 6 to keep it simple, or count number of indep 6s?
        # Counting all indep 6s is hard. Let's just use 0/1.
        
        k4s = 0
        n = len(adj)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    c = common
                    edges_in_common = 0
                    while c:
                        lsb = c & -c
                        idx = (lsb.bit_length() - 1)
                        edges_in_common += popcount(adj[idx] & common)
                        c ^= lsb
                    k4s += edges_in_common // 2
                    
        indep = 1 if has_indep_6(adj) else 0
        return k4s, indep

    # Try to find a graph for N from 35 down to 17
    best_n = 17
    best_graph = None
    
    # Seed with Paley(17) if we fail to find larger
    squares_17 = {x * x % 17 for x in range(1, 17)}
    p17_adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares_17:
                p17_adj[i] |= (1 << j)
                p17_adj[j] |= (1 << i)
    best_graph = p17_adj
    best_n = 17
    
    for N in range(35, 16, -1):
        if time.time() > deadline - 10: # Stop if close to deadline
            break
            
        # Time budget for this N
        target_time = time.time() + 5.0
        
        # Initialize random graph
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        k4s, indep = count_violations(adj)
        current_cost = k4s + 10 * indep
        
        if current_cost == 0:
            best_n = N
            best_graph = adj
            break
            
        # Local Search
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < target_time and T > T_min:
            # Pick random edge to flip
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_k4s, new_indep = count_violations(adj)
            new_cost = new_k4s + 10 * new_indep
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                k4s = new_k4s
                indep = new_indep
                if current_cost == 0:
                    break
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            T *= cooling
            
        if current_cost == 0:
            best_n = N
            best_graph = adj
            break
            
    # Convert adj list of bitmasks to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_graph[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
