import time
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 20
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_valid_addition(adj, u, v):
        # Check if adding edge (u,v) keeps graph K3,3-free
        # Constraint: for all pairs (x,y), |N(x) ∩ N(y)| <= 2
        # Adding (u,v) affects pairs involving u or v
        
        # Check pair (u,v): common neighbors
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
        
        # For all x != u,v: if v in N(x), then |N(u) ∩ N(x)| must be <= 1 (will become <=2 after add)
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[x] & (1 << v):
                # v is neighbor of x, so after adding (u,v), v becomes common neighbor of u and x
                common_ux = adj[u] & adj[x]
                if popcount(common_ux) > 1:
                    return False
            if adj[x] & (1 << u):
                common_vx = adj[v] & adj[x]
                if popcount(common_vx) > 1:
                    return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def edge_count(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    best_edges = 36  # K2,18 baseline
    best_adj = None
    
    # Initialize baseline: K2,18
    adj0 = [0] * n
    for a in range(2):
        for b in range(2, n):
            adj0[a] |= (1 << b)
            adj0[b] |= (1 << a)
    best_adj = list(adj0)
    
    # Try multiple random greedy runs
    rng = random.Random(42)
    num_trials = 0
    
    while time.time() < deadline:
        num_trials += 1
        adj = [0] * n
        # Random shuffle of edges
        edges = all_edges[:]
        rng.shuffle(edges)
        
        for (u, v) in edges:
            if is_valid_addition(adj, u, v):
                add_edge(adj, u, v)
        
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = list(adj)
            if best_edges >= 190:  # theoretical max is C(20,2)=190, unlikely but check
                break
        
        # Also try starting from baseline and adding
        if num_trials % 5 == 0:
            adj2 = list(best_adj)
            # Try to add more edges to best_adj
            remaining = [(u,v) for (u,v) in all_edges if not (adj2[u] & (1 << v))]
            rng.shuffle(remaining)
            for (u, v) in remaining:
                if is_valid_addition(adj2, u, v):
                    add_edge(adj2, u, v)
            ec2 = edge_count(adj2)
            if ec2 > best_edges:
                best_edges = ec2
                best_adj = list(adj2)
    
    # Convert best_adj to edge list
    result = []
    for i in range(n):
        mask = best_adj[i]
        for j in range(i+1, n):
            if mask & (1 << j):
                result.append([i, j])
    
    return result
