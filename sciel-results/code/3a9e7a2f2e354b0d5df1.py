import random
import time

def construct():
    n = 18
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Precompute bit counts for speed
    # popcount can be done with bin(x).count('1') or int.bit_count() in py3.10+
    # Assuming modern python
    
    def calc_score(adj):
        # adj: list of n integers (bitmasks)
        size = 1 << n
        is_tf = [False] * size
        is_tf[0] = True
        max_size = 0
        
        # Optimization: Only check masks up to a certain size if we find a good one?
        # No, we need the max.
        
        for mask in range(1, size):
            # Find lowest set bit
            u = (mask & -mask).bit_length() - 1
            rest = mask ^ (1 << u)
            
            if not is_tf[rest]:
                continue
                
            N_u = adj[u]
            common = N_u & rest
            
            # Check if common contains an edge
            has_edge = False
            c = common
            while c:
                v_bit = c & -c
                v = v_bit.bit_length() - 1
                c ^= v_bit
                if adj[v] & common:
                    has_edge = True
                    break
            
            if not has_edge:
                is_tf[mask] = True
                # Update max size
                # popcount
                pc = bin(mask).count('1')
                if pc > max_size:
                    max_size = pc
                    # Early exit if we know we can't beat current best?
                    # Not easily.
                    
        return max_size

    def creates_k4(adj, u, v):
        # Check if adding edge (u,v) creates a K4
        # Need w, x in N(u) & N(v) such that edge(w,x) exists
        common = adj[u] & adj[v]
        c = common
        while c:
            w_bit = c & -c
            w = w_bit.bit_length() - 1
            c ^= w_bit
            if adj[w] & common:
                return True
        return False

    best_score = 18 # Max possible
    best_adj = None
    
    # Initialize with a random K4-free graph
    # Strategy: Start with complete graph, remove edges to break K4s
    # Or start with random and fix
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Make it K4-free
    # Iterate and remove edges from K4s
    # This might be slow if many K4s.
    # Better: Start with sparse and add edges?
    # Let's try starting with K6,6,6 and perturbing.
    # K6,6,6 is K4-free.
    adj = [0] * n
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    current_score = calc_score(adj)
    best_score = current_score
    best_adj = adj[:]
    
    # Local Search
    while time.time() < deadline:
        # Pick random edge to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        
        if adj[u] & (1 << v):
            # Edge exists, try removing
            new_adj = adj[:]
            new_adj[u] ^= (1 << v)
            new_adj[v] ^= (1 << u)
            
            # Removing an edge always keeps it K4-free
            new_score = calc_score(new_adj)
            if new_score < best_score:
                best_score = new_score
                best_adj = new_adj
                adj = new_adj
            # If not better, keep original adj (don't update best_adj unless better)
            # Wait, we should update adj to the trial if we want to explore around it?
            # Hill climbing: only accept if better.
        else:
            # Edge missing, try adding
            if creates_k4(adj, u, v):
                continue
            
            new_adj = adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            new_score = calc_score(new_adj)
            if new_score < best_score:
                best_score = new_score
                best_adj = new_adj
                adj = new_adj
        
        # Random restart if stuck?
        # If no improvement in last X steps, restart
        # Simple: just keep going.
        
    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
