import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_result = {'n': 39, 'edges': []}
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline - 5:
            break
            
        min_deg_req = n - 39
        if min_deg_req <= 0:
            best_result = {'n': n, 'edges': []}
            break
            
        # Try to find a solution for this n
        found = False
        # Number of restarts depends on time remaining
        # Estimate time per restart
        # For n=46, local search might take ~1-10ms per step, ~100 steps.
        # So ~1ms per restart? No, more.
        # Let's try up to 500 restarts, but break early if time runs out.
        
        for restart in range(1000):
            if time.time() > deadline - 5:
                break
                
            success, edges = try_solve(n, min_deg_req, deadline - 2)
            if success:
                best_result = {'n': n, 'edges': edges}
                found = True
                break
                
        if found:
            break
            
    return best_result

def try_solve(n, min_deg_req, deadline):
    random.seed(int(time.time() * 1000) % (2**32)) # Random seed per call? No, deterministic is better for reproducibility if possible, but random restarts need randomness.
    # Actually, using a fixed seed per restart index is better for debugging, but here we want diversity.
    
    adj = [0] * n
    deg = [0] * n
    adj2 = [0] * n
    
    def popcount(x):
        return x.bit_count()
        
    def update_adj2(u):
        # Recompute adj2[u] based on adj[u]
        mask = adj[u]
        val = 0
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            val |= adj[v]
            mask ^= lsb
        adj2[u] = val
        
    def add_edge(u, v):
        nonlocal adj, deg, adj2
        
        # Check validity
        if adj[u] & (1 << v):
            return False
        if popcount(adj[u] & adj[v]) > 1:
            return False
        if adj2[u] & adj[v]:
            return False
        if adj2[v] & adj[u]:
            return False
            
        # Add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        
        # Update adj2 for u and v
        # adj2[u] gains adj[v]
        # adj2[v] gains adj[u]
        # Also, for any w in N(u), adj2[w] might change?
        # No, adj2[w] depends on N(w). N(w) didn't change.
        # Wait, adj2[w] = OR_{x in N(w)} adj[x].
        # If w is neighbor of u, then u in N(w).
        # adj[u] changed. So adj2[w] must be updated.
        
        # Affected vertices: u, v, and all neighbors of u and v.
        affected = set()
        affected.add(u)
        affected.add(v)
        
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            affected.add(w)
            mask_u ^= lsb
            
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            affected.add(w)
            mask_v ^= lsb
            
        for w in affected:
            update_adj2(w)
            
        return True

    def remove_edge(u, v):
        nonlocal adj, deg, adj2
        
        if not (adj[u] & (1 << v)):
            return False
            
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1
        
        affected = set()
        affected.add(u)
        affected.add(v)
        
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            affected.add(w)
            mask_u ^= lsb
            
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            affected.add(w)
            mask_v ^= lsb
            
        for w in affected:
            update_adj2(w)
            
        return True

    # Local Search
    max_steps = 5000
    steps = 0
    kicks = 0
    max_kicks = 100
    
    while steps < max_steps and time.time() < deadline:
        if min(deg) >= min_deg_req:
            # Extract edges
            edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append([i, j])
                    mask ^= lsb
            return True, edges
            
        steps += 1
        
        # Find vertex with min degree
        min_deg = min(deg)
        min_vertices = [i for i in range(n) if deg[i] == min_deg]
        u = random.choice(min_vertices)
        
        # Find valid neighbors for u
        valid_v = []
        mask = ((1 << n) - 1) ^ adj[u] ^ (1 << u)
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            # Check validity for (u, v)
            if popcount(adj[u] & adj[v]) <= 1 and (adj2[u] & adj[v] == 0) and (adj2[v] & adj[u] == 0):
                valid_v.append(v)
            mask ^= lsb
            
        if valid_v:
            # Pick a random valid v
            v = random.choice(valid_v)
            add_edge(u, v)
        else:
            # Kick: Remove a random edge
            # Prefer removing edges incident to high degree vertices to free up space?
            # Or just random.
            all_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        all_edges.append((i, j))
                    mask ^= lsb
            
            if all_edges:
                u_r, v_r = random.choice(all_edges)
                remove_edge(u_r, v_r)
                kicks += 1
            else:
                # No edges to remove? Should not happen if min_deg < req and n > 1
                break
                
            if kicks > max_kicks:
                break
                
    return False, None
