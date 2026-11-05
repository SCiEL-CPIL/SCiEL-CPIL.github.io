"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_c4_free(n, adj):
    """Check if graph with adjacency matrix adj on n vertices is C4-free."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                common = np.sum(adj[i] & adj[j])
                if common >= 2:
                    return False
    return True

def _min_degree(n, adj):
    return int(np.min(adj.sum(axis=1))) if n > 0 else 0

def _try_add_edge(adj, i, j, n):
    """Check if adding edge (i,j) keeps graph C4-free. Returns True if safe."""
    if adj[i][j]:
        return True
    common = np.sum(adj[i] & adj[j])
    return common < 2

def _greedy_bitmask(target_n, min_deg_req, seed_val, max_iter=50000):
    """
    Greedy construction using bitmask representation for fast C4 checking.
    Returns (success, edges_list) where edges_list is list of [u,v] pairs.
    """
    random.seed(seed_val)
    
    masks = [0] * target_n  # adjacency bitmask for each vertex
    degrees = [0] * target_n
    # reach2[v] = bitmask of vertices at distance exactly 2 from v (via one intermediate)
    # Actually, for C4 check: adding edge (u,v) creates C4 iff |N(u) ∩ N(v)| >= 2
    # So we need to track N(u) ∩ N(v) for all pairs.
    # The reach2 approach from inspiration 2: reach2[u] is the set of vertices w such that
    # there exists a path u-x-w of length 2. But that's not quite right for C4.
    
    # Actually, the C4 condition: adding edge (u,v) is safe iff |N(u) ∩ N(v)| <= 1.
    # We can maintain for each pair (u,v) the count |N(u) ∩ N(v)|, but that's O(n^2) space.
    # With bitmasks, we can compute it on the fly: popcount(masks[u] & masks[v]).
    # For n <= 47, Python integers can handle this.
    
    # Let's use a simpler approach: maintain masks, and when checking if (u,v) can be added,
    # compute common = masks[u] & masks[v], and check if popcount(common) < 2.
    
    # For efficiency, we'll precompute nothing but use the bitmask intersection.
    
    stuck_counter = 0
    max_stuck = 200
    
    for _ in range(max_iter):
        # Check if all degrees are sufficient
        if min(degrees) >= min_deg_req:
            # Success!
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if masks[i] & (1 << j):
                        edges.append([i, j])
            return True, edges
        
        # Find vertices with degree < min_deg_req
        low_verts = [v for v in range(target_n) if degrees[v] < min_deg_req]
        if not low_verts:
            # All degrees sufficient (should have been caught above)
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if masks[i] & (1 << j):
                        edges.append([i, j])
            return True, edges
        
        # Pick a low-degree vertex (randomly for diversity)
        u = random.choice(low_verts)
        
        # Find valid vertices v to connect to u
        valid_vs = []
        for v in range(target_n):
            if v == u:
                continue
            if masks[u] & (1 << v):
                continue  # Already connected
            # Check C4 condition: |N(u) ∩ N(v)| < 2
            common = masks[u] & masks[v]
            # Count bits
            bit_count = bin(common).count('1')
            if bit_count < 2:
                valid_vs.append(v)
        
        if not valid_vs:
            stuck_counter += 1
            if stuck_counter > max_stuck:
                return False, None
            continue
        
        stuck_counter = 0
        # Pick v from valid_vs
        # Strategy: pick the one that gives best degree balance
        # Or just random
        v = random.choice(valid_vs)
        
        # Add edge (u, v)
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1
    
    # Check final state
    if min(degrees) >= min_deg_req:
        edges = []
        for i in range(target_n):
            for j in range(i+1, target_n):
                if masks[i] & (1 << j):
                    edges.append([i, j])
        return True, edges
    
    return False, None

def _local_search_improve(n, edges, min_deg_req, max_steps=20000):
    """
    Try to improve a C4-free graph by adding edges to increase min degree.
    Returns (success, edges_list).
    """
    # Build adjacency from edges
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    degs = adj.sum(axis=1)
    
    for step in range(max_steps):
        if min(degs) >= min_deg_req:
            # Success
            new_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        new_edges.append([i, j])
            return True, new_edges
        
        # Find low-degree vertices
        low_verts = [v for v in range(n) if degs[v] < min_deg_req]
        if not low_verts:
            new_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        new_edges.append([i, j])
            return True, new_edges
        
        # Pick a low-degree vertex
        v = random.choice(low_verts)
        
        # Try to add an edge from v to some w
        added = False
        for w in range(n):
            if w == v or adj[v][w]:
                continue
            if _try_add_edge(adj, v, w, n):
                adj[v][w] = True
                adj[w][v] = True
                degs[v] += 1
                degs[w] += 1
                added = True
                break
        
        if not added:
            # Try all low-degree vertices
            any_added = False
            for v2 in low_verts:
                for w in range(n):
                    if w == v2 or adj[v2][w]:
                        continue
                    if _try_add_edge(adj, v2, w, n):
                        adj[v2][w] = True
                        adj[w][v2] = True
                        degs[v2] += 1
                        degs[w] += 1
                        any_added = True
                        break
                if any_added:
                    break
            
            if not any_added:
                break  # Stuck
    
    # Final check
    if min(degs) >= min_deg_req:
        new_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    new_edges.append([i, j])
        return True, new_edges
    
    return False, None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin under 150s limit
    
    best_n = 39
    best_edges = []
    
    # Try n=47 first, then 46
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
        
        min_deg_req = target_n - 39
        
        if min_deg_req <= 0:
            # Empty graph works
            if target_n > best_n:
                best_n = target_n
                best_edges = []
            continue
        
        found = False
        
        # Multiple random restarts with bitmask greedy
        num_restarts = 50
        for attempt in range(num_restarts):
            if time.time() > deadline:
                break
            
            success, edges = _greedy_bitmask(target_n, min_deg_req, seed_val=attempt * 1000 + 42)
            
            if success:
                # Verify
                adj = np.zeros((target_n, target_n), dtype=bool)
                for u, v in edges:
                    adj[u][v] = True
                    adj[v][u] = True
                
                if _is_c4_free(target_n, adj) and _min_degree(target_n, adj) >= min_deg_req:
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = edges
                    found = True
                    break
        
        if found:
            if target_n == 47:
                return {'n': best_n, 'edges': best_edges}
            # For n=46, we can try local search to see if we can push to 47,
            # but let's just return 46 for now since it's a big win
            break
    
    # If we didn't find n=46 or 47, try n=45, 44, ...
    if best_n < 46:
        for n_try in range(45, 38, -1):
            if time.time() > deadline:
                break
            
            min_deg_req = n_try - 39
            if min_deg_req <= 0:
                if n_try > best_n:
                    best_n = n_try
                    best_edges = []
                continue
            
            found = False
            num_restarts = 30
            for attempt in range(num_restarts):
                if time.time() > deadline:
                    break
                
                success, edges = _greedy_bitmask(n_try, min_deg_req, seed_val=attempt * 777 + n_try)
                
                if success:
                    adj = np.zeros((n_try, n_try), dtype=bool)
                    for u, v in edges:
                        adj[u][v] = True
                        adj[v][u] = True
                    
                    if _is_c4_free(n_try, adj) and _min_degree(n_try, adj) >= min_deg_req:
                        if n_try > best_n:
                            best_n = n_try
                            best_edges = edges
                        found = True
                        break
            
            if found:
                break
    
    # Fallback
    if best_n >= 46:
        return {'n': best_n, 'edges': best_edges}
    elif best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
