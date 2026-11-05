import random
import time

def can_add(adj, u, v):
    # Check if adding edge (u, v) maintains K2,3-free property
    # 1. Common neighbors of u and v must be <= 2
    if (adj[u] & adj[v]).bit_count() > 2:
        return False
    
    # 2. For any neighbor w of v, the pair (u, w) must have <= 2 common neighbors in new graph.
    # In new graph, N'(u) = N(u) U {v}, N'(w) = N(w).
    # Intersection = (N(u) cap N(w)) U ({v} cap N(w)).
    # Since w is neighbor of v, v is in N(w). So v is in intersection.
    # Size = |N(u) cap N(w)| + 1.
    # We need |N(u) cap N(w)| + 1 <= 2 => |N(u) cap N(w)| <= 1.
    mask_v = adj[v]
    while mask_v:
        lsb = mask_v & -mask_v
        w = lsb.bit_length() - 1
        if (adj[u] & adj[w]).bit_count() > 1:
            return False
        mask_v ^= lsb
        
    # 3. Symmetrically for neighbors of u
    # For any neighbor w of u, pair (v, w) must have <= 2 common neighbors.
    # N'(v) = N(v) U {u}, N'(w) = N(w).
    # u is in N(w) since w is neighbor of u.
    # Size = |N(v) cap N(w)| + 1.
    # Need |N(v) cap N(w)| <= 1.
    mask_u = adj[u]
    while mask_u:
        lsb = mask_u & -mask_u
        w = lsb.bit_length() - 1
        if (adj[v] & adj[w]).bit_count() > 1:
            return False
        mask_u ^= lsb
        
    return True

def greedy_construct():
    adj = [0] * 20
    edges = []
    all_edges = [(u, v) for u in range(20) for v in range(u + 1, 20)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
    return edges, adj

def local_search(edges, adj, max_steps=50000):
    current_edges = edges[:]
    current_adj = adj[:]
    current_count = len(current_edges)
    
    # Precompute all edges for random selection
    all_edges = [(u, v) for u in range(20) for v in range(u + 1, 20)]
    
    # Set of current edges for fast lookup
    edge_set = set(tuple(e) for e in current_edges)
    
    steps = 0
    while steps < max_steps:
        steps += 1
        
        # Try to add a random non-edge
        u, v = random.choice(all_edges)
        if (u, v) in edge_set:
            continue
            
        if can_add(current_adj, u, v):
            # Add edge
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            edge_set.add((u, v))
            current_edges.append([u, v])
            current_count += 1
        else:
            # Try a swap: remove a random edge that might be blocking, then add
            # Heuristic: If can_add fails, it's due to some pair (u, w) or (v, w) having too many common neighbors.
            # We can try to remove a random edge incident to u or v, or a random edge in the graph.
            # To keep it simple and fast, we try to remove a random edge from the current graph.
            if current_count > 0:
                rem_idx = random.randint(0, current_count - 1)
                ru, rv = current_edges[rem_idx]
                
                # Remove ru, rv
                current_adj[ru] &= ~(1 << rv)
                current_adj[rv] &= ~(1 << ru)
                edge_set.remove((ru, rv))
                current_edges.pop(rem_idx)
                current_count -= 1
                
                # Try to add u, v
                if can_add(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    current_edges.append([u, v])
                    current_count += 1
                else:
                    # Put back the removed edge if we can't add the new one
                    current_adj[ru] |= (1 << rv)
                    current_adj[rv] |= (1 << ru)
                    edge_set.add((ru, rv))
                    current_edges.append([ru, rv])
                    current_count += 1

    return current_edges

def construct():
    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Phase 1: Greedy to get a good starting point
    for _ in range(500):
        if time.time() - start_time > time_limit:
            break
        edges, adj = greedy_construct()
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
            
    if time.time() - start_time > time_limit:
        return best_edges
        
    # Phase 2: Local Search to improve
    # Run local search on the best greedy solution found
    # We can run multiple local searches from different starting points if time permits
    for _ in range(20): # Run a few local searches
        if time.time() - start_time > time_limit:
            break
            
        # Get a fresh greedy solution to use as start for local search
        edges, adj = greedy_construct()
        
        # If this greedy is already better, update best
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        # Run local search
        # Limit steps based on remaining time
        remaining_time = time_limit - (time.time() - start_time)
        steps = int(remaining_time * 1000) # Approx steps
        if steps < 1000:
            steps = 1000
            
        improved_edges = local_search(edges, adj, max_steps=steps)
        
        if len(improved_edges) > best_count:
            best_count = len(improved_edges)
            best_edges = improved_edges[:]

    return best_edges
