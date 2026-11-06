import time
import random
import sys

def _compute_edge_triangles(adj, n=40):
    """Compute the number of triangles containing each edge.
    Returns a dict mapping (u, v) to count, and the max count.
    """
    tri_counts = {}
    max_tri = 0
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # Count common neighbors
                common = adj[u] & adj[v]
                cnt = len(common)
                tri_counts[(u, v)] = cnt
                if cnt > max_tri:
                    max_tri = cnt
    return tri_counts, max_tri

def _apply_swap(adj, tri_counts, u, v, x, y):
    """Apply a swap: remove (u,v), add (x,y).
    Returns the change in max triangle count (delta_max).
    Updates adj and tri_counts in place.
    """
    # 1. Remove edge (u, v)
    # This edge is removed, so its count is no longer relevant for "max", 
    # but we need to update counts of other edges that share vertices u or v.
    
    # Edges affected are those incident to u or v, and those incident to x or y.
    # Specifically, a triangle (u, v, w) is broken. So for each w in common_neighbors(u, v),
    # the triangle count for (u, w) decreases by 1, and for (v, w) decreases by  1.
    
    common_uv = adj[u] & adj[v]
    for w in common_uv:
        # Update (u, w)
        key_uw = (min(u, w), max(u, w))
        tri_counts[key_uw] -= 1
        # Update (v, w)
        key_vw = (min(v, w), max(v, w))
        tri_counts[key_vw] -= 1
        
    # Remove (u, v) from graph
    adj[u].remove(v)
    adj[v].remove(u)
    del tri_counts[(min(u, v), max(u, v))]
    
    # 2. Add edge (x, y)
    # This creates new triangles (x, y, w) for each w in common_neighbors(x, y).
    common_xy = adj[x] & adj[y]
    for w in common_xy:
        # Update (x, w)
        key_xw = (min(x, w), max(x, w))
        tri_counts[key_xw] += 1
        # Update (y, w)
        key_yw = (min(y, w), max(y, w))
        tri_counts[key_yw] += 1
        
    # Add (x, y) to graph
    adj[x].add(y)
    adj[y].add(x)
    new_cnt = len(common_xy)
    tri_counts[(min(x, y), max(x, y))] = new_cnt

def _get_max_tri(tri_counts):
    """Get the maximum triangle count from the dict."""
    if not tri_counts:
        return 0
    return max(tri_counts.values())

def _simulated_annealing(adj, tri_counts, current_max, time_limit, seed):
    """
    Run Simulated Annealing to minimize the maximum book size.
    """
    random.seed(seed)
    n = 40
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_max = current_max
    best_adj = [set(s) for s in adj]
    best_tri = dict(tri_counts)
    
    # Temperature schedule
    T = 2.0
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        # Pick a random edge to remove
        # We need a list of edges. Maintaining a list is expensive to update.
        # Instead, pick a random vertex u, then a random neighbor v.
        # This biases towards high-degree vertices, which is fine.
        u = random.randint(0, n-1)
        if not adj[u]:
            continue
        v = random.choice(list(adj[u]))
        
        # Pick a random non-edge (x, y) to add
        # Ensure (x,y) is not (u,v) and not an existing edge.
        # Also ensure (x,y) doesn't create a duplicate if we are careful.
        # Strategy: Pick x, y randomly. Check if edge exists.
        # If edge exists, retry.
        # To avoid infinite loops, limit retries.
        
        found_new_edge = False
        for _ in range(10):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            # Normalize
            if x > y: x, y = y, x
            # Check if (x,y) is the edge we are removing (u,v)
            if (x == u and y == v) or (x == v and y == u):
                continue
            # Check if (x,y) is already an edge
            if y in adj[x]:
                continue
            
            # Valid new edge (x, y)
            found_new_edge = True
            break
        
        if not found_new_edge:
            continue
            
        # Calculate delta in max book size
        # We need to know the max BEFORE and AFTER.
        # Since we are doing SA, we accept if new_max < old_max OR with prob exp(-delta/T).
        # But calculating new_max exactly requires applying the swap and taking max.
        # We can apply, get max, and revert if rejected.
        
        # Apply swap
        _apply_swap(adj, tri_counts, u, v, x, y)
        
        # Get new max
        # Optimization: We can track max more efficiently, but for now max() is O(E).
        # E=401, so max() is fast.
        new_max = _get_max_tri(tri_counts)
        
        # Decide acceptance
        if new_max < current_max:
            current_max = new_max
            # Update best
            if current_max < best_max:
                best_max = current_max
                best_adj = [set(s) for s in adj]
                best_tri = dict(tri_counts)
        else:
            delta = new_max - current_max
            if delta == 0:
                # Accept neutral moves to explore
                current_max = new_max
            else:
                # Accept worsening moves with probability exp(-delta / T)
                # delta is integer >= 1
                if random.random() < 2.718281828 ** (-delta / T):
                    current_max = new_max
                else:
                    # Revert
                    _apply_swap(adj, tri_counts, x, y, u, v)
        
        # Cool down
        T *= cooling_rate
        
        # Check time again inside loop for safety
        if time.time() > deadline:
            break
            
    return best_adj, best_tri, best_max

def construct():
    n = 40
    target_edges = 401
    total_time_limit = 145.0 # Safety margin for 150s limit
    
    start_time = time.time()
    
    best_overall_max = float('inf')
    best_overall_edges = set()
    
    # We will run multiple restarts with different seeds and initial graphs.
    # Initial graph strategy: Random graph with 401 edges.
    # Or a quasi-random one.
    
    num_restarts = 5
    time_per_restart = (total_time_limit - 5) / num_restarts # Leave 5s for overhead
    
    for i in range(num_restarts):
        if time.time() - start_time > total_time_limit - 2:
            break
            
        seed_val = 1000 + i * 100
        
        # Generate initial graph
        # Random graph with 401 edges
        # To make it "quasi-random" or balanced, we can try to make it regular-ish.
        # But simple random is okay for SA to work from.
        
        edges = set()
        random.seed(seed_val)
        
        # Method: Random sampling of edges
        # Total possible edges: 40*39/2 = 780
        # We need 401.
        # Simple random selection might lead to uneven degrees.
        # Better: Generate a random permutation of all possible edges and take the first 401.
        all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_possible)
        edges = set(all_possible[:target_edges])
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Compute initial triangle counts
        tri_counts, current_max = _compute_edge_triangles(adj, n)
        
        # Run SA
        sa_adj, sa_tri, sa_max = _simulated_annealing(adj, tri_counts, current_max, time_per_restart, seed_val)
        
        if sa_max < best_overall_max:
            best_overall_max = sa_max
            # Reconstruct edges from best_adj
            best_overall_edges = set()
            for u in range(n):
                for v in sa_adj[u]:
                    if u < v:
                        best_overall_edges.add((u, v))
                        
    # Ensure we return a valid list of edges
    # If best_overall_edges is empty (shouldn't happen), fallback to initial
    if not best_overall_edges:
         # Fallback: just return a valid set of 401 edges
         all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
         random.seed(1)
         random.shuffle(all_possible)
         best_overall_edges = set(all_possible[:target_edges])
         
    # Convert to list of lists
    return [list(e) for e in best_overall_edges]
