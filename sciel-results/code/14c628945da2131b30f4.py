import time
import random
import sys

# EVOLVE-BLOCK-START

def _has_c4_bitmask(adj, n):
    """Check if graph has a C4 using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):
                return True
    return False

def _max_independent_set_size_bitmask(adj, n):
    """Find maximum independent set size using bitmask branch and bound."""
    # Greedy lower bound
    remaining = (1 << n) - 1
    alpha_lb = 0
    while remaining:
        # Pick vertex with minimum degree in remaining subgraph
        best_v = -1
        best_deg = n
        r = remaining
        while r:
            lsb = r & (-r)
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            r ^= lsb
        if best_v == -1:
            break
        alpha_lb += 1
        remaining &= ~adj[best_v]
        remaining &= ~(1 << best_v)
    
    # Branch and bound for upper bound
    best = [0]
    
    # Precompute degrees for ordering? Or just use bit_count.
    # To speed up, we can order vertices by degree in the remaining graph.
    
    def branch(remaining, current_size):
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pruning: if current_size + max possible <= best, return
        if current_size + remaining.bit_count() <= best[0]:
            return
            
        # Pick a vertex with high degree in remaining subgraph to prune more effectively
        # Or just pick the first one? High degree is better for pruning "include" branch?
        # Actually, for independent set, picking a vertex with high degree removes many neighbors,
        # which is good for the "include" branch (smaller remaining set).
        # But for the "exclude" branch, we just remove the vertex.
        # Standard heuristic: pick vertex with max degree in remaining.
        
        best_v = -1
        max_deg = -1
        r = remaining
        while r:
            lsb = r & (-r)
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            r ^= lsb
        
        v = best_v
        
        # Include v: remove v and its neighbors
        branch(remaining & ~adj[v] & ~(1 << v), current_size + 1)
        
        # Exclude v: remove v
        branch(remaining & ~(1 << v), current_size)
        
    branch((1 << n) - 1, 0)
    return best[0]

def _verify_graph(n, edges):
    """Verify graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        if u >= n or v >= n or u == v:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    # Check C4
    if _has_c4_bitmask(adj, n):
        return False
    # Check independence number < 11
    alpha = _max_independent_set_size_bitmask(adj, n)
    if alpha >= 11:
        return False
    return True

def _build_polarity_graph_q5():
    """
    Construct polarity graph from PG(2,5) - a known C4-free graph.
    PG(2,q) has (q^2+q+1) points. For q=5, that's 31 points.
    Two points are adjacent if they are perpendicular under the polarity.
    """
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(x * inv) % q for x in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n = len(points)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = (points[i][0] * points[j][0] + points[i][1] * points[j][1] + points[i][2] * points[j][2]) % q
            if dot == 0:
                edges.append([i, j])
    
    return n, edges

def _try_add_edge(adj, n, u, v):
    """Try to add edge (u,v) maintaining C4-free property."""
    if u == v or u >= n or v >= n:
        return False
    if adj[u] & (1 << v):
        return False
    
    # Check C4 condition for new edge
    # 1. u and v must have no common neighbors
    if adj[u] & adj[v]:
        return False
    
    # 2. For any x in N(v), u and x must have no common neighbors (other than v, but v is not in N(x) yet? No, v is in N(x) if x in N(v))
    # Wait, if x in N(v), then v is a neighbor of x.
    # After adding (u,v), u is a neighbor of v.
    # So v is a common neighbor of u and x?
    # N'(u) = N(u) U {v}
    # N'(x) = N(x) U {u} (if x in N(v))
    # Common neighbors of u and x in N':
    # - v is in N'(u) and v is in N'(x) (since x in N(v) implies v in N(x)). So v is a common neighbor.
    # - Any y in N(u) ∩ N(x) is a common neighbor.
    # So we need |N(u) ∩ N(x)| to be 0.
    
    for x in range(n):
        if x == u or x == v:
            continue
        # If x is neighbor of v
        if adj[v] & (1 << x):
            if adj[u] & adj[x]:
                return False
        # If x is neighbor of u
        if adj[u] & (1 << x):
            if adj[v] & adj[x]:
                return False

    return True

def _construct_graph(deadline):
    """
    Construct a C4-free graph with alpha < 11.
    Starts with PG(2,5) and tries to add vertices and edges.
    """
    start_time = time.time()
    
    # 1. Base graph
    n0, edges0 = _build_polarity_graph_q5()
    n = n0
    edges = edges0[:]
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Verify base graph
    if _has_c4_bitmask(adj, n):
        # Should not happen, but just in case
        return 0, []
    
    alpha = _max_independent_set_size_bitmask(adj, n)
    if alpha >= 11:
        # Base graph is not good enough, but PG(2,5) usually has small alpha
        # If it is, we might need to add edges immediately
        pass

    best_n = n
    best_edges = edges[:]
    
    # 2. Try to add vertices to reach n=39
    target_n = 39
    
    for target in range(n + 1, target_n + 1):
        if time.time() > deadline - 5: # Leave 5s for verification
            break
            
        new_v = n
        adj.append(0)
        
        # Try to find a valid neighborhood for new_v
        # The neighborhood S of new_v must satisfy:
        # For any u, v in S, N(u) ∩ N(v) = empty (in the current graph)
        # This ensures new_v doesn't create a C4 with u and v.
        # Also, we need to ensure alpha doesn't jump to 11.
        
        found_valid = False
        max_tries = 200
        random.seed(target) # Deterministic for reproducibility in attempts
        
        for attempt in range(max_tries):
            if time.time() > deadline - 5:
                break
                
            # Generate random subset of current vertices
            # Size of subset should be reasonable.
            # In PG(2,5), max degree is 5.
            # Let's try subsets of size 3-6.
            size = random.randint(3, 6)
            candidates = random.sample(range(n), min(size, n))
            
            # Check if candidates form an "independent set in the line graph" sense
            # i.e., no two candidates share a neighbor
            valid = True
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if adj[candidates[i]] & adj[candidates[j]]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Tentatively add new_v connected to candidates
            for u in candidates:
                adj[new_v] |= (1 << u)
                adj[u] |= (1 << new_v)
            
            # Check C4
            if not _has_c4_bitmask(adj, n + 1):
                # Check Alpha
                if time.time() < deadline - 2:
                    alpha_new = _max_independent_set_size_bitmask(adj, n + 1)
                    if alpha_new < 11:
                        # Success!
                        found_valid = True
                        # Update best
                        best_n = n + 1
                        # Reconstruct edges list
                        best_edges = []
                        for i in range(n + 1):
                            for j in range(i + 1, n + 1):
                                if adj[i] & (1 << j):
                                    best_edges.append([i, j])
                        break
                    else:
                        # Alpha is too high. Try to fix by adding edges?
                        # This is complex. For now, reject this neighborhood.
                        pass
            
            # Rollback
            for u in candidates:
                adj[new_v] &= ~(1 << u)
                adj[u] &= ~(1 << new_v)
        
        if found_valid:
            n += 1
            # Update adj to reflect the successful addition
            # Note: adj already has the successful addition in the loop above
            # We just need to continue
        else:
            # Failed to add vertex with this strategy
            # Remove the dummy vertex we added
            adj.pop()
            # Try a different strategy? Or stop?
            # If we can't add a vertex, we can't reach higher n easily.
            # But maybe we can add edges to existing graph to make room?
            # For now, if we can't add a vertex, we break.
            break
    
    # 3. If we reached target_n, verify and return
    if best_n >= 39:
        if _verify_graph(best_n, best_edges):
            return best_n, best_edges
    
    # 4. If we didn't reach 39, try to optimize the current best graph
    # Maybe we can add more edges to reduce alpha?
    # But we already checked alpha < 11 for the best_n we found.
    # If best_n < 39, we return best_n.
    
    return best_n, best_edges

def construct():
    """Construct a C4-free graph with no independent set of size 11."""
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    n, edges = _construct_graph(deadline)
    
    if n > 0 and _verify_graph(n, edges):
        return {'n': n, 'edges': edges}
    
    # Fallback: Return a valid smaller graph if construction failed
    # PG(2,5) is 31 vertices.
    n0, edges0 = _build_polarity_graph_q5()
    if _verify_graph(n0, edges0):
        return {'n': n0, 'edges': edges0}
    
    # Last resort: Empty graph? No, n>=1.
    # Single vertex
    return {'n': 1, 'edges': []}

# EVOLVE-BLOCK-END
