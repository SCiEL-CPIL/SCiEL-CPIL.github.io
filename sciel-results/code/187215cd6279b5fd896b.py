import time
import random
import sys

def _build_polarity_graph(q):
    """Build a polarity graph of PG(2,q) with C4-free property and known small alpha.
    Vertices are points of PG(2,q), edges connect orthogonal points under a polarity.
    Returns (n, adj_list, edges_list) or None if construction fails."""
    if q < 3 or q > 10:
        return None
    q2 = q * q
    q3 = q2 * q
    n_pts = q3 + q2 + 1
    
    # Generate all points of PG(2,q)
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            pt = (x, y, 1)
            key = (x, y, 1)
            if key not in seen:
                seen.add(key)
                points.append(pt)
    for x in range(q):
        pt = (x, 1, 0)
        key = (x, 1, 0)
        if key not in seen:
            seen.add(key)
            points.append(pt)
    pt = (1, 0, 0)
    key = (1, 0, 0)
    if key not in seen:
        seen.add(key)
        points.append(pt)
    
    n = len(points)
    if n != q3 + q2 + 1:
        return None
    
    # Polarity: (x1,y1,z1) perp (x2,y2,z2) iff x1*y2 + x2*y1 + z1*z2 == 0 (mod q)
    # This is a symmetric bilinear form.
    adj = [0] * n  # bitmask adjacency
    edges = []
    
    for i in range(n):
        x1, y1, z1 = points[i]
        for j in range(i+1, n):
            x2, y2, z2 = points[j]
            if (x1 * y2 + x2 * y1 + z1 * z2) % q == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
    
    return (n, adj, edges)

def _verify_c4_free(adj, n):
    """Check C4-free: no two vertices have >1 common neighbor."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if common & (common - 1):  # more than one bit set
                return False
    return True

def _max_independent_set_size(adj, n, timeout=5.0):
    """Find maximum independent set size using branch and bound."""
    start = time.time()
    best = 0
    
    # Greedy lower bound
    remaining = (1 << n) - 1
    greedy_count = 0
    while remaining:
        # Pick vertex with fewest neighbors in remaining
        best_v = -1
        best_deg = n + 1
        r = remaining
        while r:
            v = (r & (-r)).bit_length() - 1
            r &= r - 1
            deg = bin(adj[v] & remaining).count('1')
            if deg < best_deg:
                best_deg = deg
                best_v = v
        if best_v >= 0:
            greedy_count += 1
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
    best = greedy_count
    
    # Branch and bound
    def search(remaining, current, depth):
        nonlocal best
        if time.time() - start > timeout:
            return
        if remaining == 0:
            if current > best:
                best = current
            return
        # Upper bound: current + popcount(remaining)
        ub = current + bin(remaining).count('1')
        if ub <= best:
            return
        
        # Pick a vertex
        v = (remaining & (-remaining)).bit_length() - 1
        # Branch 1: include v
        search(remaining & ~(1 << v) & ~adj[v], current + 1, depth + 1)
        # Branch 2: exclude v
        search(remaining & ~(1 << v), current, depth + 1)
    
    search((1 << n) - 1, 0, 0)
    return best

def _try_add_vertex(adj, n, edges, deadline):
    """Try to add a new vertex connected to a subset of existing vertices
    while preserving C4-free property."""
    # A new vertex v can connect to a set S of existing vertices iff:
    # For any two vertices a, b in S, they must NOT share any common neighbor
    # other than v. But since v is new, the constraint is:
    # For any pair (a,b) in S, |N(a) ∩ N(b) ∩ existing| must be 0
    # (because if they shared a neighbor c, then v-a-c-b-v would be a C4)
    # Wait, that's not quite right. Let me think again.
    # 
    # C4 constraint: no two vertices have two common neighbors.
    # If we add v connected to S, then for any a, b in S:
    #   common neighbors of a and b = (N(a) ∩ N(b)) ∪ {v if both in S}
    #   So we need |N(a) ∩ N(b) ∩ existing| ≤ 1 if a≠b, and if a=b it's fine.
    #   Actually: N(a) ∩ N(b) in the new graph = (N_old(a) ∩ N_old(b)) ∪ {v} if a,b both in S
    #   So we need: if a, b both in S, then |N_old(a) ∩ N_old(b)| ≤ 1
    #   (because adding v makes it |N_old(a) ∩ N_old(b)| + 1, which must be ≤ 1)
    #   Wait no: if a and b are both neighbors of v, then v is a common neighbor.
    #   The existing common neighbors are N_old(a) ∩ N_old(b).
    #   Total common neighbors = |N_old(a) ∩ N_old(b)| + 1 (for v)
    #   We need this ≤ 1, so |N_old(a) ∩ N_old(b)| must be 0.
    #
    # So S must be a set where no two vertices share a common neighbor.
    # That means S must be an independent set in the "conflict graph" where
    # two vertices are connected if they share a common neighbor.
    
    if time.time() > deadline:
        return None
    
    # Build conflict graph: edge between a,b if N(a)∩N(b) != 0
    # S must be independent in this conflict graph
    # Find a large independent set in conflict graph
    
    # For efficiency, try greedy approach
    # Vertices that can potentially be in S: those with few conflicts
    conflict = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & adj[j]:  # share a common neighbor
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Greedy: pick vertices with fewest conflicts
    remaining = (1 << n) - 1
    S = 0
    while remaining and time.time() < deadline:
        best_v = -1
        best_conf = n + 1
        r = remaining
        while r:
            v = (r & (-r)).bit_length() - 1
            r &= r - 1
            conf_count = bin(conflict[v] & remaining).count('1')
            if conf_count < best_conf:
                best_conf = conf_count
                best_v = v
        if best_v < 0 or best_conf >= n:
            break
        S |= (1 << best_v)
        remaining &= ~(1 << best_v)
        remaining &= ~conflict[best_v]
    
    if S == 0:
        return None
    
    # Verify S is valid (no two vertices in S share a common neighbor)
    s_list = [i for i in range(n) if (S >> i) & 1]
    for i in range(len(s_list)):
        for j in range(i+1, len(s_list)):
            if adj[s_list[i]] & adj[s_list[j]]:
                return None
    
    # Add vertex n connected to S
    new_adj = adj[:] + [S]
    for i in s_list:
        new_adj[i] |= (1 << n)
    new_edges = edges[:] + [[i, n] for i in s_list]
    
    return (new_adj, n + 1, new_edges)

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin from 150s limit
    
    best_result = None
    best_n = 0
    
    # Try polarity graphs for various q
    for q in [9, 7, 5, 4]:
        if time.time() > deadline - 5:
            break
        try:
            result = _build_polarity_graph(q)
            if result is None:
                continue
            n, adj, edges = result
            
            # Verify C4-free
            if not _verify_c4_free(adj, n):
                continue
            
            # Check independence number
            alpha = _max_independent_set_size(adj, n, timeout=3.0)
            if alpha >= 11:
                continue
            
            # This is a valid graph
            if n > best_n:
                best_n = n
                best_result = {'n': n, 'edges': edges}
            
            # Try to extend
            current_adj = adj
            current_n = n
            current_edges = edges
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                ext = _try_add_vertex(current_adj, current_n, current_edges, deadline - 5)
                if ext is None:
                    break
                current_adj, current_n, current_edges = ext
                
                # Verify C4-free after extension
                if not _verify_c4_free(current_adj, current_n):
                    break
                
                # Check independence number
                alpha = _max_independent_set_size(current_adj, current_n, timeout=2.0)
                if alpha >= 11:
                    break
                
                if current_n > best_n:
                    best_n = current_n
                    best_result = {'n': current_n, 'edges': current_edges}
                    
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple valid graph
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
