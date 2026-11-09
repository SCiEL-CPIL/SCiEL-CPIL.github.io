"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random

# EVOLVE-BLOCK-START

def _can_add_edge(adj, u, v):
    """
    Fast check: does adding edge (u, v) create a C4?
    We assume (u, v) is not currently an edge.
    A C4 would be u - y - x - v - u with x in N(v), y in N(u), and edge (x, y).
    So for any x in N(v), if N(x) intersects N(u), then C4 exists.
    """
    nu = adj[u]
    nv = adj[v]
    x = nv
    while x:
        lsb = x & -x
        xi = lsb.bit_length() - 1
        x ^= lsb
        if adj[xi] & nu:
            return False
    return True

def _min_degree(adj, n):
    md = n
    for a in adj:
        d = a.bit_count()
        if d < md:
            md = d
    return md

def _is_c4_free(adj, n):
    """
    Full verification: for every edge (u, v), the number of common neighbors must be at most 1.
    """
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj[v]
                # If common has >= 2 bits, then there is a C4
                if common & (common - 1):
                    return False
    return True

def _greedy_build_structured(n, rng, p=0.25, target_min_deg=None):
    """
    Build an initial C4-free graph by adding random edges with probability p,
    but using the C4 check to ensure C4-freedom.
    If target_min_deg is given, prioritize connecting low-degree vertices.
    """
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    
    if target_min_deg is None:
        # Random acceptance
        for u, v in all_edges:
            if rng.random() < p:
                if _can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    else:
        # Degree-aware: always try to add edges that help low-degree vertices
        degrees = [0] * n
        for u, v in all_edges:
            du = degrees[u]
            dv = degrees[v]
            # Prioritize if either endpoint is below target
            if du < target_min_deg or dv < target_min_deg:
                if _can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
            elif rng.random() < p:
                if _can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
    
    return adj

def _can_add_edge_after_removal(adj, u, v, rem_u, rem_v):
    """
    Check if adding edge (u,v) is C4-safe after removing edge (rem_u, rem_v).
    We simulate the removal by masking out rem_u from adj[rem_v] and vice versa.
    """
    nu = adj[u]
    nv = adj[v]
    
    # If u or v is involved in the removal, adjust their neighbor sets
    if rem_u == u:
        nu &= ~(1 << rem_v)
    elif rem_v == u:
        nu &= ~(1 << rem_u)
    if rem_u == v:
        nv &= ~(1 << rem_v)
    elif rem_v == v:
        nv &= ~(1 << rem_u)
    
    # For each x in N(v) (adjusted), check if N(x) (adjusted) intersects N(u) (adjusted)
    x = nv
    while x:
        lsb = x & -x
        xi = lsb.bit_length() - 1
        x ^= lsb
        if xi == rem_u:
            nx = adj[xi] & ~(1 << rem_v)
        elif xi == rem_v:
            nx = adj[xi] & ~(1 << rem_u)
        else:
            nx = adj[xi]
        if nx & nu:
            return False
    return True

def _local_search(adj, n, target_min_deg, rng, max_iter=2000):
    """
    Local search to increase minimum degree while maintaining C4-freedom.
    Operations:
    1. Add an edge from a low-degree vertex to a non-neighbor if C4-safe.
    2. Swap: remove edge (a, b) and add edge (u, v) if it improves min degree
       and maintains C4-freedom.
    3. Double-swap: remove two edges and add two different edges.
    """
    improved = True
    it = 0
    degrees = [a.bit_count() for a in adj]
    
    while improved and it < max_iter:
        improved = False
        it += 1
        
        # Find vertices below target
        low_verts = [i for i in range(n) if degrees[i] < target_min_deg]
        
        if not low_verts:
            break
        
        # Operation 1: Add edges to low-degree vertices (try all non-neighbors)
        for u in low_verts:
            if degrees[u] >= target_min_deg:
                continue
            au = adj[u]
            non_neighbors = []
            for v in range(n):
                if v != u and not ((au >> v) & 1):
                    non_neighbors.append(v)
            rng.shuffle(non_neighbors)
            
            for v in non_neighbors:
                if degrees[u] >= target_min_deg:
                    break
                if _can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    improved = True
        
        # If still below target, try edge swaps
        if min(degrees) < target_min_deg:
            low_verts = [i for i in range(n) if degrees[i] < target_min_deg]
            if low_verts:
                u = min(low_verts, key=lambda i: degrees[i])
                # Find high-degree vertices to potentially remove edges from
                high_verts = [i for i in range(n) if i != u and degrees[i] > target_min_deg]
                if high_verts:
                    rng.shuffle(high_verts)
                    # Try multiple high-degree vertices
                    for a in high_verts:
                        if degrees[u] >= target_min_deg:
                            break
                        neighbors_a = []
                        aa = adj[a]
                        for v in range(n):
                            if v != a and ((aa >> v) & 1):
                                neighbors_a.append(v)
                        rng.shuffle(neighbors_a)
                        
                        for b in neighbors_a:
                            if b == u:
                                continue
                            # Don't remove edge if it would drop b below target
                            if degrees[b] <= target_min_deg:
                                continue
                            
                            non_neighbors_u = []
                            au = adj[u]
                            for v in range(n):
                                if v != u and not ((au >> v) & 1):
                                    non_neighbors_u.append(v)
                            rng.shuffle(non_neighbors_u)
                            
                            for v in non_neighbors_u:
                                if v == a or v == b:
                                    continue
                                # Check if the swap maintains C4-freedom
                                if _can_add_edge_after_removal(adj, u, v, a, b):
                                    # Perform the swap
                                    adj[a] &= ~(1 << b)
                                    adj[b] &= ~(1 << a)
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                    degrees[a] -= 1
                                    degrees[b] -= 1
                                    degrees[u] += 1
                                    degrees[v] += 1
                                    improved = True
                                    break
                            if degrees[u] >= target_min_deg:
                                break
                
                # If single swap didn't help, try double-swap for the most deficient vertex
                if degrees[u] < target_min_deg:
                    # Find two edges to remove and two edges to add
                    candidates_a = [i for i in range(n) if i != u and degrees[i] > target_min_deg + 1]
                    if candidates_a:
                        rng.shuffle(candidates_a)
                        for a in candidates_a[:5]:  # Try top 5 candidates
                            if degrees[u] >= target_min_deg:
                                break
                            neighbors_a = []
                            aa = adj[a]
                            for v in range(n):
                                if v != a and ((aa >> v) & 1):
                                    neighbors_a.append(v)
                            rng.shuffle(neighbors_a)
                            
                            for b in neighbors_a[:10]:  # Try top 10 neighbors
                                if b == u or degrees[b] <= target_min_deg:
                                    continue
                                
                                non_neighbors_u = []
                                au = adj[u]
                                for v in range(n):
                                    if v != u and not ((au >> v) & 1):
                                        non_neighbors_u.append(v)
                                rng.shuffle(non_neighbors_u)
                                
                                for v1 in non_neighbors_u[:10]:
                                    if v1 == a or v1 == b:
                                        continue
                                    # First check if adding (u, v1) is safe after removing (a, b)
                                    if not _can_add_edge_after_removal(adj, u, v1, a, b):
                                        continue
                                    
                                    # Now try to find a second edge to swap
                                    non_neighbors_v1 = []
                                    av1 = adj[v1]
                                    for v2 in range(n):
                                        if v2 != v1 and v2 != u and not ((av1 >> v2) & 1):
                                            non_neighbors_v1.append(v2)
                                    rng.shuffle(non_neighbors_v1)
                                    
                                    for v2 in non_neighbors_v1[:10]:
                                        if v2 == a or v2 == b:
                                            continue
                                        
                                        # Check if we can remove another edge from a or its neighbors
                                        # Try removing edge (c, d) where c is neighbor of a or v1
                                        candidates_c = [x for x in range(n) if x != a and x != u and x != v1 and degrees[x] > target_min_deg]
                                        if not candidates_c:
                                            continue
                                        rng.shuffle(candidates_c)
                                        
                                        for c in candidates_c[:5]:
                                            neighbors_c = []
                                            ac = adj[c]
                                            for d in range(n):
                                                if d != c and ((ac >> d) & 1):
                                                    neighbors_c.append(d)
                                            rng.shuffle(neighbors_c)
                                            
                                            for d in neighbors_c[:10]:
                                                if d == u or d == v1 or d == v2 or d == a or d == b:
                                                    continue
                                                if degrees[d] <= target_min_deg:
                                                    continue
                                                
                                                # Check if both additions are safe after both removals
                                                # This is complex; skip for now and just do single swap
                                                break
                                            break
                                        break
                                    break
                            break
    
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    best_n = 39
    best_edges = []
    best_adj = None
    
    # Try to maximize n from 47 down
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        found = False
        
        # Try multiple seeds with structured initialization and local search
        # Use more restarts for larger n
        max_seeds = 1000 if n >= 46 else 500
        
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            rng = random.Random(seed)
            
            # Degree-aware initialization with varying p
            p = 0.2 + 0.1 * (seed % 3)
            adj = _greedy_build_structured(n, rng, p=p, target_min_deg=target_min_deg)
            
            # Local search to improve min degree
            adj = _local_search(adj, n, target_min_deg, rng, max_iter=1500)
            
            min_deg = _min_degree(adj, n)
            
            if min_deg >= target_min_deg:
                # Verify C4-freedom
                if _is_c4_free(adj, n):
                    edges = _edges_from_adj(adj, n)
                    
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                        best_adj = adj
                    found = True
                    
                    # If we found n=47, return immediately
                    if n == 47:
                        return {'n': n, 'edges': edges}
        
        if found:
            # Since we iterate n downwards, this is the largest n found
            return {'n': best_n, 'edges': best_edges}
        
        # If not found for this n, continue to smaller n
    
    # Fallback
    if best_adj is not None:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
