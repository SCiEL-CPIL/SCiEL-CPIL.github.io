import time
import random
import sys
import numpy as np

def _construct_polarity_graph(q):
    """Construct the polarity graph of PG(2, q). C4-free."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    adj = [0] * n_pts  # bitmask adjacency
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    edges = []
    for i in range(n_pts):
        m = adj[i]
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
    
    return n_pts, edges, adj


def _has_independent_set(adj, n, limit):
    """Check if graph has independent set of size >= limit. Uses branch and bound with bitmasks."""
    if n > 64:
        return True  # unlikely given n <= 43
    
    all_mask = (1 << n) - 1
    
    # Precompute non-neighbor masks for faster pruning
    non_nbr = [0] * n
    for i in range(n):
        # non_nbr[i] = all vertices not adjacent to i and not i itself
        non_nbr[i] = all_mask & ~(adj[i] | (1 << i))
    
    # Use iterative deepening / branch and bound
    # State: available_mask (vertices that can still be added to independent set)
    # We want to know if we can pick 'limit' vertices from available_mask such that no two are adjacent
    
    sys.setrecursionlimit(10000)
    
    # Precompute degree within available set for ordering
    def _search(avail, depth_needed):
        if depth_needed <= 0:
            return True
        if avail == 0:
            return False
        cnt = avail.bit_count()
        if cnt < depth_needed:
            return False
        
        # Greedy lower bound: try to greedily pick vertices
        # If greedy can't find depth_needed, prune
        greedy_avail = avail
        greedy_count = 0
        while greedy_avail and greedy_count < depth_needed:
            # Pick vertex with minimum degree in remaining
            best_v = -1
            best_deg = 1000
            m = greedy_avail
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                deg = (adj[v] & greedy_avail).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            if best_v == -1:
                break
            greedy_count += 1
            greedy_avail &= non_nbr[best_v]
        if greedy_count < depth_needed:
            return False
        
        # Pick a vertex with high degree in the available subgraph (to prune more)
        best_v = -1
        best_deg = -1
        m = avail
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = (adj[v] & avail).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= lsb
        
        if best_v == -1:
            return False
        
        v_bit = 1 << best_v
        
        # Option 1: Include best_v in independent set
        # Then we cannot include any neighbor of best_v
        new_avail = avail & non_nbr[best_v]
        if new_avail.bit_count() >= depth_needed - 1:
            if _search(new_avail, depth_needed - 1):
                return True
        
        # Option 2: Exclude best_v
        new_avail = avail & ~v_bit
        if _search(new_avail, depth_needed):
            return True
        
        return False
    
    return _search(all_mask, limit)


def _is_c4_free(adj, n):
    """Check C4-free: no two vertices have >= 2 common neighbors."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # has at least 2 bits set
                return False
    return True


def _try_add_vertex(adj, n, limit=11, deadline=None):
    """
    Try to add a new vertex to the graph. Returns (new_adj, new_n, edges_to_add) or (None, None, None).
    The new vertex must have neighbors S such that:
    1. For all u, v in S: adj[u] & adj[v] == 0 (C4-free condition)
    2. The resulting graph has no independent set of size >= limit
    """
    # Find a maximal set S of vertices with pairwise disjoint neighborhoods
    # This is equivalent to finding a clique in the "conflict" graph where
    # vertices u,v are connected if adj[u] & adj[v] != 0
    
    # Greedy: try random orderings, build S greedily
    candidates = list(range(n))
    
    best_S = set()
    best_result = None
    
    # Try multiple strategies for building S
    for attempt in range(300):
        if deadline and time.time() > deadline - 5:
            break
        
        # Different strategies based on attempt number
        if attempt % 3 == 0:
            random.shuffle(candidates)
        elif attempt % 3 == 1:
            # Sort by degree ascending (low degree vertices less likely to conflict)
            candidates.sort(key=lambda v: (adj[v]).bit_count())
        else:
            # Sort by degree descending
            candidates.sort(key=lambda v: (adj[v]).bit_count(), reverse=True)
        
        S = set()
        for v in candidates:
            valid = True
            for u in S:
                if adj[u] & adj[v]:
                    valid = False
                    break
            if valid:
                S.add(v)
        
        if len(S) <= len(best_S):
            continue
        best_S = S.copy()
        
        # Try adding vertex with neighbors S
        new_n = n + 1
        new_adj = adj[:] + [0]
        new_adj[n] = 0
        for u in S:
            new_adj[n] |= (1 << u)
            new_adj[u] |= (1 << n)
        
        # Check independent set
        if not _has_independent_set(new_adj, new_n, limit):
            edges_to_add = [[n, u] for u in S]
            return new_adj, new_n, edges_to_add
    
    # If we didn't find a valid S with the best greedy, try the best_S found
    if best_S:
        new_n = n + 1
        new_adj = adj[:] + [0]
        new_adj[n] = 0
        for u in best_S:
            new_adj[n] |= (1 << u)
            new_adj[u] |= (1 << n)
        
        if not _has_independent_set(new_adj, new_n, limit):
            edges_to_add = [[n, u] for u in best_S]
            return new_adj, new_n, edges_to_add
    
    return None, None, None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_n = 0
    best_edges = []
    
    # Start with q=5 polarity graph (31 vertices)
    n, edges, adj = _construct_polarity_graph(5)
    
    if not _is_c4_free(adj, n):
        return {'n': 0, 'edges': []}
    
    if _has_independent_set(adj, n, 11):
        return {'n': 0, 'edges': []}
    
    best_n = n
    best_edges = [e[:] for e in edges]
    
    # Try to add vertices
    current_adj = adj[:]
    current_n = n
    current_edges = [e[:] for e in edges]
    
    while time.time() < deadline - 25:
        result = _try_add_vertex(current_adj, current_n, limit=11, deadline=deadline)
        if result[0] is not None:
            current_adj, current_n, new_edges = result
            current_edges.extend(new_edges)
            if current_n > best_n:
                best_n = current_n
                best_edges = [e[:] for e in current_edges]
        else:
            # Try a few more random restarts before giving up
            found = False
            for _ in range(50):
                if time.time() > deadline - 15:
                    break
                result = _try_add_vertex(current_adj, current_n, limit=11, deadline=deadline)
                if result[0] is not None:
                    current_adj, current_n, new_edges = result
                    current_edges.extend(new_edges)
                    if current_n > best_n:
                        best_n = current_n
                        best_edges = [e[:] for e in current_edges]
                    found = True
                    break
            if not found:
                break
    
    return {'n': best_n, 'edges': best_edges}
