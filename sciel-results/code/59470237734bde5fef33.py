"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of PG(2, q).
    Vertices are points in PG(2, q).
    Two points are adjacent if they are orthogonal with respect to the polarity.
    The graph is C4-free.
    """
    # Generate representatives of points in PG(2, q)
    reps = []
    # Case 1: x != 0 -> normalize x=1
    for y in range(q):
        for z in range(q):
            rep = (1, y, z)
            reps.append(rep)
    # Case 2: x == 0, y != 0 -> normalize y=1
    for z in range(q):
        rep = (0, 1, z)
        reps.append(rep)
    # Case 3: x == 0, y == 0, z != 0 -> normalize z=1
    rep = (0, 0, 1)
    reps.append(rep)
    
    n = len(reps)
    # Polarity: u . v = u0*v0 + u1*v1 - u2*v2 = 0 (mod q)
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        u = reps[i]
        for j in range(i + 1, n):
            v = reps[j]
            val = (u[0]*v[0] + u[1]*v[1] - u[2]*v[2]) % q
            if val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
                
    return n, edges, adj

def _check_no_c4(n, edges):
    """Check that no two vertices have two common neighbours."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return False
    return True

def _max_independent_set_size(n, adj, deadline=None):
    """
    Estimate the size of the maximum independent set.
    Uses a greedy heuristic.
    """
    if deadline and time.time() > deadline:
        return n
    
    remaining = set(range(n))
    is_size = 0
    while remaining:
        min_deg = n
        min_v = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        if min_v is None:
            break
        is_size += 1
        remaining.remove(min_v)
        for u in list(adj[min_v]):
            if u in remaining:
                remaining.remove(u)
    return is_size

def _has_independent_set_of_size(n, adj, target, deadline=None):
    """
    Exact check for independent set of size 'target'.
    Returns True if found.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    
    all_mask = (1 << n) - 1
    start_t = time.time()
    
    def branch(rem, curr):
        if time.time() - start_t > 5:
            return False
        if curr >= target:
            return True
        if rem == 0:
            return False
        if curr + rem.bit_count() < target:
            return False
            
        # Pick vertex with max degree in rem to prune
        temp = rem
        v_idx = -1
        max_deg = -1
        while temp:
            lsb = temp & -temp
            v = (lsb.bit_length() - 1)
            deg = (masks[v] & rem).bit_count()
            if deg > max_deg:
                max_deg = deg
                v_idx = v
            temp -= lsb
            
        if v_idx == -1:
            return False
        v_bit = 1 << v_idx
        
        # Include v
        if branch(rem & ~(masks[v_idx] | v_bit), curr + 1):
            return True
        # Exclude v
        if branch(rem & ~v_bit, curr):
            return True
        return False
        
    return branch(all_mask, 0)

def _try_greedy_edge_addition(n, deadline):
    """
    Try to construct a C4-free graph on n vertices with alpha < 11
    by greedily adding edges.
    Returns (n, edges) if successful, else None.
    """
    adj = [set() for _ in range(n)]
    edges = []
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    # We'll try multiple random orderings
    for attempt in range(3):
        if time.time() > deadline:
            break
            
        adj = [set() for _ in range(n)]
        edges = []
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            # Check if adding edge (u, v) creates a C4
            # A C4 is created if there exists a vertex w such that
            # w is adjacent to both u and v, and there exists another
            # vertex w' adjacent to both u and v.
            # Actually, C4-free means no two vertices share more than one common neighbor.
            # Adding edge (u,v) creates a C4 iff there exists w such that
            # w ~ u and w ~ v, AND there's already a path of length 2 between u and v via some w'.
            # Wait: adding edge (u,v) to a C4-free graph creates a C4 iff
            # u and v already have a common neighbor w (then u-w-v-u is a triangle, not C4)
            # No. A C4 is a cycle of length 4.
            # Adding edge (u,v) creates a C4 iff there is a path of length 3 from u to v
            # that doesn't use edge (u,v). That is, u-a-b-v where a~b, a~u, b~v, and (u,v) not in graph.
            # So we need to check: does there exist a path u-a-b-v?
            # This is equivalent to: is there a vertex a in N(u) and a vertex b in N(v) such that a~b?
            
            # More efficient: for each w in N(u), check if any neighbor of w is in N(v).
            # But we need to be careful about the structure.
            
            # Actually, the condition for C4-freeness is:
            # For any two vertices x, y, |N(x) ∩ N(y)| ≤ 1.
            # Adding edge (u,v) affects the common neighbors of pairs involving u and v.
            # Specifically, for any w, if w ~ u and w ~ v, then after adding (u,v),
            # u and v now share w as a common neighbor. But that's fine as long as they don't
            # share TWO common neighbors.
            # Wait, no. The condition is about ANY pair of vertices.
            # Adding edge (u,v) means:
            # 1. u and v are now adjacent.
            # 2. For any w, if w ~ u and w ~ v, then u and v now have w as a common neighbor.
            #    But u and v might already have other common neighbors? No, because if they had
            #    a common neighbor before, that would mean a path of length 2, which is fine.
            #    The issue is: after adding (u,v), do any two vertices now have ≥2 common neighbors?
            
            # Let's think differently. In a C4-free graph, for any edge (a,b) being added,
            # we need to ensure that no new C4 is formed. A C4 containing the new edge (u,v)
            # would be u-v-w-x-u where v~w, w~x, x~u, and (u,v) is the new edge.
            # So we need to check: is there a path of length 3 from v to u (not using edge (u,v))?
            # That is, does there exist w such that v~w, and x such that w~x, x~u?
            # Equivalently: is there a vertex w in N(v) such that N(w) ∩ N(u) ≠ ∅?
            
            creates_c4 = False
            for w in adj[v]:
                # Check if any neighbor of w is in N(u)
                for x in adj[w]:
                    if x in adj[u]:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if not creates_c4:
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        # Check if we have a valid graph
        if len(edges) > 0:
            # Quick check: is alpha < 11?
            alpha_est = _max_independent_set_size(n, adj, deadline)
            if alpha_est < 11:
                # Verify C4-freeness
                if _check_no_c4(n, edges):
                    return n, edges
    
    return None

def _remove_vertices_to_reduce_alpha(n, edges, adj, target_alpha, deadline):
    """
    Remove vertices from the graph until alpha < target_alpha.
    Returns (new_n, new_edges, new_adj) or None if timeout.
    """
    active = set(range(n))
    current_adj = [set(s) for s in adj]
    
    max_iters = 200
    for _ in range(max_iters):
        if time.time() > deadline:
            return None
        
        # Check alpha
        active_list = list(active)
        k = len(active_list)
        if k <= 39:  # Can't beat current best
            break
        
        # Build subgraph adj
        idx_map = {v: i for i, v in enumerate(active_list)}
        sub_adj = [set() for _ in range(k)]
        for i in range(k):
            v = active_list[i]
            for u in current_adj[v]:
                if u in active:
                    sub_adj[i].add(idx_map[u])
        
        # Check if alpha < 11
        if not _has_independent_set_of_size(k, sub_adj, 11, deadline=deadline):
            # Valid!
            new_edges = []
            for i in range(k):
                v = active_list[i]
                for u in current_adj[v]:
                    if u in active and idx_map[u] > i:
                        new_edges.append([v, u])
            return k, new_edges, sub_adj
        
        # Need to remove a vertex. Find an IS of size 11.
        # Use a simple heuristic: remove the vertex with lowest degree in the subgraph
        # that is part of many independent sets.
        # Approximate: remove a vertex with low degree.
        
        # Find a vertex to remove
        min_deg = k
        rem_v = None
        for i in range(k):
            deg = len(sub_adj[i])
            if deg < min_deg:
                min_deg = deg
                rem_v = i
        
        if rem_v is not None:
            active.remove(active_list[rem_v])
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s margin
    
    best_n = 0
    best_edges = []
    
    # Strategy 1: Try greedy edge addition on 43 vertices
    result = _try_greedy_edge_addition(43, deadline)
    if result:
        n, edges = result
        if n > best_n:
            best_n = n
            best_edges = edges
    
    # Strategy 2: Use polarity graph of PG(2, 8) and remove vertices
    if time.time() < deadline - 20:
        try:
            n8, edges8, adj8 = _construct_polarity_graph(8)
            # n8 = 73
            if n8 > best_n:
                res = _remove_vertices_to_reduce_alpha(n8, edges8, adj8, 11, deadline)
                if res:
                    k, new_edges, _ = res
                    if k > best_n:
                        best_n = k
                        best_edges = new_edges
        except Exception:
            pass
    
    # Strategy 3: Use polarity graph of PG(2, 5) as fallback
    if best_n < 31:
        try:
            n5, edges5, _ = _construct_polarity_graph(5)
            if _check_no_c4(n5, edges5):
                adj5 = [set() for _ in range(n5)]
                for u, v in edges5:
                    adj5[u].add(v)
                    adj5[v].add(u)
                if _max_independent_set_size(n5, adj5) < 11:
                    best_n = n5
                    best_edges = edges5
        except Exception:
            pass
    
    # Fallback: 10 disjoint triangles (n=30)
    if best_n < 30:
        edges_30 = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges_30 += [[a,b], [a,c], [b,c]]
        best_n = 30
        best_edges = edges_30
    
    # Final safety check
    if best_n > 0 and best_edges:
        if not _check_no_c4(best_n, best_edges):
            # Reset to fallback
            edges_30 = []
            for t in range(10):
                a, b, c = 3*t, 3*t+1, 3*t+2
                edges_30 += [[a,b], [a,c], [b,c]]
            best_n = 30
            best_edges = edges_30
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
