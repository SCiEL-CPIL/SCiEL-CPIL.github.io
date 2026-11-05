import time
import random
import sys

_DEADLINE = 118.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _check_c4(adj):
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _alpha_greedy_lb(adj):
    n = len(adj)
    remaining = set(range(n))
    alpha = 0
    while remaining:
        best_v = None
        best_deg = 10**9
        for v in remaining:
            d = len(adj[v] & remaining)
            if d < best_deg:
                best_deg = d
                best_v = v
        alpha += 1
        remaining.discard(best_v)
        remaining -= adj[best_v]
    return alpha

def _max_independent_set(adj, limit=11):
    n = len(adj)
    masks = []
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks.append(m)

    sys.setrecursionlimit(10000)

    def branch(remaining_mask, current_size):
        if current_size >= limit:
            return True
        if remaining_mask == 0:
            return False
        if current_size + bin(remaining_mask).count("1") < limit:
            return False
        rem = remaining_mask
        best_v = -1
        best_deg = -1
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            deg = bin(remaining_mask & masks[v]).count("1")
            if deg > best_deg:
                best_deg = deg
                best_v = v
        new_rem = remaining_mask & ~(masks[best_v] | (1 << best_v))
        if branch(new_rem, current_size + 1):
            return True
        new_rem2 = remaining_mask & ~(1 << best_v)
        if branch(new_rem2, current_size):
            return True
        return False

    full = (1 << n) - 1
    return branch(full, 0)

def _has_indep_11(adj):
    n = len(adj)
    if n < 11:
        return False
    return _max_independent_set(adj, 11)

def _build_polarity_graph_q(q):
    """Polarity graph of PG(2,q) using x*x' + y*y' + z*z' = 0 mod q."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv = pow(x, q-2, q)
                    cx, cy, cz = 1, (y*inv) % q, (z*inv) % q
                elif y != 0:
                    inv = pow(y, q-2, q)
                    cx, cy, cz = 0, 1, (z*inv) % q
                else:
                    cx, cy, cz = 0, 0, 1
                key = (cx, cy, cz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    npts = len(points)
    adj = [set() for _ in range(npts)]
    for i in range(npts):
        for j in range(i+1, npts):
            a = points[i]
            b = points[j]
            s = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) % q
            if s == 0:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _can_add_edge(adj, u, v):
    """Check if adding edge (u,v) keeps the graph C4-free."""
    # After adding u-v, for any w in N(u) and x in N(v), we need w != x
    # Actually: adding edge (u,v) creates a C4 if there exists w such that
    # w is adjacent to both u and v (making u-w-v a path of length 2),
    # AND there exists another path of length 2 from u to v through some x.
    # More precisely: C4 is u-a-b-v-u. So we need a in N(u)∩N(v) and b in N(u)∩N(v) with a≠b.
    # Wait, that's not right either. Let me think again.
    # A C4 is a 4-cycle: v1-v2-v3-v4-v1.
    # If we add edge (u,v), a C4 could be u-v-w-x-u where w~v, w~x, x~u, and x≠v, w≠u.
    # So: exists w in N(v) and x in N(u) such that w~x, w≠u, x≠v.
    # Equivalently: N(u) ∩ N(v) must not contain any edge? No...
    # Actually: the new edge (u,v) plus existing edges. A C4 using the new edge:
    # u-v-w-x-u where w is adjacent to v, x is adjacent to u, and w is adjacent to x.
    # So we need: exists w in N(v)\{u} and x in N(u)\{v} such that w~x.
    # This is equivalent to: (N(u) \ {v}) and (N(v) \ {u}) have an edge between them.
    # Or: there exists w in N(v) and x in N(u) with w≠u, x≠v, and w~x.
    # Note that w and x could be the same vertex only if w~w which is impossible.
    # So: for all w in N(v) and x in N(u) with w≠u, x≠v, we must have w not adjacent to x.
    # Equivalently: N(u) \ {v} and N(v) \ {u} must be an independent set (no edges between them).
    # Wait, that's not quite right. We need NO edge between any w in N(v)\{u} and any x in N(u)\{v}.
    # But that's a strong condition. Let me reconsider.
    # Actually, the standard C4-free condition is: no two vertices have two common neighbors.
    # Adding edge (u,v): for any pair of vertices (a,b), the number of common neighbors can change.
    # The only pairs affected are those involving u or v.
    # For pair (u,v): they now share no common neighbors necessarily, but the edge is added.
    # For pair (u,w) where w in N(v): u and w now both adjacent to v. If they were already adjacent to some common neighbor, that's fine.
    # The real check: adding (u,v) creates a C4 iff there exist w,x such that u~w, w~x, x~v, and the path u-w-x-v is not using the new edge.
    # But w~x must be an existing edge. So: exists w in N(u) and x in N(v) with w~x, w≠v, x≠u.
    # This is: the bipartite subgraph between N(u)\{v} and N(v)\{u} has at least one edge.
    nu = adj[u] - {v}
    nv = adj[v] - {u}
    for w in nv:
        if adj[w] & nu:
            return False
    return True

def _local_search_reduce_alpha(adj, max_iters=5000):
    """Local search: try to add edges to reduce independence number while maintaining C4-free."""
    n = len(adj)
    adj = [set(s) for s in adj]
    best_adj = [set(s) for s in adj]
    best_alpha = _alpha_greedy_lb(adj)
    
    for it in range(max_iters):
        if _deadline_hit():
            break
        
        # Find a greedy independent set
        remaining = set(range(n))
        indep = []
        while remaining:
            v = min(remaining, key=lambda x: len(adj[x] & remaining))
            indep.append(v)
            remaining.difference_update(adj[v])
            remaining.discard(v)
        
        if len(indep) <= 10:
            break
        
        # Try to add an edge between two vertices in the independent set
        added = False
        random.shuffle(indep)
        for i in range(len(indep)):
            if _deadline_hit():
                break
            for j in range(i+1, len(indep)):
                u, v = indep[i], indep[j]
                if v in adj[u]:
                    continue
                if _can_add_edge(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
            if added:
                break
        
        if not added:
            # Try random edge additions
            for _ in range(20):
                if _deadline_hit():
                    break
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or v in adj[u]:
                    continue
                if _can_add_edge(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
        
        if not added:
            break
        
        cur_alpha = _alpha_greedy_lb(adj)
        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best_adj = [set(s) for s in adj]
    
    return best_adj, best_alpha

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 118.0
    
    best_result = None
    best_n = 0
    
    # Try q=7 and q=9 polarity graphs with subset selection and local search
    for q in [9, 7]:
        if _deadline_hit():
            break
        try:
            full_adj = _build_polarity_graph_q(q)
            n_full = len(full_adj)
            
            # First try the full graph
            if _alpha_greedy_lb(full_adj) < 11:
                if not _has_indep_11(full_adj):
                    n = len(full_adj)
                    if n > best_n:
                        best_n = n
                        edges = []
                        for i in range(n):
                            for j in full_adj[i]:
                                if i < j:
                                    edges.append([i, j])
                        best_result = {'n': n, 'edges': edges}
            else:
                # Local search on full graph
                searched_adj, alpha = _local_search_reduce_alpha(full_adj, max_iters=3000)
                if alpha < 11:
                    if not _has_indep_11(searched_adj):
                        n = len(searched_adj)
                        if n > best_n:
                            best_n = n
                            edges = []
                            for i in range(n):
                                for j in searched_adj[i]:
                                    if i < j:
                                        edges.append([i, j])
                            best_result = {'n': n, 'edges': edges}
            
            # Try subsets of various sizes
            for n_target in range(min(n_full, 43), max(30, best_n), -1):
                if _deadline_hit():
                    break
                if n_target <= best_n:
                    break
                for attempt in range(30):
                    if _deadline_hit():
                        break
                    indices = random.sample(range(n_full), n_target)
                    sub_adj = [set() for _ in range(n_target)]
                    idx_map = {old: new for new, old in enumerate(indices)}
                    
                    for new_i in range(n_target):
                        old_i = indices[new_i]
                        for new_j in range(new_i+1, n_target):
                            old_j = indices[new_j]
                            if old_j in full_adj[old_i]:
                                sub_adj[new_i].add(new_j)
                                sub_adj[new_j].add(new_i)
                    
                    if _alpha_greedy_lb(sub_adj) < 11:
                        if not _has_indep_11(sub_adj):
                            if n_target > best_n:
                                best_n = n_target
                                edges = []
                                for i in range(n_target):
                                    for j in sub_adj[i]:
                                        if i < j:
                                            edges.append([i, j])
                                best_result = {'n': n_target, 'edges': edges}
                                break
                    else:
                        # Try local search to reduce alpha
                        searched_adj, alpha = _local_search_reduce_alpha(sub_adj, max_iters=1000)
                        if alpha < 11:
                            if not _has_indep_11(searched_adj):
                                if n_target > best_n:
                                    best_n = n_target
                                    edges = []
                                    for i in range(n_target):
                                        for j in searched_adj[i]:
                                            if i < j:
                                                edges.append([i, j])
                                    best_result = {'n': n_target, 'edges': edges}
                                    break
        except Exception:
            pass
    
    # Fallback to q=5 full graph
    if best_result is None or best_n < 31:
        try:
            adj1 = _build_polarity_graph_q(5)
            if _alpha_greedy_lb(adj1) < 11 and not _has_indep_11(adj1):
                n = len(adj1)
                if n > best_n:
                    best_n = n
                    edges = []
                    for i in range(n):
                        for j in adj1[i]:
                            if i < j:
                                edges.append([i, j])
                    best_result = {'n': n, 'edges': edges}
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to 30 vertices (10 disjoint triangles)
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    return best_result
