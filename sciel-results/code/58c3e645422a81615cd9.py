import time
import random
import sys

_DEADLINE = 115.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _build_polarity_graph_q(q):
    """Polarity graph of PG(2,q). Vertices are points, edge if orthogonal."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
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

def _max_independent_set_size(adj, limit=11):
    """Check if there's an independent set of size >= limit. Returns True if exists."""
    n = len(adj)
    if n < limit:
        return False
    
    # Build adjacency masks
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
    
    sys.setrecursionlimit(10000)
    
    def branch(remaining_mask, current_size):
        if current_size >= limit:
            return True
        if remaining_mask == 0:
            return False
        # Upper bound: if we can't reach limit, prune
        if current_size + bin(remaining_mask).count("1") < limit:
            return False
        
        # Choose vertex with max degree in remaining
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
        
        if best_v == -1:
            return False
        
        # Branch 1: include best_v
        new_rem = remaining_mask & ~(masks[best_v] | (1 << best_v))
        if branch(new_rem, current_size + 1):
            return True
        
        # Branch 2: exclude best_v
        new_rem2 = remaining_mask & ~(1 << best_v)
        if branch(new_rem2, current_size):
            return True
        
        return False
    
    full = (1 << n) - 1
    return branch(full, 0)

def _greedy_alpha_lb(adj):
    """Lower bound on independence number via greedy."""
    n = len(adj)
    remaining = set(range(n))
    alpha = 0
    while remaining:
        best_v = min(remaining, key=lambda x: len(adj[x] & remaining))
        alpha += 1
        remaining.discard(best_v)
        remaining -= adj[best_v]
    return alpha

def _has_c4_after_add(adj, u, v):
    """Check if adding edge (u,v) creates a C4."""
    common = adj[u] & adj[v]
    for a in common:
        for b in common:
            if a != b and b in adj[a]:
                return True
    return False

def _local_search_reduce_alpha(adj, n_target, max_iters=3000):
    """Try to reduce alpha by adding edges (maintaining C4-free)."""
    n = len(adj)
    adj = [set(s) for s in adj]
    best_adj = [set(s) for s in adj]
    best_alpha = _greedy_alpha_lb(adj)
    
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
        
        added = False
        random.shuffle(indep)
        
        # Try to add edges between pairs in the independent set
        for i in range(len(indep)):
            if _deadline_hit():
                break
            for j in range(i+1, len(indep)):
                u, v = indep[i], indep[j]
                if v in adj[u]:
                    continue
                if _has_c4_after_add(adj, u, v):
                    continue
                adj[u].add(v)
                adj[v].add(u)
                added = True
                break
            if added:
                break
        
        if not added:
            # Try random edge additions
            for _ in range(30):
                if _deadline_hit():
                    break
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or v in adj[u]:
                    continue
                if _has_c4_after_add(adj, u, v):
                    continue
                adj[u].add(v)
                adj[v].add(u)
                added = True
                break
        
        if not added:
            break
        
        cur_alpha = _greedy_alpha_lb(adj)
        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best_adj = [set(s) for s in adj]
            
    return best_adj, best_alpha

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 115.0
    
    best_result = None
    best_n = 0
    
    # Try polarity graphs for q=3,5,7
    for q in [7, 5, 3]:
        if _deadline_hit():
            break
        
        full_adj = _build_polarity_graph_q(q)
        n_full = len(full_adj)
        
        # Try to find subgraphs of decreasing size
        for n_target in range(min(43, n_full), 29, -1):
            if _deadline_hit():
                break
            if n_target <= best_n:
                continue
            
            found = False
            # Try multiple random subsets
            for attempt in range(30):
                if _deadline_hit():
                    break
                
                if attempt == 0:
                    indices = list(range(n_target))
                else:
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
                
                # Check if alpha < 11
                if _greedy_alpha_lb(sub_adj) >= 11:
                    continue
                
                if not _max_independent_set_size(sub_adj, 11):
                    if n_target > best_n:
                        best_n = n_target
                        edges = []
                        for i in range(n_target):
                            for j in sub_adj[i]:
                                if i < j:
                                    edges.append([i, j])
                        best_result = {'n': n_target, 'edges': edges}
                        found = True
                        break
                else:
                    # Try local search to reduce alpha
                    improved_sub, imp_alpha = _local_search_reduce_alpha(sub_adj, n_target, max_iters=1500)
                    if imp_alpha < 11 and not _max_independent_set_size(improved_sub, 11):
                        if n_target > best_n:
                            best_n = n_target
                            edges = []
                            for i in range(n_target):
                                for j in improved_sub[i]:
                                    if i < j:
                                        edges.append([i, j])
                            best_result = {'n': n_target, 'edges': edges}
                            found = True
                            break
            
            if found:
                break
    
    # Fallback: use q=3 polarity graph (13 vertices) or q=5 (31 vertices)
    if best_result is None or best_n < 31:
        adj_fb = _build_polarity_graph_q(5)
        n_fb = len(adj_fb)
        edges_fb = []
        for i in range(n_fb):
            for j in adj_fb[i]:
                if i < j:
                    edges_fb.append([i, j])
        return {'n': n_fb, 'edges': edges_fb}
    
    return best_result
