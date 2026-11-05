"""Combined: polarity graph construction + aggressive local search to reduce independence number."""
import time
import random
import sys

# EVOLVE-BLOCK-START
_DEADLINE = 118.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _check_c4(adj):
    """adj: list of sets. Return True if no C4 (no two vertices with >=2 common neighbours)."""
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _alpha_greedy_lb(adj):
    """Greedy lower bound on independence number."""
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
    """Return True if there exists an independent set of size >= limit."""
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
        # Upper bound: current_size + popcount(remaining_mask)
        if current_size + bin(remaining_mask).count("1") < limit:
            return False
        # Pick a vertex with highest degree in remaining
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
        # Branch: include best_v
        new_rem = remaining_mask & ~(masks[best_v] | (1 << best_v))
        if branch(new_rem, current_size + 1):
            return True
        # Branch: exclude best_v
        new_rem2 = remaining_mask & ~(1 << best_v)
        if branch(new_rem2, current_size):
            return True
        return False

    full = (1 << n) - 1
    return branch(full, 0)

def _has_indep_11(adj):
    """Check if there is an independent set of size 11."""
    n = len(adj)
    if n < 11:
        return False
    return _max_independent_set(adj, 11)

def _build_polarity_graph_q5():
    """Polarity graph of PG(2,5): 31 vertices, C4-free."""
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv = pow(x, 3, 5)
                    cx, cy, cz = 1, (y*inv) % 5, (z*inv) % 5
                elif y != 0:
                    inv = pow(y, 3, 5)
                    cx, cy, cz = 0, 1, (z*inv) % 5
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
            s = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) % 5
            if s == 0:
                adj[i].add(j)
                adj[j].add(i)
    if not _check_c4(adj):
        adj = []
        for t in range(10):
            adj.append(set())
            adj.append(set())
            adj.append(set())
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            adj[a].add(b); adj[b].add(a)
            adj[a].add(c); adj[c].add(a)
            adj[b].add(c); adj[c].add(b)
    return adj

def _build_polarity_edges(q):
    """Build polarity graph of PG(2,q) as incidence graph (points + lines)."""
    m = q * q + q + 1
    n = 2 * m
    pts = []
    for x in range(q):
        for y in range(q):
            pts.append((x, y, 1))
    for x in range(q):
        pts.append((x, 1, 0))
    pts.append((1, 0, 0))

    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    lines = set()
    for i in range(m):
        for j in range(i+1, m):
            L = cross(pts[i], pts[j])
            lines.add(L)
    line_list = list(lines)

    def incident(p, L):
        return (p[0]*L[0] + p[1]*L[1] + p[2]*L[2]) % q == 0

    adj = [set() for _ in range(n)]
    for i in range(m):
        for j in range(m):
            if incident(pts[i], line_list[j]):
                u, v = i, m + j
                adj[u].add(v); adj[v].add(u)
    return n, adj

def _has_c4_edge(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4."""
    if u in adj[v]:
        return False
    common = adj[u] & adj[v]
    for a in common:
        for b in common:
            if a != b and b in adj[a]:
                return True
    return False

def _local_search_reduce_alpha(adj, target_n, max_iters=5000):
    """Try to reduce independence number below 11 by adding edges, maintaining C4-free."""
    n = len(adj)
    adj = [set(s) for s in adj]
    best_adj = [set(s) for s in adj]
    best_alpha = _alpha_greedy_lb(adj)
    
    # First, try to grow to target_n by adding vertices
    for _ in range(target_n - n):
        if _deadline_hit():
            break
        added = False
        for attempt in range(100):
            new_idx = n
            new_neigh = set()
            # Random subset with moderate density
            for v in range(n):
                if random.random() < 0.25:
                    new_neigh.add(v)
            # Check C4 condition: for any two neighbors a,b of new vertex, N(a)∩N(b) must be empty
            valid = True
            neigh_list = list(new_neigh)
            for i in range(len(neigh_list)):
                for j in range(i+1, len(neigh_list)):
                    a, b = neigh_list[i], neigh_list[j]
                    if adj[a] & adj[b]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            for v in range(n):
                if v in new_neigh:
                    adj[v].add(new_idx)
            adj.append(set(new_neigh))
            n = new_idx + 1
            added = True
            break
        if not added:
            # Try with fewer neighbors
            for attempt in range(50):
                new_idx = n
                new_neigh = set()
                for v in range(n):
                    if random.random() < 0.1:
                        new_neigh.add(v)
                valid = True
                neigh_list = list(new_neigh)
                for i in range(len(neigh_list)):
                    for j in range(i+1, len(neigh_list)):
                        a, b = neigh_list[i], neigh_list[j]
                        if adj[a] & adj[b]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                for v in range(n):
                    if v in new_neigh:
                        adj[v].add(new_idx)
                adj.append(set(new_neigh))
                n = new_idx + 1
                added = True
                break
    
    # Now aggressively add edges to reduce independence number
    for it in range(max_iters):
        if _deadline_hit():
            break
        # Find a large independent set
        remaining = set(range(n))
        indep = []
        while remaining:
            v = min(remaining, key=lambda x: len(adj[x] & remaining))
            indep.append(v)
            remaining.difference_update(adj[v])
            remaining.discard(v)
        
        if len(indep) <= 10:
            break
        
        # Try to add an edge within the independent set that doesn't create C4
        added = False
        random.shuffle(indep)
        for i in range(len(indep)):
            for j in range(i+1, len(indep)):
                u, v = indep[i], indep[j]
                if v in adj[u]:
                    continue
                if _has_c4_edge(adj, n, u, v):
                    continue
                adj[u].add(v); adj[v].add(u)
                added = True
                break
            if added:
                break
        
        if not added:
            # Try random pairs
            for _ in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or v in adj[u]:
                    continue
                if _has_c4_edge(adj, n, u, v):
                    continue
                adj[u].add(v); adj[v].add(u)
                added = True
                break
        if not added:
            break
    
    # Update best if improved
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
    
    # Try multiple starting constructions
    constructions = []
    
    # 1. Polarity graph of PG(2,5) - 31 vertices
    adj1 = _build_polarity_graph_q5()
    constructions.append(adj1)
    
    # 2. Incidence graph of PG(2,4) - 42 vertices
    try:
        n2, adj2 = _build_polarity_edges(4)
        constructions.append(adj2)
    except Exception:
        pass
    
    # 3. Incidence graph of PG(2,3) - 28 vertices
    try:
        n3, adj3 = _build_polarity_edges(3)
        constructions.append(adj3)
    except Exception:
        pass
    
    # 4. Ten disjoint triangles - 30 vertices (fallback)
    adj4 = []
    for t in range(10):
        adj4.append(set())
        adj4.append(set())
        adj4.append(set())
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        adj4[a].add(b); adj4[b].add(a)
        adj4[a].add(c); adj4[c].add(a)
        adj4[b].add(c); adj4[c].add(b)
    constructions.append(adj4)
    
    # For each construction, try to improve
    for adj in constructions:
        if _deadline_hit():
            break
        if len(adj) < 11:
            continue
        
        # Verify C4-free
        if not _check_c4(adj):
            continue
        
        # Try local search to reduce alpha
        target = max(len(adj), 39)
        best_adj, best_alpha = _local_search_reduce_alpha(adj, target, max_iters=3000)
        
        if best_alpha >= 11:
            continue
        
        # Verify exact alpha
        if _has_indep_11(best_adj):
            continue
        
        n = len(best_adj)
        if n > best_n:
            best_n = n
            edges = []
            for i in range(n):
                for j in best_adj[i]:
                    if i < j:
                        edges.append([i, j])
            best_result = {'n': n, 'edges': edges}
    
    # Fallback to seed if nothing better
    if best_result is None or best_n < 30:
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    return best_result
# EVOLVE-BLOCK-END
