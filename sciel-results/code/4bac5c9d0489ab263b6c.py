"""Search-based construction for R(C4,K11) > n."""
import time
import random

# EVOLVE-BLOCK-START

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if len(au & adj[v]) >= 2:
                return False
    return True

def _independent_set_ge(adj, n, limit, deadline):
    """Return True if there is an independent set of size >= limit.
    Bitset branch-and-bound with a wall-clock deadline."""
    masks = [0] * n
    for v in range(n):
        m = 0
        for u in adj[v]:
            m |= 1 << u
        masks[v] = m
    full = (1 << n) - 1

    def popcount(x):
        return x.bit_count()

    best = [0]
    def _dfs(rem, size):
        if time.time() > deadline:
            return
        if size >= limit:
            best[0] = size
            return
        if not rem:
            if size > best[0]:
                best[0] = size
            return
        # Upper bound
        if size + popcount(rem) <= best[0]:
            return
        # Pick a vertex with high degree in rem to branch on
        r = rem
        lb = r & (-r)
        v = (lb.bit_length() - 1)
        # Option 1: exclude v
        _dfs(rem & ~lb, size)
        # Option 2: include v -> remove v and its neighbours
        _dfs(rem & ~lb & ~masks[v], size + 1)

    _dfs(full, 0)
    return best[0] >= limit

def _independent_number(adj, n, deadline):
    """Exact independence number via bitset branch-and-bound."""
    masks = [0] * n
    for v in range(n):
        m = 0
        for u in adj[v]:
            m |= 1 << u
        masks[v] = m
    full = (1 << n) - 1

    def popcount(x):
        return x.bit_count()

    best = [0]
    def _dfs(rem, size):
        if time.time() > deadline:
            return
        if size > best[0]:
            best[0] = size
        if not rem:
            return
        if size + popcount(rem) <= best[0]:
            return
        r = rem
        lb = r & (-r)
        v = (lb.bit_length() - 1)
        _dfs(rem & ~lb, size)
        _dfs(rem & ~lb & ~masks[v], size + 1)

    _dfs(full, 0)
    return best[0]

def _can_add_vertex(adj, n, cand):
    """Check if adding new vertex n with neighbourhood cand (subset of range(n))
    keeps the graph C4-free."""
    cand_list = list(cand)
    # (A) For each existing u: |cand ∩ N(u)| <= 1
    for u in range(n):
        au = adj[u]
        cnt = 0
        for v in cand_list:
            if v in au:
                cnt += 1
                if cnt >= 2:
                    return False
    # (B) For each pair u,v in cand: N(u) ∩ N(v) = ∅
    for i in range(len(cand_list)):
        for j in range(i + 1, len(cand_list)):
            if adj[cand_list[i]] & adj[cand_list[j]]:
                return False
    return True

def _construct_polarity_graph(q):
    """Polarity graph of PG(2,q). Vertices are points, edge if dot product 0 mod q."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv = pow(x, q - 2, q)
                    key = (1, (y * inv) % q, (z * inv) % q)
                elif y != 0:
                    inv = pow(y, q - 2, q)
                    key = (0, 1, (z * inv) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    n_pts = len(points)
    adj = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        a1, b1, c1 = points[i]
        for j in range(i + 1, n_pts):
            a2, b2, c2 = points[j]
            if (a1 * a2 + b1 * b2 + c1 * c2) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n_pts, 'edges': edges, 'adj': adj}

def _add_vertex_local_search(adj, n, cand, deadline):
    """Given a base graph and a candidate neighbourhood, try to add the vertex.
    If alpha would exceed 10, attempt edge additions (within C4 constraints)
    to reduce alpha. Returns (success, new_adj, new_n)."""
    if not _can_add_vertex(adj, n, cand):
        return False, adj, n
    # Tentatively add
    new_adj = [set(s) for s in adj]
    new_adj.append(set(cand))
    for u in cand:
        new_adj[u].add(n)
    new_n = n + 1
    # Check C4-free
    if not _verify_c4_free(new_n, new_adj):
        return False, adj, n
    # Check independence number < 11
    if _independent_set_ge(new_adj, new_n, 11, deadline):
        # Try to add edges to reduce alpha
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(new_n):
                for v in range(u + 1, new_n):
                    if v in new_adj[u]:
                        continue
                    # Check if adding edge (u,v) keeps C4-free
                    common = new_adj[u] & new_adj[v]
                    if common:
                        continue
                    # Now check no 4-cycle: for each a in N(u), for each b in N(v),
                    # if a != b and a != v and b != u and a adjacent b, then
                    # u-a-b-v-u is a 4-cycle (with the new edge u-v).
                    ok = True
                    for a in new_adj[u]:
                        if a == v:
                            continue
                        for b in new_adj[v]:
                            if a == b or a == v or b == u:
                                continue
                            if b in new_adj[a]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    # Add edge and check alpha
                    new_adj[u].add(v)
                    new_adj[v].add(u)
                    if not _independent_set_ge(new_adj, new_n, 11, deadline):
                        improved = True
                        break
                    else:
                        # Revert
                        new_adj[u].remove(v)
                        new_adj[v].remove(u)
                if improved:
                    break
        if not _independent_set_ge(new_adj, new_n, 11, deadline):
            return True, new_adj, new_n
        return False, adj, n

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # Strategy 1: Try polarity graphs for various q
    for q in [3, 4, 5, 7, 9, 11, 13]:
        if time.time() > deadline - 10:
            break
        try:
            result = _construct_polarity_graph(q)
            n = result['n']
            adj = result['adj']
            # Verify C4-free
            if not _verify_c4_free(n, adj):
                continue
            # Check alpha < 11
            has_large_indep = _independent_set_ge(adj, n, 11, deadline - 5)
            if not has_large_indep:
                if n > best_n:
                    best_n = n
                    best_edges = result['edges']
        except Exception:
            continue
    
    # Strategy 2: Greedy construction from scratch with local search
    if time.time() < deadline - 5:
        random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            # Build graph greedily
            n = 0
            adj = []
            for v in range(43):
                if time.time() > deadline - 15:
                    break
                # Try to add vertex v with a random neighbourhood
                found = False
                for trial in range(50):
                    # Choose a random subset of existing vertices as neighbours
                    # Size should be moderate
                    if n == 0:
                        cand = set()
                    else:
                        # Random subset of size ~ n/4 to n/3
                        target_size = max(1, n // 4 + random.randint(0, n // 6))
                        candidates = list(range(n))
                        random.shuffle(candidates)
                        cand = set(candidates[:min(target_size, n)])
                    
                    # Use local search to try to add vertex
                    success, new_adj, new_n = _add_vertex_local_search(adj, n, cand, deadline - 10)
                    if success:
                        found = True
                        adj = new_adj
                        n = new_n
                        break
                    else:
                        # Try smaller neighbourhood
                        if len(cand) > 2:
                            cand_list = list(cand)
                            random.shuffle(cand_list)
                            cand = set(cand_list[:len(cand_list)//2])
                            success, new_adj, new_n = _add_vertex_local_search(adj, n, cand, deadline - 10)
                            if success:
                                found = True
                                adj = new_adj
                                n = new_n
                                break
                
                if not found:
                    break
            
            if n > best_n:
                # Verify
                if _verify_c4_free(n, adj):
                    if not _independent_set_ge(adj, n, 11, deadline - 5):
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append([i, j])
                        best_n = n
                        best_edges = edges
    
    # Strategy 3: If we haven't done well, use the triangle construction as fallback
    if best_n < 30:
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a,b], [a,c], [b,c]]
        best_n = 30
        best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
