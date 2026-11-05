"""Combined polarity-graph seed + local-search construction for R(C4,K11)."""
import time
import random
from itertools import combinations

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
        # find vertex v in rem
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
    keeps the graph C4-free.
    Conditions:
      (A) For each existing u: |cand ∩ N(u)| <= 1
      (B) For each pair u,v in cand: N(u) ∩ N(v) = ∅
      (C) For each u in cand: u is not adjacent to itself (trivial) and
          the new vertex's neighbourhood is cand. Also the new vertex must
          not create C4 with existing structure beyond (A) and (B).
          Additionally, for the new vertex w and any existing u not in cand,
          w and u share at most 1 common neighbour: common = cand ∩ N(u),
          which is exactly condition (A).
          For two existing vertices, their common neighbours are unchanged
          unless one of them is in cand, in which case the new vertex w is a
          new common neighbour only if both are in cand; that is covered by
          the fact that w is adjacent to both, but we must ensure w and the
          existing vertex pair don't create a C4. A C4 w-u-v-w' requires
          another vertex w', which doesn't exist. So (A) and (B) suffice.
    """
    cand_list = list(cand)
    # (A)
    for u in range(n):
        au = adj[u]
        cnt = 0
        for v in cand_list:
            if v in au:
                cnt += 1
                if cnt >= 2:
                    return False
    # (B)
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
        # We can add edge (u,v) if u,v not adjacent and adding it keeps C4-free
        # and reduces alpha.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(new_n):
                for v in range(u + 1, new_n):
                    if v in new_adj[u]:
                        continue
                    # Check if adding edge (u,v) keeps C4-free
                    # New common neighbours of any pair: adding edge (u,v)
                    # creates a C4 if there exists w such that w adjacent to both
                    # u and v already? No. Adding edge (u,v) creates a 4-cycle
                    # if there is a path u-a-b-v of length 3, i.e., a in N(u)∩? 
                    # Actually a C4 is u-a-b-v-u. This exists if there is a
                    # vertex a adjacent to u and b adjacent to v and a-b edge,
                    # with a != v, b != u, and the cycle is simple.
                    # Simpler: after adding edge, check the two conditions for
                    # the affected pairs only. But to be safe, do a full check
                    # but it's O(n^2) per candidate which is expensive.
                    # Instead, use the condition: adding edge (u,v) is C4-safe
                    # iff |N(u) ∩ N(v)| == 0 (no common neighbour) AND for every
                    # w in N(u) and z in N(v), w != z and not (w adjacent z) ... 
                    # This is getting complex. Do a targeted check.
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
                            if b == u or b == a:
                                continue
                            if b in new_adj[a]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    # Try adding the edge
                    new_adj[u].add(v)
                    new_adj[v].add(u)
                    if not _independent_set_ge(new_adj, new_n, 11, deadline):
                        improved = True
                        break
                    else:
                        new_adj[u].remove(v)
                        new_adj[v].remove(u)
            if improved:
                break
        if _independent_set_ge(new_adj, new_n, 11, deadline):
            return False, adj, n
    return True, new_adj, new_n

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    best_n = 0
    best_edges = []

    # Strategy 1: polarity graph seeds
    for q in [3, 4, 5, 7, 9, 11, 13]:
        if time.time() > deadline - 5:
            break
        try:
            result = _construct_polarity_graph(q)
            n = result['n']
            adj = result['adj']
            if not _verify_c4_free(n, adj):
                continue
            if _independent_set_ge(adj, n, 11, deadline):
                continue
            if n > best_n:
                best_n = n
                best_edges = result['edges']
                # Local search: try to add more vertices
                if time.time() < deadline - 10:
                    random.seed(42)
                    cur_adj = [set(s) for s in adj]
                    cur_n = n
                    for _ in range(200):
                        if time.time() > deadline - 10:
                            break
                        if cur_n >= 43:
                            break
                        # Try random neighbourhoods
                        found = False
                        for trial in range(30):
                            if time.time() > deadline - 10:
                                break
                            if cur_n == 0:
                                cand = set()
                            else:
                                # moderate degree
                                target = max(1, cur_n // 5 + random.randint(0, cur_n // 10))
                                cands = list(range(cur_n))
                                random.shuffle(cands)
                                cand = set(cands[:min(target, cur_n)])
                            ok, new_adj, new_n = _add_vertex_local_search(
                                cur_adj, cur_n, cand, deadline)
                            if ok:
                                cur_adj = new_adj
                                cur_n = new_n
                                found = True
                                if cur_n > best_n:
                                    best_n = cur_n
                                    # build edges
                                    best_edges = []
                                    for i in range(cur_n):
                                        for j in cur_adj[i]:
                                            if i < j:
                                                best_edges.append([i, j])
                                break
                        if not found:
                            # try smaller neighbourhoods
                            for trial in range(20):
                                if time.time() > deadline - 10:
                                    break
                                if cur_n == 0:
                                    cand = set()
                                else:
                                    target = max(1, cur_n // 8)
                                    cands = list(range(cur_n))
                                    random.shuffle(cands)
                                    cand = set(cands[:min(target, cur_n)])
                                ok, new_adj, new_n = _add_vertex_local_search(
                                    cur_adj, cur_n, cand, deadline)
                                if ok:
                                    cur_adj = new_adj
                                    cur_n = new_n
                                    if cur_n > best_n:
                                        best_n = cur_n
                                        best_edges = []
                                        for i in range(cur_n):
                                            for j in cur_adj[i]:
                                                if i < j:
                                                    best_edges.append([i, j])
                                    break
        except Exception:
            continue

    # Strategy 2: greedy from scratch with random restarts
    if time.time() < deadline - 5:
        random.seed(123)
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            adj = []
            n = 0
            for _ in range(43):
                if time.time() > deadline - 10:
                    break
                found = False
                for trial in range(40):
                    if time.time() > deadline - 10:
                        break
                    if n == 0:
                        cand = set()
                    else:
                        target = max(1, n // 5 + random.randint(0, n // 10))
                        cands = list(range(n))
                        random.shuffle(cands)
                        cand = set(cands[:min(target, n)])
                    ok, new_adj, new_n = _add_vertex_local_search(
                        adj, n, cand, deadline)
                    if ok:
                        adj = new_adj
                        n = new_n
                        found = True
                        if n > best_n:
                            best_n = n
                            best_edges = []
                            for i in range(n):
                                for j in adj[i]:
                                    if i < j:
                                        best_edges.append([i, j])
                        break
                if not found:
                    break

    # Fallback: disjoint triangles (n=30, alpha=10)
    if best_n < 30:
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
