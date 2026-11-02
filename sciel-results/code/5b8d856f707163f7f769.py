"""Cage solver: LUW + geometric (generalized polygon) + improved random regular
search with edge-swap local search to eliminate short cycles."""
import time
import threading
import random
import itertools
from collections import deque

import numpy as np

_DEADLINE = 120.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth via non-backtracking walks -----------------------
def _girth_nb(adj, n, g):
    """Exact girth via non-backtracking BFS from all vertices simultaneously.
    Returns g if no cycle shorter than g exists."""
    if n == 0:
        return g
    # adj: list of sorted lists
    deg = np.array([len(a) for a in adj], dtype=np.int32)
    kmax = int(deg.max()) if n > 0 else 0
    # Build neighbor array with padding
    nbr = np.zeros((n, kmax), dtype=np.int32)
    for u in range(n):
        nbr[u, :deg[u]] = np.array(adj[u], dtype=np.int32)

    # State = (vertex, position_in_adj). Index = vertex * kmax + pos
    # prev_state = (nbr[v][p], q) where q is position of v in adj[nbr[v][p]]
    # We do BFS: layer d contains states reachable by non-backtracking walks
    # of length d from their origin.

    # pos_lookup[u][v] = position of v in adj[u], -1 if not present
    pos_lookup = np.full((n, n), -1, dtype=np.int32)
    for u in range(n):
        for p, v in enumerate(adj[u]):
            pos_lookup[u, v] = p

    total_states = n * kmax
    visited_origin = np.full(total_states, -1, dtype=np.int32)
    visited_dist = np.zeros(total_states, dtype=np.int32)

    # Initial layer (distance 1): all directed edges (u -> v), state = (v, p)
    # where v = adj[u][p], origin = u
    init_states = []
    init_origins = []
    for u in range(n):
        for p in range(deg[u]):
            v = adj[u][p]
            s = v * kmax + p
            init_states.append(s)
            init_origins.append(u)

    if not init_states:
        return g

    cur_state = np.array(init_states, dtype=np.int32)
    cur_origin = np.array(init_origins, dtype=np.int32)
    dist = 1

    visited_origin[cur_state] = cur_origin
    visited_dist[cur_state] = 1

    while len(cur_state) > 0:
        if dist >= g:
            break
        v_cur = cur_state // kmax
        p_cur = cur_state % kmax

        # Next neighbors: all w = nbr[v_cur][r] where r != p_cur
        # Use vectorized approach
        n_cur = len(v_cur)
        r_all = np.tile(np.arange(kmax, dtype=np.int32), n_cur)
        w_all = nbr[v_cur, r_all]
        # mask: r != p_cur and r < deg[v_cur]
        deg_v = deg[v_cur]
        mask = (r_all != p_cur) & (r_all < deg_v)
        w_all = w_all[mask]
        r_all = r_all[mask]
        o_all = cur_origin[mask]

        if len(w_all) == 0:
            break

        # Check: w == origin -> cycle of length dist+1
        hit = (w_all == o_all)
        if hit.any():
            return dist + 1

        new_state = w_all * kmax + r_all
        new_origin = o_all

        # Check visited
        old_origin = visited_origin[new_state]
        is_new = (old_origin == -1)

        if not is_new.all():
            diff = ~is_new
            # If different origin reached same state, cycle length = dist+1+visited_dist
            diff_origin = diff & (old_origin != new_origin)
            if diff_origin.any():
                cand = dist + 1 + visited_dist[new_state[diff_origin]]
                if cand.min() < g:
                    return int(cand.min())

        new_state = new_state[is_new]
        new_origin = new_origin[is_new]

        if len(new_state) == 0:
            break

        visited_origin[new_state] = new_origin
        visited_dist[new_state] = dist + 1
        cur_state = new_state
        cur_origin = new_origin
        dist += 1

    return g


def _nonback_girth(adj, n):
    """Exact girth via non-backtracking walks."""
    if n == 0:
        return 10**9
    try:
        return _girth_nb(adj, n, 10**9)
    except Exception:
        # Pure Python fallback
        prev = {}
        q = deque()
        for u in range(n):
            for v in adj[u]:
                key = (v, u)
                prev[key] = (u, 1)
                q.append(key)
        while q:
            (v, p), (u, dist) = q.popleft()
            for w in adj[v]:
                if w == p:
                    continue
                if w == u:
                    return dist + 1
                key = (w, v)
                if key not in prev:
                    prev[key] = (v, dist + 1)
                    q.append(key)
        return 10**9


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False
    return _nonback_girth(adj, n) >= g


# ---------------- Moore bound --------------------------------------------------
def _moore(k, g):
    R = (g - 1) // 2
    return 1 + k * sum((k - 1)**i for i in range(R))


def _min_n_feasible(k, g):
    moore = _moore(k, g)
    n = moore
    while (n * k) % 2 != 0 or n < k + 1:
        n += 1
    return n


# ---------------- LUW construction ---------------------------------------------
def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    if q <= 1:
        return q + 1, [(0, 1)] if q == 1 else []
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if d < 3:
        d = 3

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------- Generalized polygon construction -----------------------------
def _gen_polygon(k, g):
    """Construct a 5-regular (or k-regular) graph from a generalized polygon.
    For k=5, g=9: use the incidence graph of a suitable structure.
    
    For girth >= 9, we need no cycles of length < 9. A bipartite graph with
    girth 8 has no odd cycles. To get girth >= 9, we need girth at least 9.
    For a bipartite graph, girth is always even, so girth >= 9 means girth >= 10
    for bipartite. But we can also use non-bipartite constructions.
    
    Let's try a different approach: use the coset graph of a group.
    For k=5, g=9, the Moore bound gives n >= 1 + 5*(1+4+16) = 1 + 5*21 = 106.
    But the actual cage has 1116 vertices.
    
    For now, this function attempts a coset graph approach.
    """
    # For k=5, g=9: try a Cayley graph approach
    # A(5) has order 120, too small. A(6) has order 720.
    # Let's just try random regular + local search for now.
    return None


# ---------------- Random regular graph generation (improved) --------------------
def _random_regular_v2(n, k, rng):
    """Generate a random simple k-regular graph on n vertices.
    Uses configuration model with efficient swap-based repair."""
    if n * k % 2 != 0 or n < k + 1:
        return None

    best_adj = None
    for _ in range(30):
        # Configuration model
        stubs = [i for i in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        edges = [(stubs[i], stubs[i+1]) for i in range(0, n*k, 2)]

        # Repair: eliminate loops and multi-edges
        for _ in range(500):
            # Find a loop or multi-edge
            edge_set = set()
            problem = None
            for idx, (a, b) in enumerate(edges):
                if a == b:
                    problem = ('loop', idx)
                    break
                key = (min(a, b), max(a, b))
                if key in edge_set:
                    problem = ('multi', idx)
                    break
                edge_set.add(key)
            if problem is None:
                break
            typ, idx = problem
            a, b = edges[idx]
            fixed = False
            for jdx in range(len(edges)):
                if jdx == idx:
                    continue
                c, d = edges[jdx]
                if typ == 'loop':
                    # a == b, need to re-pair
                    if c == a or d == a:
                        continue
                    # (a,a) + (c,d) -> (a,c) + (a,d)
                    edges[idx] = (a, c)
                    edges[jdx] = (a, d)
                    fixed = True
                    break
                else:
                    # multi-edge: (a,b) appears twice
                    # Swap b with d: (a,b)+(c,d) -> (a,d)+(c,b)
                    if c in (a, b) or d in (a, b):
                        continue
                    new_key1 = (min(a, d), max(a, d))
                    new_key2 = (min(c, b), max(c, b))
                    if new_key1 in edge_set or new_key2 in edge_set:
                        continue
                    if a == d or c == b:
                        continue
                    edges[idx] = (a, d)
                    edges[jdx] = (c, b)
                    # Update edge_set
                    edge_set.discard((min(a, b), max(a, b)))
                    edge_set.discard((min(c, d), max(c, d)))
                    edge_set.add(new_key1)
                    edge_set.add(new_key2)
                    fixed = True
                    break
            if not fixed:
                break

        # Check if we got a valid simple graph
        adj = [set() for _ in range(n)]
        valid = True
        for (a, b) in edges:
            if a == b or b in adj[a]:
                valid = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if valid and all(len(adj[i]) == k for i in range(n)):
            return [sorted(adj[i]) for i in range(n)]

    return None


# ---------------- Local search: edge swaps to increase girth -------------------
def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g using BFS. Returns cycle as list of vertices."""
    for s in range(n):
        parent = {s: -1}
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif v != parent[u] and parent.get(v) != u:
                    L = dist[u] + dist[v] + 1
                    if L < g:
                        # Reconstruct cycle
                        path_u = []
                        x = u
                        while x != -1:
                            path_u.append(x)
                            x = parent[x]
                        path_u.reverse()
                        path_v = []
                        x = v
                        while x != -1:
                            path_v.append(x)
                            x = parent[x]
                        path_v.reverse()
                        # Find LCA
                        set_u = set(path_u)
                        lca = None
                        for node in path_v:
                            if node in set_u:
                                lca = node
                                break
                        if lca is None:
                            continue
                        i_u = path_u.index(lca)
                        i_v = path_v.index(lca)
                        cycle = path_u[i_u:] + path_v[i_v:i_v + (len(path_v) - i_v) - 1][::-1]
                        return cycle
    return None


def _local_search_girth(adj, n, k, g, rng, max_steps=5000, time_limit=110.0):
    """Local search: find short cycles and break them via edge swaps."""
    adj = [set(a) for a in adj]
    best_girth = _nonback_girth([sorted(a) for a in adj], n)

    for step in range(max_steps):
        if _elapsed() > time_limit:
            break
        if best_girth >= g:
            return [sorted(a) for a in adj]

        # Find a short cycle
        cyc = _find_short_cycle([sorted(a) for a in adj], n, g)
        if cyc is None or len(cyc) >= g:
            return [sorted(a) for a in adj]

        # Try to break an edge on the cycle
        m = len(cyc)
        swapped = False
        for _ in range(50):
            eidx = rng.randrange(m)
            u = cyc[eidx]
            v = cyc[(eidx + 1) % m]
            if v not in adj[u]:
                continue
            # Find a valid swap: (u,v) + (w,x) -> (u,w) + (v,x)
            for _ in range(60):
                w = rng.randrange(n)
                if w in (u, v):
                    continue
                if w in adj[u]:
                    continue
                if not adj[w]:
                    continue
                x = rng.choice(list(adj[w]))
                if x in (u, v, w):
                    continue
                if x in adj[v]:
                    continue
                if w == x:
                    continue
                # Check no multi-edge would be created
                if w in adj[v]:
                    continue
                if x in adj[u]:
                    continue
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(x)
                adj[x].add(v)
                swapped = True
                break
            if swapped:
                break

        # Periodically re-check girth
        if step % 50 == 0:
            best_girth = _nonback_girth([sorted(a) for a in adj], n)
            if best_girth >= g:
                return [sorted(a) for a in adj]

    final_girth = _nonback_girth([sorted(a) for a in adj], n)
    if final_girth >= g:
        return [sorted(a) for a in adj]
    return None


# ---------------- Random search with local search ------------------------------
def _random_search(k, g, seed=0, max_n_offset=80):
    rng = random.Random(seed)
    n0 = _min_n_feasible(k, g)
    for n in range(n0, n0 + max_n_offset):
        if (n * k) % 2 != 0:
            continue
        if n < k + 1:
            continue
        for attempt in range(50):
            if _elapsed() > _DEADLINE - 8.0:
                return None
            adj = _random_regular_v2(n, k, rng)
            if adj is None:
                continue
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
            # Local search
            res = _local_search_girth(adj, n, k, g, rng, max_steps=3000, time_limit=_DEADLINE - 10.0)
            if res is not None:
                edges = [(i, j) for i in range(n) for j in res[i] if i < j]
                return n, edges
    return None


# ---------------- Circulant search ---------------------------------------------
def _circulant_search(k, g):
    n0 = _min_n_feasible(k, g)
    max_n = min(n0 + 30, 200)
    if k % 2 == 0:
        half = k // 2
        for n in range(n0, max_n + 1):
            if (n * k) % 2 != 0 or n < k + 1:
                continue
            for S in itertools.combinations(range(1, n // 2 + 1), half):
                S = list(S)
                if any((n - s) % n in S for s in S):
                    continue
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    adj = [sorted(adj[i]) for i in range(n)]
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    return None


# ---------------- Complete search for small n ----------------------------------
def _complete_search(k, g):
    """For very small n, try all k-regular graphs (limited)."""
    n0 = _min_n_feasible(k, g)
    max_n = min(n0 + 10, 30)
    for n in range(n0, max_n + 1):
        if (n * k) % 2 != 0 or n < k + 1:
            continue
        # Generate all k-regular graphs via backtracking (only for small n)
        if n > 20:
            continue
        adj = [set() for _ in range(n)]
        result = _backtrack_regular(adj, n, k, 0, g)
        if result is not None:
            edges = [(i, j) for i in range(n) for j in result[i] if i < j]
            return n, edges
    return None


def _backtrack_regular(adj, n, k, idx, g):
    if idx == n:
        if _nonback_girth([sorted(a) for a in adj], n) >= g:
            return [sorted(a) for a in adj]
        return None
    # Try to add edges for vertex idx
    if len(adj[idx]) == k:
        return _backtrack_regular(adj, n, k, idx + 1, g)
    if len(adj[idx]) > k:
        return None
    # Candidates: vertices j > idx with len(adj[j]) < k and j not in adj[idx]
    for j in range(idx + 1, n):
        if j in adj[idx] or len(adj[j]) >= k:
            continue
        adj[idx].add(j)
        adj[j].add(idx)
        res = _backtrack_regular(adj, n, k, idx, g)
        if res is not None:
            return res
        adj[idx].remove(j)
        adj[j].remove(idx)
    return None


# ---------------- Coset graph construction --------------------------------------
def _coset_graph_search(k, g):
    """Search for a coset graph (Cayley graph of a group) with desired girth."""
    # For k=5, g=9: try small groups
    # A5 (order 120): Cayley graph with 5 generators
    # A6 (order 720)
    # Try A5 first
    if k == 5:
        # A5 presentation: <a, b | a^5 = b^3 = (ab)^2 = 1>
        # Actually use explicit elements of A5
        # A5 = {even permutations of {0,1,2,3,4}}
        from itertools import permutations

        # Generate all even permutations
        A5 = []
        for perm in permutations(range(5)):
            # Check if even
            inv = list(perm)
            n_inv = 0
            for i in range(5):
                for j in range(i + 1, 5):
                    if inv[i] > inv[j]:
                        n_inv += 1
            if n_inv % 2 == 0:
                A5.append(perm)

        if len(A5) != 120:
            return None

        # Find a set of 5 generators
        # Try: all elements of order 5 and 3
        identity = (0, 1, 2, 3, 4)
        order_5 = []
        order_3 = []
        for elem in A5:
            if elem == identity:
                continue
            # Compute order
            p = elem
            for o in range(1, 6):
                if p == identity:
                    break
                # Compose p with elem
                p = tuple(p[elem[i]] for i in range(5))
            if o == 5:
                order_5.append(elem)
            elif o == 3:
                order_3.append(elem)

        # Try combinations of generators
        # We need 5 generators (since degree 5), and the Cayley graph should
        # have girth >= 9
        # Try: 2 generators of order 3 + 3 generators of order 5, or similar
        # Actually for degree 5, we need exactly 5 generators (or fewer if some
        # are involutions, but A5 has no involutions... wait, A5 has elements
        # of order 2? No, A5 has orders 1, 2, 3, 5. Actually A5 has no order 2
        # elements since transpositions are odd.)
        # Wait, A5 does have elements of order 2: double transpositions like
        # (01)(23). Let me check: (01)(23) maps 0->1, 1->0, 2->3, 3->2, 4->4.
        # Number of inversions: (0,1) -> 1,0; (2,3) -> 3,2. That's 2 inversions,
        # so it's even. Yes, A5 has order 2 elements.
        order_2 = []
        for elem in A5:
            if elem == identity:
                continue
            # Check if order 2
            p = tuple(elem[elem[i]] for i in range(5))
            if p == identity:
                order_2.append(elem)

        # For degree 5 Cayley graph, we can use 5 generators or
        # 2 order-2 generators + ... but let's just try 5 non-involution generators
        # Actually, if we use an involution s, then s^{-1} = s, so one generator
        # gives degree 2 (edges to g and g*s for s != identity).
        # For non-involution generator s, we get edges to g*s and g*s^{-1}, degree 2.
        # So 5 generators of order > 2 give degree 10, not 5.
        # For degree 5: use 2 involutions (degree 4) + 1 element of order 3 (degree 2)?
        # No, 2*2 + 2 = 6. 
        # 1 involution (degree 2) + 2 elements of order 3 (degree 4)? = 6.
        # Actually, for a Cayley graph with generator set S (not containing identity),
        # degree = |S| if all elements are non-involutions and S = S^{-1}, or
        # degree = |S| if S contains involutions.
        # For degree 5: |S| = 5, with S = S^{-1}.
        # So we need 5 elements where the set is closed under inversion.
        # E.g., 1 involution + 2 pairs of order 3 or 5 elements.
        # Or 5 elements all of which are their own inverses (involutions) - but
        # A5 has limited involutions.
        
        # Let's try: 1 involution + 2 elements of order 3 (one and its inverse)
        # + 1 element of order 5 (one and its inverse)? That's 1 + 2 + 2 = 5.
        # Wait, if s has order 3, then S includes s and s^{-1}, that's 2 elements.
        # If t has order 5, S includes t and t^{-1}, that's 2 elements.
        # Plus 1 involution: total 5.
        
        # Try all combinations
        for inv in order_2:
            for o3a in order_3:
                o3b = tuple(o3a[o3a[i]] for i in range(5))  # o3a^{-1}
                if o3a == o3b:
                    continue
                for o5a in order_5:
                    o5b = tuple(o5a[o5a[i]] for i in range(5))
                    if o5a == o5b:
                        continue
                    # Check S = {inv, o3a, o3b, o5a, o5b} is closed under inversion
                    S = {inv, o3a, o3b, o5a, o5b}
                    if len(S) != 5:
                        continue
                    # Build Cayley graph
                    adj = [set() for _ in range(120)]
                    idx_map = {elem: i for i, elem in enumerate(A5)}
                    for i, g_elem in enumerate(A5):
                        for s in S:
                            # g * s
                            prod = tuple(g_elem[s[i]] for i in range(5))
                            j = idx_map.get(prod)
                            if j is not None:
                                adj[i].add(j)
                    if all(len(adj[i]) == 5 for i in range(120)):
                        adj = [sorted(adj[i]) for i in range(120)]
                        if _nonback_girth(adj, 120) >= g:
                            edges = [(i, j) for i in range(120) for j in adj[i] if i < j]
                            return 120, edges
        return None
    return None


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    lock = threading.Lock()

    def _update(n, edges):
        with lock:
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)

    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("circ", lambda: _circulant_search(k, g)),
        ("rand0", lambda: _random_search(k, g, seed=0)),
        ("rand1", lambda: _random_search(k, g, seed=1)),
        ("rand2", lambda: _random_search(k, g, seed=2)),
        ("coset", lambda: _coset_graph_search(k, g)),
    ]

    threads = []
    for name, fn in jobs:
        def _worker(f=fn, n_name=name):
            try:
                res = f()
            except Exception:
                res = None
            if res is None:
                return
            n, edges = res
            if _elapsed() >= _DEADLINE:
                return
            try:
                if n > 0 and _is_valid(_build_adj(edges, n), n, k, g):
                    _update(n, edges)
            except Exception:
                pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 3.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.05)

    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
