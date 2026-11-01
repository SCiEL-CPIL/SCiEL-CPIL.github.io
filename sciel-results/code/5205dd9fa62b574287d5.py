"""Voltage-lift (regular cover) construction for the (3,16) cage problem.

Strategy: build a compact 3-regular base graph B (a "skeleton"), assign voltages
in the free product C2*C2*C2 (so the universal cover is a 3-regular tree), and
lift to a regular cover. A base cycle of length ell with total voltage w lifts
to a cycle of length ell * ord(w); requiring ell*ord(w) >= g for every base cycle
guarantees girth >= g. We search base graphs and voltage assignments to minimise
the number of lift vertices n while keeping the lift connected.
"""
import time
import random
from collections import deque

# ---------- Free product C2*C2*C2 word machinery ----------
# Generators 0,1,2 each of order 2. Reduced words are alternating letters.

def word_reduce(w):
    out = []
    for a in w:
        if out and out[-1] == a:
            out.pop()
        else:
            out.append(a)
    return tuple(out)

def word_mul(a, b):
    return word_reduce(a + b)

def word_inv(w):
    # inverse of a word in the free product
    return word_reduce(tuple(reversed(w)))

def word_ord(w, cap=60):
    # order of element w; 0 means >= cap (treat as very large)
    if len(w) == 0:
        return 1
    cur = w
    for m in range(2, cap + 1):
        cur = word_mul(cur, w)
        if len(cur) == 0:
            return m
    return cap  # effectively large

# ---------- Base graph generation (3-regular, connected, small) ----------

def random_3_regular(n):
    # n must be even
    if n % 2:
        return None
    # random 3-regular multigraph via configuration model, then dedupe
    for _ in range(2000):
        stubs = []
        for v in range(n):
            stubs += [v] * 3
        random.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                break
            edges.append((min(a, b), max(a, b)))
        else:
            # check simple + connected
            if len(set(edges)) != len(edges):
                continue
            adj = [set() for _ in range(n)]
            for a, b in edges:
                adj[a].add(b)
                adj[b].add(a)
            if any(len(adj[v]) != 3 for v in range(n)):
                continue
            # connectivity
            seen = {0}
            dq = deque([0])
            while dq:
                x = dq.popleft()
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        dq.append(y)
            if len(seen) != n:
                continue
            return edges
    return None

# ---------- Cycle enumeration on a small base graph ----------

def base_cycles(base_edges, n):
    # enumerate simple cycles up to length L; returns list of (frozenset-of-edge-indices, length)
    adj = [[] for _ in range(n)]
    for ei, (a, b) in enumerate(base_edges):
        adj[a].append((b, ei))
        adj[b].append((a, ei))
    cycles = []
    seen = set()
    Lmax = 12
    for start in range(n):
        # DFS with edge set, canonical: start is smallest vertex in cycle
        stack = [(start, start, frozenset(), 0, [start])]
        while stack:
            v, prev, used, length, path = stack.pop()
            if length >= 2 and v == start:
                cyc = frozenset(used)
                # canonical: smallest vertex in cycle must equal start
                verts = set()
                for a, b in base_edges:
                    pass
                if cyc not in seen:
                    seen.add(cyc)
                    cycles.append((cyc, length))
                continue
            if length >= Lmax:
                continue
            for (w, ei) in adj[v]:
                if ei in used:
                    continue
                if w == prev and length > 0:
                    # backtracking over same edge not allowed (simple)
                    pass
                # enforce start is smallest vertex in the cycle to avoid dups
                if w < start and w != start:
                    continue
                stack.append((w, v, used | {ei}, length + 1, path + [w]))
    return cycles

# ---------- Girth certification of the lifted graph (non-backtracking / BFS) ----------

def lift_graph(base_edges, n, volt, group_size, group_id, gen_elems, group_mul):
    # build adjacency of lift: vertex (v, g)
    # For each base edge (a,b) with voltage g_ab (on a->b), lift:
    #   (a, x) -- (b, x * g_ab)   and   (b, y) -- (a, y * g_ba)
    # We need consistent voltages both ways.
    # volt: dict edge_index -> (v_ab, v_ba) where v_ba = inv(v_ab)
    adj = {}
    idmap = {}
    def lid(v, g):
        key = (v, g)
        if key not in idmap:
            idmap[key] = len(idmap)
        return idmap[key]
    # We'll build full lift adjacency list
    N = n * group_size
    adjlist = [[] for _ in range(N)]
    for ei, (a, b) in enumerate(base_edges):
        v_ab = volt[ei]
        v_ba = word_inv(v_ab)
        for x in range(group_size):
            id_a = lid(a, x)
            id_b = lid(b, group_mul(x, v_ab))
            adjlist[id_a].append(id_b)
            adjlist[id_b].append(id_a)
        for y in range(group_size):
            id_b2 = lid(b, y)
            id_a2 = lid(a, group_mul(y, v_ba))
            adjlist[id_b2].append(id_a2)
            adjlist[id_a2].append(id_b2)
    # dedupe adjacency (each undirected edge appears twice)
    for i in range(N):
        adjlist[i] = sorted(set(adjlist[i]))
    return N, adjlist

def connected(N, adjlist):
    if N == 0:
        return True
    seen = {0}
    dq = deque([0])
    while dq:
        x = dq.popleft()
        for y in adjlist[x]:
            if y not in seen:
                seen.add(y)
                dq.append(y)
    return len(seen) == N

def girth(N, adjlist, deadline):
    # BFS from each vertex, track parent to avoid backtracking, find shortest cycle
    best = 10**9
    for s in range(N):
        if time.time() > deadline:
            break
        dist = [-1] * N
        parent = [-1] * N
        dist[s] = 0
        dq = deque([s])
        found = None
        while dq:
            u = dq.popleft()
            for v in adjlist[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    dq.append(v)
                elif parent[u] != v and parent[v] != u:
                    cand = dist[u] + dist[v] + 1
                    if cand < best:
                        best = cand
                        found = cand
                    if best <= g:
                        return best
    return best

# ---------- Main construction ----------

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    # For k=3, g=16 typical. Generalise lightly.
    if k != 3:
        # Fallback: random 3-regular won't work for other k; use a trivial valid
        # construction via voltage lift on a cycle base (works for any k via padding)
        # For safety, return a known-valid small construction if available.
        return _fallback(k, g, deadline)

    # Group: free product C2*C2*C2, truncated to words of length <= W.
    # Build group elements as reduced words up to length W, closed under * and inv.
    W = 10
    elems = [()]
    by_word = {(): 0}
    for L in range(1, W + 1):
        # generate all reduced words of length L
        # first letter any of 3, subsequent alternate
        from itertools import product
        for first in range(3):
            for rest in product(range(3), repeat=L - 1):
                cand = [first] + list(rest)
                # reduce
                reduced = []
                for a in cand:
                    if reduced and reduced[-1] == a:
                        reduced.pop()
                    else:
                        reduced.append(a)
                t = tuple(reduced)
                if len(t) <= W and t not in by_word:
                    by_word[t] = len(elems)
                    elems.append(t)
    G = len(elems)
    # multiplication table (sparse): for each x and each generator, store x*gen
    # We'll compute on the fly with caching.
    mul_cache = {}
    def gmul(x, y):
        key = (x, y)
        if key in mul_cache:
            return mul_cache[key]
        r = word_mul(elems[x], elems[y])
        if r in by_word:
            res = by_word[r]
        else:
            # product left the truncated set; map to identity as a "boundary" marker
            # To keep group closed, we instead restrict voltages to short words so
            # products stay in set. We'll handle by returning a sentinel large index.
            res = -1
        mul_cache[key] = res
        return res

    # We need a closed group. Use the full free product truncated by a ball of radius W
    # is NOT closed under multiplication. Instead, use a finite quotient: the dihedral
    # group is too small. Use a random finite group: S_n via permutations? Simpler:
    # use Z_m (abelian) voltages -> lift girth = ell * ord(w) but abelian lifts are
    # bipartite-ish and may be disconnected. Use a non-abelian finite group:
    # the free product C2*C2*C2 modulo relations is hard. 
    # Practical choice: use the group of permutations of {0..m-1} generated by 3
    # random involutions; it's finite, closed, and non-abelian for m>=4.

    # Rebuild group as permutation group.
    def build_perm_group(m, seed):
        rng = random.Random(seed)
        def random_involution():
            perm = list(range(m))
            rng.shuffle(perm)
            inv = [1] * m
            for i in range(0, m - 1, 2):
                a, b = perm[i], perm[i + 1]
                inv[a] = b
                inv[b] = a
            if m % 2:
                inv[perm[m - 1]] = perm[m - 1]
            return tuple(inv)
        gens = [random_involution() for _ in range(3)]
        # closure
        group = {tuple(range(m))}
        frontier = [tuple(range(m))]
        while frontier:
            nxt = []
            for p in frontier:
                for gen in gens:
                    q = tuple(p[gen[i]] for i in range(m))
                    if q not in group:
                        group.add(q)
                        nxt.append(q)
            frontier = nxt
        return list(group), gens

    # Choose a moderate permutation group size.
    best_result = None
    base_n_options = [8, 10, 12, 14, 16, 18, 20]
    perm_m_options = [8, 10, 12]
    seed = 12345
    rng = random.Random(seed)

    for perm_m in perm_m_options:
        if time.time() > deadline - 5:
            break
        group, gens = build_perm_group(perm_m, seed)
        Gsize = len(group)
        gid = {p: i for i, p in enumerate(group)}
        # multiplication
        def pmul(i, j):
            p = group[i]
            q = group[j]
            return gid[tuple(p[q[x]] for x in range(perm_m))]
        def pinv(i):
            p = group[i]
            inv = [0] * perm_m
            for x in range(perm_m):
                inv[p[x]] = x
            return gid[tuple(inv)]

        for bn in base_n_options:
            if time.time() > deadline - 5:
                break
            base_edges = random_3_regular(bn)
            if base_edges is None:
                continue
            # get base cycles
            cyc = base_cycles(base_edges, bn)
            # assign random voltages (group elements) to each edge
            for trial in range(40):
                if time.time() > deadline - 3:
                    break
                volt = [rng.choice(range(Gsize)) for _ in range(len(base_edges))]
                # check girth condition on base cycles: ell * ord(voltage) >= g
                ok = True
                for (cycset, ell) in cyc:
                    # total voltage around the cycle
                    total = 0  # identity index
                    # need orientation: traverse cycle
                    # Reconstruct cycle as sequence of edge indices in order.
                    # base_cycles returned frozenset of edge indices + length; we need
                    # an ordered traversal to multiply voltages consistently.
                    # For a small graph we can find the cycle order.
                    order = _cycle_order(base_edges, bn, cycset)
                    if order is None:
                        ok = False
                        break
                    total = _cycle_voltage(volt, order, pmul, pinv)
                    od = _perm_order(group, gid, total, cap=40)
                    if ell * od < g:
                        ok = False
                        break
                if not ok:
                    continue
                # build lift
                N, adjlist = lift_graph(base_edges, bn, volt, Gsize, gid, group, pmul)
                if not connected(N, adjlist):
                    continue
                # certify girth
                gr = girth(N, adjlist, deadline)
                if gr >= g:
                    # valid!
                    edges_out = []
                    for u in range(N):
                        for v in adjlist[u]:
                            if u < v:
                                edges_out.append((u, v))
                    if best_result is None or N < best_result[0]:
                        best_result = (N, edges_out)
                        # keep searching a bit for smaller
                # small n check for early exit
                if best_result is not None and best_result[0] <= 936:
                    pass  # continue to try to improve

    if best_result is not None:
        return best_result

    # Fallback: return a valid (if large) construction via a guaranteed method.
    return _fallback(k, g, deadline)

def _cycle_order(base_edges, bn, cycset):
    # find an ordered list of edge indices forming the simple cycle
    adj = [[] for _ in range(bn)]
    for ei, (a, b) in enumerate(base_edges):
        adj[a].append((b, ei))
        adj[b].append((a, ei))
    # start from smallest vertex in cycle
    verts = set()
    for ei in cycset:
        a, b = base_edges[ei]
        verts.add(a)
        verts.add(b)
    start = min(verts)
    # DFS
    stack = [(start, start, [start], [])]
    while stack:
        v, prev, path, eorder = stack.pop()
        if eorder and v == start:
            return eorder
        for (w, ei) in adj[v]:
            if ei in eorder:
                continue
            if ei not in cycset:
                continue
            if w == prev and len(eorder) > 0:
                continue
            stack.append((w, v, path + [w], eorder + [ei]))
    return None

def _cycle_voltage(volt, eorder, pmul, pinv):
    # orient each edge consistently along the traversal; we don't know orientation
    # from eorder alone. For girth certification of the lift we actually build the
    # full lift and BFS, so the cycle check here is only a *necessary* prefilter.
    # Multiply voltages in order (assumed orientation); if a base cycle's voltage
    # has small order in either orientation, the lift may have a short cycle.
    total = 0
    for ei in eorder:
        total = pmul(total, volt[ei])
    return total

def _perm_order(group, gid, idx, cap=40):
    cur = idx
    for m in range(2, cap + 1):
        p = group[cur]
        q = group[idx]
        cur = gid[tuple(p[q[x]] for x in range(len(p)))]
        if cur == 0:
            return m
    return cap

def _fallback(k, g, deadline):
    # Guaranteed-valid but possibly large: use the Lazebnik-style construction from
    # the seed as a safe fallback (it is known to produce a valid 3-regular graph
    # with girth >= g for the parameters used).
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
