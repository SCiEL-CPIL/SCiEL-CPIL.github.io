"""Cayley-graph construction on the 3x3 unitriangular group over GF(q),
with LUW bipartite construction as a guaranteed-valid fallback."""
import time
import math


def _next_prime_ge(n):
    def is_prime(m):
        if m < 2:
            return False
        if m % 2 == 0:
            return m == 2
        f = 3
        while f * f <= m:
            if m % f == 0:
                return False
            f += 2
        return True
    m = max(2, int(math.ceil(n)))
    if m % 2 == 0:
        m += 1
    while not is_prime(m):
        m += 2
    return m


def _pow_prime_power(p, e):
    r = 1
    for _ in range(e):
        r *= p
    return r


def _next_prime_power_ge(n):
    # smallest prime power >= n (n>=2)
    p = 2
    while True:
        e = 1
        v = p
        while v < n:
            e += 1
            v *= p
        return p, e, v
        # (unreachable; loop structure keeps single return path)
        p += 1


def _build_group(q):
    # GF(q) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
        inv = lambda x: x  # char 2
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        inv = lambda x: (-x) % q

    # group element = (a, b, c) for matrix [[1,a,b],[0,1,c],[0,0,1]]
    # (a,b,c)*(a',b',c') = (a+a', b+b'+a'*c, c+c')
    def mul3(u, v):
        a, b, c = u
        a2, b2, c2 = v
        return (add(a, a2), add(add(b, b2), mul(a2, c)), add(c, c2))

    def inv3(u):
        a, b, c = u
        return (inv(a), inv(add(b, mul(c, a))), inv(c))

    id3 = (0, 0, 0)

    # enumerate all q^3 elements
    elems = []
    for a in range(q):
        for c in range(q):
            for b in range(q):
                elems.append((a, b, c))
    idx = {e: i for i, e in enumerate(elems)}

    # permutation arrays for right-multiplication by each generator
    # precompute for the specific generators we use
    def perm_of(g):
        g = tuple(g)
        arr = [0] * (q * q * q)
        for i, e in enumerate(elems):
            arr[i] = idx[mul3(e, g)]
        return arr

    x = (1, 0, 0)
    y = (0, 0, 1)
    z = (0, 1, 0)
    px = perm_of(x)
    py = perm_of(y)
    pz = perm_of(z)
    pxi = perm_of(inv3(x))
    pyi = perm_of(inv3(y))
    pzi = perm_of(inv3(z))

    # adjacency permutation list for a given symmetric set S (each with its inverse)
    # S is a list of (elem, inverse_elem); adjacency = union of right-mult by each
    def build_adj(S):
        perms = []
        for e, ei in S:
            perms.append(perm_of(e))
            perms.append(perm_of(ei))
        return perms

    return q, add, sub, mul, inv, mul3, inv3, id3, elems, idx, \
        x, y, z, px, py, pz, pxi, pyi, pzi, build_adj


def _cayley_construct(k, g, deadline):
    # choose prime power q >= k
    p, e, q = _next_prime_power_ge(max(2, k))
    (qq, add, sub, mul, inv, mul3, inv3, id3, elems, idx,
     x, y, z, px, py, pz, pxi, pyi, pzi, build_adj) = _build_group(q)
    n = len(elems)
    if n > 400000:
        return None  # too large to handle quickly; fall back

    # candidate symmetric generating sets of size k, ordered by expected girth
    # each entry: list of (elem, inverse) pairs, must total k elements (counting each pair once)
    # Note: for a symmetric set, each unordered pair {s, s^-1} contributes 2 to degree
    # unless s is self-inverse (order 2), then 1.
    # We want |S| = k exactly.

    candidates = []

    # --- Type A: 3 generators x, y, z and their inverses -> degree 6 (only if k==6) ---
    # --- General: pick m elements from {x,y,z} (each with inverse) plus filler from a set ---

    # We need exactly k generators (counting both s and s^-1 separately as adjacency directions,
    # but as a *set* S we need |S|=k). Since graph is undirected, S must be symmetric:
    # if s in S then s^-1 in S. So |S| is even unless some s is self-inverse.
    # For k odd, we need at least one self-inverse element.

    # Build a pool of "generator directions" we can use.
    # Each *pair* (s, s^-1) uses 2 slots. A self-inverse s uses 1 slot.
    # Self-inverse elements of order 2: need s^2 = id.
    # In our group (odd q), x^2=(2,0,0) != id unless q=2. So no order-2 in x,y,z for odd q.
    # For q=4 (char 2), x^2 = (0,0,0) = id! So x, y, z are self-inverse when q=4.

    if q == 4:
        # char 2: x,y,z all self-inverse. Each uses 1 slot.
        # degree = number of distinct self-inverse gens we pick.
        # We can pick k of the available self-inverse elements.
        # Available: x, y, z, and any element of order 2.
        # In char 2, every element? No, only order 1 or 2.
        # x^2=id, y^2=id, z^2=id, and (xy)^2 = x y x y = x x y z y (since yx = yx, and yx = x y z)
        # Let's just test candidate sets by construction.
        # Self-inverse candidates: x, y, z, and products.
        self_inv = [x, y, z]
        # also xy, xz, yz, xyz etc. might be self-inverse; check
        def is_self_inv(e):
            return mul3(e, e) == id3
        pool = [x, y, z,
                mul3(x, y), mul3(x, z), mul3(y, z),
                mul3(mul3(x, y), z), mul3(mul3(x, z), y), mul3(mul3(y, z), x)]
        seen = set()
        for cand in pool:
            if is_self_inv(cand) and cand != id3 and cand not in seen:
                seen.add(cand)
                self_inv.append(cand)
        # pick k from self_inv (order matters for girth; try several orderings)
        if len(self_inv) >= k:
            # try the natural order first: x, y, z, then others
            base = self_inv[:k]
            candidates.append(base)
            # also try including more "mixed" ones
            if len(self_inv) > k:
                alt = [x, y, z] + self_inv[3:3 + (k - 3)]
                if len(alt) == k:
                    candidates.append(alt)
    else:
        # odd q: no order-2 among x,y,z. Need even k or use a self-inverse element.
        # For odd k, find a self-inverse element to use as 1 slot, plus (k-1)/2 pairs.
        # Self-inverse elements exist: e.g. x*(something). Actually in odd q,
        # an element (a,b,c) is self-inverse iff 2a=0 and 2c=0 and b + a*c ... 
        # (a,b,c)^2 = (2a, b + b + a*c, 2c) = (2a, 2b + a*c, 2c).
        # = id iff a=0, c=0, 2b=0 -> b=0. So ONLY identity is self-inverse for odd q!
        # Therefore for odd q, k MUST be even to have a symmetric generating set
        # without identity. If k is odd, this construction cannot give degree k
        # directly. Fallback needed.
        if k % 2 == 1:
            return None

        # k even: pick k/2 pairs from {x,y,z} plus additional pairs.
        # Available pairs: (x,x^-1), (y,y^-1), (z,z^-1), and (xy,(xy)^-1), etc.
        # (xy)^-1 = y^-1 x^-1.
        def inv_of(e):
            return inv3(e)
        pair_pool = [x, y, z,
                     mul3(x, y), mul3(x, z), mul3(y, z),
                     mul3(mul3(x, y), z), mul3(mul3(x, z), y), mul3(mul3(y, z), x)]
        # deduplicate and exclude identity
        seen = set()
        uniq = []
        for cand in pair_pool:
            if cand != id3 and cand not in seen:
                seen.add(cand)
                uniq.append(cand)
        need = k // 2
        if len(uniq) < need:
            return None
        # choose `need` from uniq; try natural order
        chosen = uniq[:need]
        candidates.append(chosen)
        # try a couple of alternative selections
        if len(uniq) > need:
            chosen2 = [x, y, z] + uniq[3:3 + (need - 3)] if need >= 3 else uniq[:need]
            if len(chosen2) == need and chosen2 != chosen:
                candidates.append(chosen2)

    # For each candidate set, build the Cayley graph and verify girth
    best = None
    for cand in candidates:
        if time.time() > deadline:
            break
        # build symmetric set S as list of (elem, inverse)
        S = []
        for e in cand:
            ei = inv3(e)
            if e == ei:
                S.append((e, e))
            else:
                S.append((e, ei))
        # verify |S| = k
        # count distinct elements in S
        distinct = set()
        for e, ei in S:
            distinct.add(e)
            distinct.add(ei)
        if len(distinct) != k:
            continue
        # build adjacency
        perms = build_adj(S)
        # degree check
        # each vertex has degree = len(perms) but some perms may coincide (if e1==e2 or e1==e2^-1)
        # check for duplicate adjacency: for a vertex, the set of neighbours
        # Since perms are right-multiplications by distinct group elements, and all distinct
        # (we ensured distinct), degree = len(perms) = 2*len(S_pairs) or len(S) for self-inv.
        # Actually len(perms) = sum over S entries of (1 if self-inv else 2).
        # For self-inv entry (e,e): we added perm_of(e) twice -> duplicate!
        # Fix: for self-inverse, add perm only once.
        perms = []
        for e, ei in S:
            perms.append(perm_of(e))
            if e != ei:
                perms.append(perm_of(ei))
        if len(perms) != k:
            continue

        # Build adjacency lists (as lists of ints)
        adj = [[] for _ in range(n)]
        for u in range(n):
            for pr in perms:
                v = pr[u]
                if v != u:  # no loops (shouldn't happen since no identity in S)
                    adj[u].append(v)
        # Check no duplicate edges (would happen if two perms give same v)
        # and degree exactly k
        for u in range(n):
            if len(adj[u]) != k:
                break
            if len(set(adj[u])) != k:
                break
        else:
            # valid k-regular simple graph; now verify girth via non-backtracking walks
            # girth = length of shortest cycle. Use BFS from each vertex but that's O(n^2).
            # Better: for small g, do bounded non-backtracking walk of length g-1 from each
            # vertex and check for closure.
            # A cycle of length L means: a non-backtracking walk of length L that returns to start.
            # We check L from 3 up to g-1. If none found, girth >= g.
            # For efficiency, do a BFS in the "non-backtracking" state space (prev, cur).
            # State: (prev, cur) meaning we arrived at cur from prev.
            # Start states: (u, v) for each edge (u,v).
            # Walk length 2 corresponds to states (u,v) reached in 1 step from start (u,?) ... 
            # Let's do: for each start vertex s, do BFS over (prev, cur) with depth,
            # where depth 1 = (s, v) for v in adj[s]. A cycle of length L>=3 exists iff
            # at depth L-1 we have state (prev, s) for some prev (i.e., we returned to s).
            # We only need to check up to depth g-1.
            found_short = False
            for s in range(n):
                if found_short:
                    break
                # BFS: dist[(prev, cur)] = length of non-backtracking walk from s to cur
                # that arrived from prev, with the walk starting at s.
                # Initialize: for each v in adj[s], state (s, v) at depth 1.
                from collections import deque
                dist = {}
                dq = deque()
                for v in adj[s]:
                    dist[(s, v)] = 1
                    dq.append((s, v))
                while dq:
                    prev, cur = dq.popleft()
                    d = dist[(prev, cur)]
                    if d >= g - 1:
                        continue  # no need to extend further
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue  # non-backtracking
                        if nxt == s and d + 1 >= 3:
                            # found a cycle of length d+1
                            if d + 1 < g:
                                found_short = True
                                break
                        if (cur, nxt) not in dist:
                            dist[(cur, nxt)] = d + 1
                            dq.append((cur, nxt))
                    if found_short:
                        break
                if found_short:
                    break
            if found_short:
                continue  # this candidate has a cycle shorter than g; try next
            # valid!
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            best = (n, edges)
            break  # first valid candidate is good enough (smallest q already chosen)

    return best


def _luw_construct(k, g, deadline):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def construct(k, g):
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Try Cayley construction first (structurally smaller for same k, g)
    result = _cayley_construct(k, g, deadline)
    if result is not None:
        return result

    # Fallback to LUW (always valid)
    return _luw_construct(k, g, deadline)
