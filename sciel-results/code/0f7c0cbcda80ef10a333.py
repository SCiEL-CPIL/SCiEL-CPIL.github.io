"""Search for a small 5-regular graph with girth >= g.

Strategy:
  1. Fast randomized search: generate random k-regular graphs via stub matching,
     verify girth exactly with a fast non-backtracking BFS, keep the smallest valid one.
  2. 2-switch local search: perturb a candidate graph to reduce short cycles while
     preserving k-regularity; re-check girth after each accepted move.
  3. Use a deadline guard to return the best valid result before the wall-clock limit.
  4. Fallback: Lazebnik-Ustimenko-Woldar construction (known small cages for some params).

Key optimizations:
  * Girth check uses dict-based BFS on non-backtracking states for speed on sparse graphs.
  * Vectorized stub-matching permutation for speed.
  * 2-switch moves reuse the stub array and only touch 4 edges.
  * Adaptive trial budget: more trials for smaller n where success is more likely.
  * Early exit in girth check: as soon as any short cycle is found, return False.
  * Skip n values that are below the Moore bound or where n*k is odd.
"""
import time
import numpy as np

# ---------------- deterministic RNG ----------------
def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = np.repeat(np.arange(n, dtype=np.int64), k)
    perm = np.arange(m, dtype=np.int64)
    # Fisher-Yates shuffle using numpy for speed
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
        seen[u].add(v)
        seen[v].add(u)
    return adj

def _swap_regular(adj, n, k, rng, stubs=None):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges).
    If stubs is provided (a list of length n*k), it is reused."""
    m = n * k
    if stubs is None:
        stubs = [i for i in range(n) for _ in range(k)]
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    # remove edges (u1,v1) and (u2,v2); require they are distinct
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    # add edges (u1,u2) and (v1,v2); forbid self-loops / parallel edges
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

# ---------------- exact girth via non-backtracking walks ----------------
def _girth_ok(adj, g):
    """Return True iff the graph has no cycle of length < g.
    Uses BFS on non-backtracking states (u, prev) from each source s, with an
    early cutoff at depth g-1. Uses numpy for speed."""
    n = len(adj)
    max_depth = g - 1
    # Convert adj to numpy arrays for faster iteration
    adj_arrays = [np.array(adj[i], dtype=np.int32) for i in range(n)]
    for s in range(n):
        # visited: 2D boolean array [n][n] for state (current, prev)
        visited = np.zeros((n, n), dtype=np.bool_)
        # Queue as pre-allocated arrays
        q_u = np.zeros(20000, dtype=np.int32)
        q_p = np.zeros(20000, dtype=np.int32)
        q_d = np.zeros(20000, dtype=np.int32)
        head = 0
        tail = 0
        for v in adj_arrays[s]:
            visited[v, s] = True
            q_u[tail] = v
            q_p[tail] = s
            q_d[tail] = 1
            tail += 1
        while head < tail:
            u = q_u[head]
            p = q_p[head]
            d = q_d[head]
            head += 1
            if d >= max_depth:
                continue
            for w in adj_arrays[u]:
                if w == p:
                    continue
                if w == s:
                    return False
                if not visited[w, u]:
                    visited[w, u] = True
                    if tail >= len(q_u):
                        q_u = np.concatenate([q_u, np.zeros(20000, dtype=np.int32)])
                        q_p = np.concatenate([q_p, np.zeros(20000, dtype=np.int32)])
                        q_d = np.concatenate([q_d, np.zeros(20000, dtype=np.int32)])
                    q_u[tail] = w
                    q_p[tail] = u
                    q_d[tail] = d + 1
                    tail += 1
    return True

def _shortest_cycle_len(adj, g):
    """Return the length of the shortest cycle, or a value >= g if none shorter
    exists.  Uses BFS on non-backtracking states from each vertex."""
    n = len(adj)
    adj_arrays = [np.array(adj[i], dtype=np.int32) for i in range(n)]
    for s in range(n):
        visited = np.zeros((n, n), dtype=np.bool_)
        q_u = np.zeros(20000, dtype=np.int32)
        q_p = np.zeros(20000, dtype=np.int32)
        q_d = np.zeros(20000, dtype=np.int32)
        head = 0
        tail = 0
        for v in adj_arrays[s]:
            visited[v, s] = True
            q_u[tail] = v
            q_p[tail] = s
            q_d[tail] = 1
            tail += 1
        while head < tail:
            u = q_u[head]
            p = q_p[head]
            d = q_d[head]
            head += 1
            if d >= g - 1:
                continue
            for w in adj_arrays[u]:
                if w == p:
                    continue
                if w == s:
                    return d + 1
                if not visited[w, u]:
                    visited[w, u] = True
                    if tail >= len(q_u):
                        q_u = np.concatenate([q_u, np.zeros(20000, dtype=np.int32)])
                        q_p = np.concatenate([q_p, np.zeros(20000, dtype=np.int32)])
                        q_d = np.concatenate([q_d, np.zeros(20000, dtype=np.int32)])
                    q_u[tail] = w
                    q_p[tail] = u
                    q_d[tail] = d + 1
                    tail += 1
    return g

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 115.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    # Moore bound (lower bound on n)
    if g % 2 == 1:  # odd girth
        # 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        moore = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    else:  # even girth
        # 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        moore = 2 * sum((k-1)**i for i in range(g//2))
    n_start = max(moore, 2*k)

    # Search increasing n; budget trials by n so small n gets more effort.
    # Skip n values where n*k is odd (impossible for k-regular simple graph when k is odd and n is odd)
    n = n_start
    while n <= 1500 and time.time() < deadline:
        # Skip impossible n: need n*k even for a simple k-regular graph
        if (n * k) % 2 != 0:
            n += 1
            continue

        # Time budget per n value
        remaining = deadline - time.time()
        if remaining < 2.0:
            break

        # Allocate time: spend proportional effort on early n values
        if n < n_start + 30:
            time_per_n = min(remaining * 0.20, 20.0)
        elif n < n_start + 100:
            time_per_n = min(remaining * 0.10, 10.0)
        elif n < n_start + 300:
            time_per_n = min(remaining * 0.05, 5.0)
        else:
            time_per_n = min(remaining * 0.03, 3.0)

        n_deadline = time.time() + time_per_n

        # Precompute stubs for this n to reuse in swaps
        stubs = [i for i in range(n) for _ in range(k)]

        # Adaptive number of trials based on time budget
        max_trials = max(5, int(time_per_n / 0.4))
        max_switches = max(10, int(time_per_n / 0.2))

        for t in range(max_trials):
            if time.time() > n_deadline or time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            
            # Check girth
            if _girth_ok(adj, g):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            
            # 2-switch local search: try to improve girth
            best_sc = _shortest_cycle_len(adj, g)
            best_adj = [list(x) for x in adj]
            best_best_sc = best_sc
            no_improve_count = 0
            max_no_improve = 30
            
            switch_count = 0
            while switch_count < max_switches and no_improve_count < max_no_improve and time.time() < n_deadline and time.time() < deadline:
                if _swap_regular(adj, n, k, rng, stubs):
                    sc = _shortest_cycle_len(adj, g)
                    if sc >= g:
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
                    if sc > best_best_sc:
                        best_best_sc = sc
                        best_adj = [list(x) for x in adj]
                        no_improve_count = 0
                    elif sc == best_best_sc:
                        # Plateau move: accept to escape local minimum
                        no_improve_count += 1
                    else:
                        # Reject: revert
                        adj = [list(x) for x in best_adj]
                        no_improve_count += 1
                switch_count += 1
            
            # Continue from the best state we found
            adj = [list(x) for x in best_adj]
            best_sc = best_best_sc

        n += 1

    # Fallback: the Lazebnik-Ustimenko-Woldar construction
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
