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


# ---------------- exact girth via non-backtracking walks (vectorized) ----------
def _girth_np(adj_lists, n, g):
    """Return the exact girth (shortest cycle length) of a simple graph, or
    g if no cycle shorter than g exists. Uses non-backtracking BFS from all
    vertices simultaneously, vectorized with NumPy."""
    deg = np.array([len(a) for a in adj_lists], dtype=np.int32)
    m = int(deg.sum())
    pos = np.zeros(m, dtype=np.int32)
    for u in range(n):
        pos[u * (k := deg[u]): (u + 1) * k] = np.arange(k)
    # neighbor lookup table: nbr[u][p] = adj[u][p]
    nbr = np.zeros((n, k), dtype=np.int32)
    for u in range(n):
        nbr[u, :] = np.array(adj_lists[u], dtype=np.int32)

    # state (v, p)  ->  index v*k + p
    # prev state: the (u, q) we came from, where u = nbr[v][p]
    # We want to detect when a non-backtracking walk returns to its origin.

    # BFS layers. Layer 1 = all directed edges (v, p) with v = nbr[u][p].
    # Represent a state by its index. We store for each state the origin vertex
    # (u) and the distance (number of edges so far).

    # For efficiency, do layered BFS with NumPy.
    # current: array of state indices
    # origin: array of origin vertex for each state
    # dist: scalar (all same within a layer)

    # Initialize layer 1: for each directed edge (u -> v) where v = nbr[u][p]
    # state index = v*k + p, origin = u, dist = 1
    u_idx = np.repeat(np.arange(n, dtype=np.int32), deg)
    p_idx = pos
    v_idx = nbr[u_idx, p_idx]
    init_state = v_idx * k + p_idx
    init_origin = u_idx

    # prev_state for state (v, p) is (u, q) where u = nbr[v][p] and q = position
    # of v in adj[u]. We need pos[u][v].
    # Build pos lookup: pos_lookup[u][v] = position of v in adj[u], or -1.
    pos_lookup = np.full((n, n), -1, dtype=np.int32)
    for u in range(n):
        for p, v in enumerate(adj_lists[u]):
            pos_lookup[u, v] = p

    # visited: for each state, store the origin vertex that first reached it,
    # and the distance. We'll use arrays.
    visited_origin = np.full(n * k, -1, dtype=np.int32)
    visited_dist = np.zeros(n * k, dtype=np.int32)

    # Mark initial layer
    visited_origin[init_state] = init_origin
    visited_dist[init_state] = 1

    cur_state = init_state
    cur_origin = init_origin
    dist = 1

    while len(cur_state) > 0:
        # Extend: from state (v, p), origin u, go to (w, r) where w != v's parent.
        # w = nbr[v][r], r in 0..k-1, r != p (non-backtracking)
        # new state index = w*k + r
        # new origin = u (same origin)
        # Check: if w == u -> cycle of length dist+1.
        # If state (w, r) already visited with a DIFFERENT origin -> cycle.
        #   cycle length = dist + 1 + visited_dist[(w,r)] ... actually need to
        #   be careful. Standard: if we reach (w, r) from (v, p) and (w, r) was
        #   first reached from some state, the cycle length = dist + 1 +
        #   visited_dist[(w, r)] only if same origin... Let me use the clean
        #   formulation: a non-backtracking walk from origin u: at each step
        #   the state is (current, prev). A cycle forms when current == origin
        #   (and prev != the neighbor we started from, but non-backtracking
        #   handles that). So we only need to check w == origin.

        v_cur = cur_state // k
        p_cur = cur_state % k

        # all next neighbors
        r_all = np.tile(np.arange(k, dtype=np.int32), len(v_cur))
        w_all = nbr[v_cur, r_all]
        # mask out r == p_cur (backtrack)
        mask = (r_all != p_cur)
        w_all = w_all[mask]
        r_all = r_all[mask]
        o_all = cur_origin[mask]

        # check w == origin
        hit = (w_all == o_all)
        if hit.any():
            return dist + 1

        new_state = w_all * k + r_all
        new_origin = o_all

        # check visited
        old_origin = visited_origin[new_state]
        is_new = (old_origin == -1)

        if not is_new.all():
            # some states were visited. If visited with a different origin,
            # that means two non-backtracking walks from different origins met
            # at the same state -> forms a cycle. Cycle length:
            #   dist + 1 + visited_dist[new_state] (the two walks join).
            # Actually the cycle is: origin_a -> ... -> (w,r) <- ... <- origin_b.
            # length = dist + 1 (from origin_a to w) + visited_dist (from w back
            # to origin_b) ... but the two paths meet at (w, r). The cycle
            # length = dist + 1 + visited_dist[new_state].
            diff = is_new == False
            cand = (dist + 1 + visited_dist[new_state[diff]])
            if cand.min() < g:
                return int(cand.min())
            # For same-origin revisit, that's just the same walk, ignore.
            # Actually if old_origin == new_origin it's a cycle too but we'd have
            # caught w==origin. So diff-origin is the only new cycle info.

        # proceed with only new states
        new_state = new_state[is_new]
        new_origin = new_origin[is_new]

        if len(new_state) == 0:
            break

        visited_origin[new_state] = new_origin
        visited_dist[new_state] = dist + 1
        cur_state = new_state
        cur_origin = new_origin
        dist += 1

        if dist >= g:
            break

    return g


def _nonback_girth(adj, n):
    """Exact girth via non-backtracking walks from every vertex."""
    if n == 0:
        return 10 ** 9
    try:
        return _girth_np(adj, n, 10 ** 9)
    except Exception:
        # fallback to pure-python
        prev = {}
        q = deque()
        for u in range(n):
            for p in adj[u]:
                key = (p, u)
                prev[key] = (u, 1)
                q.append(key)
        while q:
            (v, p), (u, dist) = q.popleft()
            for w in adj[v]:
                if w == p:
                    continue
                key = (w, v)
                if w == u:
                    return dist + 1
                if key in prev:
                    continue
                prev[key] = (v, dist + 1)
                q.append(key)
        return 10 ** 9


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


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


# ---------------- Moore bound feasibility --------------------------------------
def _moore(k, g):
    R = (g - 1) // 2
    return 1 + k * sum((k - 1) ** i for i in range(R))


def _min_n_feasible(k, g):
    """Smallest n for which a k-regular girth-g graph could exist (parity +
    Moore bound)."""
    moore = _moore(k, g)
    n = moore
    while (n * k) % 2 != 0 or n < k + 1:
        n += 1
    return n


# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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


# ---------------- construction 2: Circulant Search (Enhanced) ------------------
def _check_circulant_girth(n, steps, g):
    """Check if circulant graph on Z_n with given steps has girth >= g.
    steps: list of positive integers. For odd k, one step is n//2.
    Returns True if girth >= g, False otherwise.
    """
    # Generate all possible sums of length L < g
    # For efficiency, we only need to check up to g-1
    # But checking all subsets is expensive.
    # We use a BFS on the group Z_n.
    # State: (current_pos, prev_step_index)
    # But for circulant, we can just check if 0 is reachable in < g steps
    # without backtracking? No, girth is shortest cycle.
    # A cycle of length L exists if there are s1, ..., sL in Steps (with signs)
    # such that sum = 0 mod n.
    
    # We can use a set of reachable positions at each distance.
    # dist 0: {0}
    # dist 1: {s, -s for s in steps}
    # dist 2: {a+b, a-b, -a+b, -a-b ...}
    
    # To be fast, we limit the search depth to g-1.
    # If 0 is found at depth L < g (and L>0), girth < g.
    
    # Note: 0 is always reachable at dist 0. We ignore dist 0.
    # We must avoid backtracking in the walk to find *simple* cycles?
    # No, girth is the length of the shortest *simple* cycle.
    # In a Cayley graph, a closed walk of length L might not be a simple cycle.
    # However, if there is a closed walk of length L, there is a simple cycle of length <= L.
    # So if we find a closed walk of length < g, girth < g.
    # Conversely, if girth >= g, there is no closed walk of length < g (except 0).
    # Wait, a closed walk can repeat vertices.
    # But if a closed walk of length L exists, it contains a simple cycle of length <= L.
    # So checking for closed walks is sufficient to detect girth < g.
    
    # Optimization: Use a set of visited states (pos, prev_step) to avoid
    # infinite loops, but since we stop at depth g-1, it's bounded.
    
    # Steps include n//2 if present.
    # Let's define the generator set G.
    G = set()
    for s in steps:
        G.add(s % n)
        G.add((-s) % n)
    
    # Remove 0 from G (should not be there for simple graph)
    G.discard(0)
    
    # BFS from 0
    # State: (pos, prev_gen)
    # prev_gen is the generator used to arrive at pos.
    # We start by taking one step from 0.
    
    visited = set()
    # dist 1
    current_states = []
    for g_gen in G:
        pos = g_gen
        # State is (pos, g_gen)
        # Check if pos == 0? No, g_gen != 0.
        visited.add((pos, g_gen))
        current_states.append((pos, g_gen))
    
    dist = 1
    while dist < g - 1 and current_states:
        next_states = []
        for pos, prev_gen in current_states:
            for g_gen in G:
                if g_gen == (-prev_gen) % n:
                    continue # Backtracking
                new_pos = (pos + g_gen) % n
                # Check if we return to 0
                if new_pos == 0:
                    return False # Found cycle of length dist+1 < g
                # Check if state visited
                state = (new_pos, g_gen)
                if state not in visited:
                    visited.add(state)
                    next_states.append(state)
        current_states = next_states
        dist += 1
        
    return True


def _circulant_search(k, g, max_n=None):
    n0 = _min_n_feasible(k, g)
    if max_n is None:
        max_n = min(n0 + 600, 2000)
    
    if k % 2 == 0:
        half = k // 2
        for n in range(n0, max_n + 1):
            if (n * k) % 2 != 0:
                continue
            if n < k + 1:
                continue
            # Try random step sets
            # We need 'half' distinct steps from 1..n//2
            # such that no step is n//2 (if n even) and steps are distinct.
            # Actually, if n is even, we can use n//2 as one of the steps.
            # But for simplicity, let's just pick 'half' steps from 1..n//2.
            # If n is even and we don't use n//2, the graph is bipartite?
            # No, circulant graphs with even n and steps not including n/2
            # might still have odd cycles.
            
            # Random search for steps
            for _ in range(100):
                if _elapsed() > _DEADLINE - 5.0:
                    return None
                # Pick 'half' random distinct steps from 1 to n//2
                # Ensure steps are not 0
                available = list(range(1, n // 2 + 1))
                if len(available) < half:
                    continue
                steps = random.sample(available, half)
                # Sort for consistency
                steps.sort()
                
                if _check_circulant_girth(n, steps, g):
                    # Construct edges
                    edges = []
                    for i in range(n):
                        for s in steps:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                            # If s == n//2, j < i is possible?
                            # If s == n//2, i+s = i+n/2.
                            # If i < n/2, i < i+n/2.
                            # If i >= n/2, i > i+n/2 - n? No.
                            # We just add (i, (i+s)%n) if i < (i+s)%n.
                    return n, edges
    else:
        m = (k - 1) // 2
        for n in range(n0, max_n + 1, 2): # n must be even for k=5?
            # n*k must be even. k=5 is odd, so n must be even.
            if n < k + 1:
                continue
            for _ in range(100):
                if _elapsed() > _DEADLINE - 5.0:
                    return None
                # Pick m steps from 1..n//2-1
                # And include n//2
                available = list(range(1, n // 2))
                if len(available) < m:
                    continue
                steps = random.sample(available, m)
                steps.append(n // 2)
                steps.sort()
                
                if _check_circulant_girth(n, steps, g):
                    edges = []
                    for i in range(n):
                        for s in steps:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges
    return None


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}

    jobs = [
        ("circ", lambda: _circulant_search(k, g)),
        ("luw", lambda: _luw(k, g)),
    ]

    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, k, g), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 5.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.03)

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


def _worker(name, fn, holder, k, g):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE - 2.0:
        return
    try:
        adj = _build_adj(edges, n)
        if n > 0 and _is_valid(adj, n, k, g):
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)
    except Exception:
        pass
