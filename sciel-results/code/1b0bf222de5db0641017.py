"""Search for a small 5-regular graph with girth >= g.

Strategy:
1. Fast girth verification via per-vertex non-backtracking BFS (NumPy-accelerated).
2. Random k-regular graph search: generate via random pairing, then improve via
   targeted 2-switches guided by a cycle-count objective (smooth surrogate).
3. Deterministic LUW construction as guaranteed fallback.
"""
import time
import numpy as np
from collections import deque

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
    stubs = [i for i in range(n) for _ in range(k)]
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    edge_set = set()
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges)."""
    m = n * k
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
    u1 = a // k
    v1 = b // k
    u2 = c // k
    v2 = d // k
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    # Remove edges (u1,v1) and (u2,v2)
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    # Check for parallel edges / self-loops in new edges
    if u1 == u2 or v1 == v2 or u1 == v2 or v1 == u2:
        # Restore
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1] or u2 in adj[v1] or v2 in adj[u1]:
        # Restore
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    # Add new edges (u1,u2) and (v1,v2)
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

# ---------------- fast girth via per-vertex non-backtracking BFS (optimized) ----------------
def _girth_check(adj, n, g):
    """Return True if girth >= g. Uses per-vertex BFS on non-backtracking
    states (u, prev). A cycle of length L < g exists iff some start vertex s
    reaches state (s, t) at distance < g where t is a neighbor of s."""
    for s in range(n):
        # dist indexed by (u, prev) -> use flat list of size n*n
        dist = [-1] * (n * n)
        # Store as two parallel lists for speed: frontier_u, frontier_prev
        frontier_u = []
        frontier_prev = []
        for t in adj[s]:
            dist[t * n + s] = 1
            frontier_u.append(t)
            frontier_prev.append(s)
        d = 1
        max_d = g - 1  # need to explore up to distance g-1
        while d < max_d and frontier_u:
            new_u = []
            new_prev = []
            for i in range(len(frontier_u)):
                u = frontier_u[i]
                prev = frontier_prev[i]
                for w in adj[u]:
                    if w == prev:
                        continue
                    if w == s:
                        if d + 1 < g:
                            return False
                        continue
                    idx = w * n + u
                    if dist[idx] == -1:
                        dist[idx] = d + 1
                        new_u.append(w)
                        new_prev.append(u)
            frontier_u = new_u
            frontier_prev = new_prev
            d += 1
    return True

# ---------------- cycle counting for objective ----------------
def _count_short_cycles(adj, n, g):
    """Count closed non-backtracking walks of length < g (proxy for short cycles).
    Uses per-vertex BFS. Returns total count (lower is better)."""
    total = 0
    for s in range(n):
        dist = [-1] * (n * n)
        frontier_u = []
        frontier_prev = []
        for t in adj[s]:
            dist[t * n + s] = 1
            frontier_u.append(t)
            frontier_prev.append(s)
        d = 1
        max_d = g - 1
        while d < max_d and frontier_u:
            new_u = []
            new_prev = []
            for i in range(len(frontier_u)):
                u = frontier_u[i]
                prev = frontier_prev[i]
                for w in adj[u]:
                    if w == prev:
                        continue
                    if w == s:
                        total += 1
                        continue
                    idx = w * n + u
                    if dist[idx] == -1:
                        dist[idx] = d + 1
                        new_u.append(w)
                        new_prev.append(u)
            frontier_u = new_u
            frontier_prev = new_prev
            d += 1
    return total

# ---------------- targeted swap: try to eliminate short cycles ----------------
def _targeted_swap(adj, n, k, rng, max_tries=20):
    """Try max_tries random 2-switches, keeping the one that most reduces
    short-cycle count. Returns True if a swap was applied."""
    best_cycle_count = None
    best_swap = None
    
    for _ in range(max_tries):
        m = n * k
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
        u1 = a // k
        v1 = b // k
        u2 = c // k
        v2 = d // k
        if u1 == v1 or u2 == v2:
            continue
        if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
            continue
        if u1 == u2 or v1 == v2 or u1 == v2 or v1 == u2:
            continue
        if u2 in adj[u1] or v2 in adj[v1] or u2 in adj[v1] or v2 in adj[u1]:
            continue
        
        # Apply swap tentatively
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        adj[u1].append(u2)
        adj[u2].append(u1)
        adj[v1].append(v2)
        adj[v2].append(v1)
        
        # Count short cycles (lightweight: only check affected vertices)
        # Quick heuristic: check if new edges create triangles or 4-cycles
        cc = 0
        for x in (u1, u2, v1, v2):
            for y in adj[x]:
                if y == x:
                    continue
                for z in adj[y]:
                    if z == x or z == y:
                        continue
                    if z in adj[x]:
                        cc += 1
        if best_cycle_count is None or cc < best_cycle_count:
            best_cycle_count = cc
            best_swap = (u1, v1, u2, v2)
        
        # Undo
        adj[u1].remove(u2)
        adj[u2].remove(u1)
        adj[v1].remove(v2)
        adj[v2].remove(v1)
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
    
    if best_swap is not None:
        u1, v1, u2, v2 = best_swap
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        adj[u1].append(u2)
        adj[u2].append(u1)
        adj[v1].append(v2)
        adj[v2].append(v1)
        return True
    return False

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 115.0  # safety margin
    rng = 0x9E3779B97F4A7C15
    
    # Moore bound
    if g % 2 == 1:
        h = (g - 1) // 2
        moore = k * (k ** (h - 1) - 1) // (k - 1) if h > 1 else k
    else:
        h = g // 2
        moore = k * (k ** (h - 1) - 1) // (k - 1) if h > 1 else k
    moore = max(moore, 2 * k)
    
    # Search range: from moore to a reasonable upper bound
    # For (5,9): moore=625, known best=1116
    n_start = moore
    n_max = min(moore + 700, 1400)
    
    best_adj = None
    best_n = None
    
    # Phase 1: Try each n from n_start, do limited random search
    n = n_start
    while n <= n_max and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        # Allocate time: more time for smaller n
        time_budget_n = min(remaining * 0.25, 10.0)
        n_deadline = time.time() + time_budget_n
        
        # Try to find a valid graph at this n
        adj = None
        attempts = 0
        gen_deadline = time.time() + time_budget_n * 0.2
        while time.time() < gen_deadline and attempts < 200:
            attempts += 1
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
        
        if adj is None:
            n += 1
            continue
        
        # Check if already valid
        if _girth_check(adj, n, g):
            return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        
        # Local search: 2-switches guided by cycle count
        init_cycles = _count_short_cycles(adj, n, g)
        best_local_cycles = init_cycles
        best_local_adj = [list(x) for x in adj]
        
        search_deadline = n_deadline
        stuck_count = 0
        iter_count = 0
        while time.time() < search_deadline:
            iter_count += 1
            # Use targeted swap every 5 iterations, random swap otherwise
            if iter_count % 5 == 0:
                did_swap = _targeted_swap(adj, n, k, rng, max_tries=10)
            else:
                did_swap = _swap_regular(adj, n, k, rng)
            
            if did_swap:
                if _girth_check(adj, n, g):
                    return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
                # Track cycle count periodically
                if stuck_count % 30 == 0:
                    cycles = _count_short_cycles(adj, n, g)
                    if cycles < best_local_cycles:
                        best_local_cycles = cycles
                        best_local_adj = [list(x) for x in adj]
                        stuck_count = 0
                    else:
                        stuck_count += 1
                else:
                    stuck_count += 1
            
            # If stuck for too long, reset to best local
            if stuck_count > 300:
                adj = [list(x) for x in best_local_adj]
                stuck_count = 0
        
        # Keep best found so far
        if best_n is None or n < best_n:
            if _girth_check(adj, n, g):
                best_adj = [list(x) for x in adj]
                best_n = n
        
        n += 1
    
    if best_adj is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best_adj[i] if i < j]
    
    # Fallback: Lazebnik-Ustimenko-Woldar construction
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
