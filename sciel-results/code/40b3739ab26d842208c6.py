"""Search for a small 5-regular graph with girth >= g.

Unlike the seed (which builds a full 1116-vertex cage), we only need a k-regular
graph whose shortest cycle has length >= g; it need not be a cage.  We therefore
generate random k-regular graphs and verify their girth exactly with
non-backtracking walks, keeping the smallest verified one.  A deadline guard
returns the best valid result before the wall-clock limit.
"""
import random
import time
import numpy as np
from array import array

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
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    # random perfect matching
    perm = list(range(m))
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
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges)."""
    # Pick two distinct edges (u,v) and (x,y)
    # To do this efficiently, we can pick a random edge from the list of all edges.
    # But we don't have a global edge list. We can pick a random vertex u, then a random neighbor v.
    # Then pick a random vertex x, then a random neighbor y.
    # Ensure (u,v) != (x,y) and they are not the same edge reversed.
    
    u = _randrange(rng, n)
    v = adj[u][_randrange(rng, k)]
    
    x = _randrange(rng, n)
    y = adj[x][_randrange(rng, k)]
    
    # Check if edges are distinct
    if (u == x and v == y) or (u == y and v == x):
        return False
    
    # Attempt switch: remove (u,v), (x,y); add (u,x), (v,y)
    # Check for self-loops and parallel edges
    if u == x or v == y:
        return False
    if x in adj[u] or y in adj[v]:
        return False
    
    # Perform switch
    adj[u].remove(v)
    adj[v].remove(u)
    adj[x].remove(y)
    adj[y].remove(x)
    
    adj[u].append(x)
    adj[x].append(u)
    adj[v].append(y)
    adj[y].append(v)
    
    return True

# ---------------- exact girth via non-backtracking walks ----------------
def _girth(adj, g):
    """Return the length of the shortest cycle, or a value >= g if none shorter
    exists.  Uses non-backtracking walks from every vertex."""
    n = len(adj)
    if n == 0:
        return g
    
    # Build adjacency matrix for fast lookup
    # Since k is small, we can use a set or a boolean matrix.
    # For speed in BFS, a boolean matrix or set of neighbors is good.
    # Let's use a list of sets for O(1) neighbor check.
    adj_sets = [set(neighs) for neighs in adj]
    
    # BFS on directed edges (u, v)
    # State: (u, v) means we are at v, came from u.
    # dist[(u, v)] = length of walk.
    # Start: all (u, v) for u in adj[v] have dist 1.
    # Actually, a walk of length 1 is just an edge.
    # A cycle of length L corresponds to a path of length L-1 from v to u where (u,v) is an edge?
    # No. A cycle v0-v1-...-v_{L-1}-v0.
    # Directed edges: (v0,v1), (v1,v2), ..., (v_{L-1},v0).
    # If we start at (v0, v1) with dist 1, we can extend to (v1, v2) with dist 2, etc.
    # If we reach (v_{L-1}, v0) with dist L, and v0 is the start vertex...
    # Wait, the state (u, v) doesn't track the start vertex.
    # Standard BFS for girth:
    # dist[v][u] = shortest non-backtracking walk from v to u?
    # Let's use the standard algorithm:
    # Initialize dist[v][u] = INF for all u,v.
    # For each edge (v, u), set dist[v][u] = 1.
    # BFS queue: all edges (v, u).
    # Pop (prev, curr).
    # For each neighbor next of curr where next != prev:
    #    if dist[curr][next] is INF:
    #        dist[curr][next] = dist[prev][curr] + 1
    #        push (curr, next)
    #    else if next == prev: # This is the backtracking case, already excluded by next != prev
    #    else: # We found a cycle?
    #        # If we try to go to next, and dist[curr][next] is already set,
    #        # it means there is a shorter walk from curr to next.
    #        # The cycle length would be dist[curr][next] + 1?
    #        # Let's trace:
    #        # Walk: v0 -> v1 -> ... -> curr -> next.
    #        # If dist[curr][next] is set, it means there is a walk from curr to next of length L'.
    #        # The walk we are constructing has length dist[prev][curr] + 1.
    #        # This doesn't directly give the cycle length unless we track the start.
    
    # Correct approach:
    # dist[v][u] = length of shortest non-backtracking walk from v to u.
    # Initialize: for each edge (v, u), dist[v][u] = 1.
    # Queue: all (v, u) with dist 1.
    # When processing (prev, curr):
    #   For each next in adj[curr] where next != prev:
    #     if dist[curr][next] == INF:
    #        dist[curr][next] = dist[prev][curr] + 1
    #        push (curr, next)
    #     else:
    #        # We found a cycle.
    #        # The walk from curr to next has length dist[curr][next].
    #        # The walk we are extending is prev -> curr -> next.
    #        # This forms a cycle?
    #        # No, dist[curr][next] is the shortest NB walk from curr to next.
    #        # If we add the edge curr->next to the walk ending at curr...
    #        # Actually, if dist[curr][next] is already set, it means we have seen this state.
    #        # The cycle length is dist[prev][curr] + 1 + (distance from next to prev?)
        # This is getting complicated.
        
    # Simpler: Just check for cycles of length 3, 4, ..., g-1 explicitly?
    # For g=9, we need to check 3..8.
    # This is feasible for n=400-1000.
    
    # Let's use the BFS on states (start, current, prev) is too large.
    # Let's use the property: girth is the smallest L such that there is a walk of length L-1 from v to u where (u,v) is an edge.
    # So we can run BFS from each vertex v? That's O(n * (n*k)) which is too slow.
    
    # Let's stick to the directed edge BFS but track the "start" vertex implicitly?
    # No, the standard algorithm for girth is:
    # 1. Create a graph of directed edges.
    # 2. Run BFS from all directed edges simultaneously?
    #    dist[e] = length of shortest path ending at edge e.
    #    Start: all edges have dist 1.
    #    Transition: (u->v) -> (v->w) if w != u.
    #    If we reach (v->w) where (w->v) is also in the graph... no.
    #    If we reach (v->w) and dist[v->w] is already set, we have a cycle?
    #    No, that just means we found a shorter path to that edge.
    
    # The correct check:
    # A cycle of length L exists iff there is a directed edge (u, v) such that
    # there is a non-backtracking walk of length L-1 from v to u.
    # So we can compute the shortest non-backtracking distance between all pairs?
    # That's O(n^2) states. n=500 -> 250k states.
    # Transitions: each state (v, u) goes to (u, w) for w in adj[u], w != v.
    # Total transitions: n * k * (k-1) ~ 500 * 5 * 4 = 10k.
    # BFS on this state graph is very fast.
    
    # State: (v, u) means "at u, came from v".
    # dist[v, u] = shortest NB walk length ending at state (v, u).
    # Initialize: for each edge (v, u), dist[v, u] = 1.
    # Queue: all (v, u).
    # Pop (v, u).
    # For each w in adj[u] where w != v:
    #    if dist[u, w] == INF:
    #       dist[u, w] = dist[v, u] + 1
    #       push (u, w)
    #    else:
    #       # We have a cycle?
    #       # The walk to (u, w) has length dist[u, w].
    #       # The walk we are considering is (v, u) -> (u, w).
    #       # This doesn't close a cycle yet.
    
    # Wait, if we initialize dist[v, u] = 1 for ALL edges, then when we process (v, u) and go to (u, w),
    # if dist[u, w] is already 1 (meaning (u, w) is an edge, which it is), we don't push it.
    # This BFS finds the shortest NB walk between any two directed edges?
    # No, it finds the shortest NB walk *starting from any edge* to reach state (u, w).
    
    # To find a cycle:
    # We need a walk from v to u (length L-1) such that (u, v) is an edge.
    # So we need dist[v, u] to be defined, and (u, v) is an edge.
    # But dist[v, u] is the shortest NB walk from *any* start to (v, u).
    # This doesn't guarantee the walk started at v.
    
    # Correct Algorithm:
    # For each vertex s:
    #   Run BFS from s on the non-backtracking graph.
    #   dist[s][u][v] = shortest NB walk from s to u, last step from v?
    #   This is O(n * n * k) which is too slow for n=1000.
    
    # Alternative:
    # Check for cycles of length 3, 4, 5, 6, 7, 8 directly.
    # Given n=500, k=5.
    # Cycles of length 3: check triangles. O(n * k^2) = 500 * 25 = 12500.
    # Cycles of length 4: O(n * k^3) = 500 * 125 = 62500.
    # Cycles of length 5: O(n * k^4) = 500 * 625 = 312500.
    # Cycles of length 6: O(n * k^5) = 500 * 3125 = 1.5M.
    # Cycles of length 7: O(n * k^6) = 500 * 15625 = 7.8M.
    # Cycles of length 8: O(n * k^7) = 500 * 78125 = 39M.
    # Total ~ 50M operations. In Python, this might take 1-5 seconds.
    # We can optimize by early exiting if we find a cycle.
    
    # Let's implement a function to check if girth < g.
    # It returns the length of the shortest cycle found, or g if none < g.
    
    adj_sets = [set(neighs) for neighs in adj]
    
    # Check 3-cycles
    for v in range(n):
        for u in adj[v]:
            if u > v:
                for w in adj[u]:
                    if w > u and w in adj_sets[v]:
                        return 3
    # Check 4-cycles
    # v-u-w-x-v
    # Fix v, u. w is neighbor of u, w != v. x is neighbor of w, x != u.
    # Check if x == v.
    for v in range(n):
        for u in adj[v]:
            for w in adj[u]:
                if w == v: continue
                for x in adj[w]:
                    if x == u: continue
                    if x == v:
                        return 4
                    # Optimization: break if x < v? No, need to check all.
    
    # This is getting messy. Let's use the BFS on directed edges but correctly.
    # The state is (prev, curr).
    # We want to know if there is a path from s to t of length L-1.
    # If we run BFS from *all* vertices simultaneously, we lose the "start" info.
    
    # However, we can use the following trick:
    # A graph has girth >= g iff the non-backtracking matrix B has no eigenvalues... no.
    
    # Let's try the direct cycle enumeration for small g.
    # Since g=9, we check 3..8.
    
    # 3-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            if u > v:
                if vs & adj_sets[u]:
                    return 3
    
    # 4-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            if u > v:
                # Common neighbors of v and u? No, that's triangles.
                # For 4-cycle v-u-w-x-v, u and x are neighbors of v, w is neighbor of u and x.
                # Fix v.
                # For each pair of neighbors u, x of v (u < x):
                #   Check if u and x have a common neighbor w.
                #   If yes, cycle v-u-w-x-v.
                neighs_v = adj[v]
                for i in range(len(neighs_v)):
                    u = neighs_v[i]
                    for j in range(i+1, len(neighs_v)):
                        x = neighs_v[j]
                        # Check common neighbor of u and x
                        # Must be w != v
                        if adj_sets[u] & (adj_sets[x] - {v}):
                            return 4
    
    # 5-cycles
    # v-u-w-x-y-v
    # Fix v.
    # u in adj[v]
    # w in adj[u], w != v
    # x in adj[w], x != u
    # y in adj[x], y != w
    # Check if y in adj[v]
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            for w in adj[u]:
                if w == v: continue
                for x in adj[w]:
                    if x == u: continue
                    for y in adj[x]:
                        if y == w: continue
                        if y in vs and y != u: # y != u is implied if no triangles? No.
                            # Check if v-u-w-x-y-v is a simple cycle
                            # v, u, w, x, y must be distinct.
                            # v != u, v != w, v != x, v != y
                            # u != w, u != x, u != y
                            # w != x, w != y
                            # x != y
                            # We know w!=v, x!=u, y!=w.
                            # Need to check:
                            # x != v? If x==v, then v-u-w-v is a 3-cycle, already checked.
                            # y != u? If y==u, then v-u-w-u is a 3-cycle? No, v-u-w-u is not a cycle in simple graph unless w=u which is false.
                            # Actually if y==u, the walk is v-u-w-u... back to u. Not a simple cycle to v.
                            # So if y in vs, and the vertices are distinct, it's a 5-cycle.
                            if len({v, u, w, x, y}) == 5:
                                return 5
    
    # 6-cycles
    # v-u-w-x-y-z-v
    # This is O(n * k^5).
    # We can optimize by fixing v and u.
    # Then we need a path of length 4 from u to v, not visiting v or u in the middle.
    # Path u-w-x-y-z-v.
    # w in adj[u], w != v.
    # x in adj[w], x != u.
    # y in adj[x], y != w.
    # z in adj[y], z != x.
    # Check if z in adj[v].
    
    # This is getting too slow in Python if we do it naively.
    # But n is ~500.
    # 5-cycles: 500 * 5 * 4 * 4 * 4 = 320,000 iterations. Fast.
    # 6-cycles: 500 * 5 * 4 * 4 * 4 * 4 = 1,280,000 iterations. Fast.
    # 7-cycles: 500 * 5 * 4^5 = 5,120,000. Fast.
    # 8-cycles: 500 * 5 * 4^6 = 20,480,000. Maybe 1-2 seconds.
    
    # Let's implement 6, 7, 8 similarly.
    
    # 6-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            for w in adj[u]:
                if w == v: continue
                for x in adj[w]:
                    if x == u or x == v: continue
                    for y in adj[x]:
                        if y == w or y == v: continue
                        for z in adj[y]:
                            if z == x: continue
                            if z in vs and z != u:
                                if len({v, u, w, x, y, z}) == 6:
                                    return 6
    
    # 7-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            for w in adj[u]:
                if w == v: continue
                for x in adj[w]:
                    if x == u or x == v: continue
                    for y in adj[x]:
                        if y == w or y == v: continue
                        for z in adj[y]:
                            if z == x or z == v: continue
                            for t in adj[z]:
                                if t == y: continue
                                if t in vs and t != u:
                                    if len({v, u, w, x, y, z, t}) == 7:
                                        return 7
                                    
    # 8-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            for w in adj[u]:
                if w == v: continue
                for x in adj[w]:
                    if x == u or x == v: continue
                    for y in adj[x]:
                        if y == w or y == v: continue
                        for z in adj[y]:
                            if z == x or z == v: continue
                            for t in adj[z]:
                                if t == y or t == v: continue
                                for s in adj[t]:
                                    if s == z: continue
                                    if s in vs and s != u:
                                        if len({v, u, w, x, y, z, t, s}) == 8:
                                            return 8
                                            
    return g

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 110.0
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth(adj, g) >= g:
            best = adj
            best_n = n
            return True
        return False

    # Search n starting from Moore bound
    # Moore bound for k=5, g=9 is 426.
    # Let's start at 430 and go up.
    
    # If k is generic, calculate Moore bound?
    # r = (g-1)//2
    # mb = 1 + k * sum((k-1)^i for i in range(r))
    
    r = (g - 1) // 2
    mb = 1
    term = 1
    for i in range(r):
        term *= (k - 1)
        mb += k * term
    # Ensure n*k is even
    if (k * mb) % 2 != 0:
        mb += 1
        
    # Start search
    n = mb
    while n <= 1000 and time.time() < deadline:
        # Try a few random graphs
        found = False
        for _ in range(20):
            if time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        
        # If not found randomly, try local search on one instance
        adj = _pairing_regular(n, k, rng)
        if adj is not None:
            # Try to increase girth
            for _ in range(5000):
                if time.time() > deadline:
                    break
                if _swap_regular(adj, n, k, rng):
                    if consider(n, adj):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        
        n += 2

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # Fallback: Lazebnik
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
