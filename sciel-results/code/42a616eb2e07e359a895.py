"""Rewritten program: Simulated Annealing for 4-regular girth-10 cage.
Approach: Start with random 4-regular graph, minimize short cycles via 2-switches with SA."""
import random
import math
import sys
from collections import deque

def _count_short_cycles(adj, g, max_len=9):
    """Count cycles of length 3 to max_len (or g-1)."""
    n = len(adj)
    count = 0
    adj_set = {v: set(adj[v]) for v in range(n)}
    
    # Count triangles (3)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for w in adj[u]:
                    if v < w and w in adj_set[v]:
                        count += 1
    # Count 4-cycles
    # A 4-cycle is u-v-w-x-u. 
    # We can count paths of length 2 between pairs.
    # For each pair (u, v), let c(u,v) be number of common neighbors.
    # Each 4-cycle contributes 2 to the sum of c(u,v) over all edges? No.
    # Number of 4-cycles = sum_{u<v} C(common_neighbors(u,v), 2) - (triangles involving u,v)*?
    # Actually, simpler: 
    # Iterate all pairs (u, v). 
    # Let k_uv be common neighbors.
    # Number of 4-cycles containing edge (u,v) is k_uv - 1 (if u,v adjacent) ? No.
    # If u,v are NOT adjacent, each pair of common neighbors forms a 4-cycle u-w1-v-w2-u.
    # So sum over all non-adjacent pairs (u,v) of C(k_uv, 2).
    # If u,v ARE adjacent, the common neighbors w form triangles u-v-w. 
    # The 4-cycles involving edge (u,v) are u-v-w-x-u where w is neighbor of v, x is neighbor of u, w!=x, w!=u, x!=v.
    # This is getting complex. 
    # For SA, we can just use a BFS-based check for "any short cycle" and use that as a binary signal, 
    # or count the *length* of the shortest cycle.
    
    # Better Energy: Length of the shortest cycle.
    # We want to maximize the shortest cycle length.
    # Energy = -shortest_cycle_length.
    # Or Energy = 1 if short cycle exists, else 0? 
    # Gradient is zero if we just use binary.
    
    # Let's use the number of non-backtracking walks of length L that close.
    # Too slow.
    
    # Let's just use the existing _has_short_cycle logic but return the length found.
    # If no short cycle, return g.
    return _shortest_cycle_length(adj, g)

def _shortest_cycle_length(adj, g):
    """Return the length of the shortest cycle, or g if none found < g."""
    n = len(adj)
    min_len = g
    # BFS from each vertex for depth g//2
    for s in range(n):
        # State: (prev, cur, dist)
        # We store dist from s to cur via prev.
        # Actually, standard BFS on the non-backtracking graph.
        # dist[(prev, cur)] = distance from s.
        # If we reach (cur, s) with dist d, we have a cycle of length d+1?
        # Start: s -> nb. dist=1.
        # If nb == s? No, simple graph.
        # If we reach s again? 
        # The path is s -> ... -> cur -> s.
        # Length is dist + 1.
        
        from collections import deque
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((s, nb))
            dist[(s, nb)] = 1
        
        while dq:
            prev, cur = dq.popleft()
            d = dist[(prev, cur)]
            if d >= min_len - 1: # If d+1 >= min_len, we don't care
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == s:
                    if d + 1 < min_len:
                        min_len = d + 1
                else:
                    nstate = (cur, nb)
                    if nstate not in dist:
                        dist[nstate] = d + 1
                        dq.append(nstate)
    return min_len

def _random_4_regular(n, rng):
    """Build a random simple 4-regular graph on n vertices."""
    while True:
        stubs = []
        for v in range(n):
            stubs.extend([v] * 4)
        rng.shuffle(stubs)
        adj = {v: [] for v in range(n)}
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                ok = False
                break
            if b in adj[a]:
                ok = False
                break
            adj[a].append(b)
            adj[b].append(a)
        if ok:
            return adj

def _sa_improve(adj, g, rng, steps=50000):
    """Simulated Annealing to increase girth."""
    n = len(adj)
    adj_set = {v: set(adj[v]) for v in range(n)}
    
    current_len = _shortest_cycle_length(adj, g)
    best_len = current_len
    best_adj = {v: list(adj[v]) for v in range(n)}
    
    T = 1.0
    cooling = 0.999
    
    for step in range(steps):
        if T < 0.01:
            break
            
        # Pick a random 2-switch
        # Pick two random edges (a,b) and (c,d)
        # To be efficient, pick a random vertex a, neighbor b.
        # Pick random vertex c, neighbor d.
        # Ensure they are distinct and not forming loops/multi-edges.
        
        a = rng.randrange(n)
        b = adj_set[a][rng.randrange(len(adj_set[a]))]
        c = rng.randrange(n)
        d = adj_set[c][rng.randrange(len(adj_set[c]))]
        
        # Check validity of switch
        # Edges (a,b) and (c,d) must exist.
        # New edges (a,c) and (b,d) must not exist.
        # a,b,c,d must be distinct.
        if len({a, b, c, d}) < 4:
            continue
        if c in adj_set[a] or d in adj_set[b]:
            continue
        if a in adj_set[c] or b in adj_set[d]: # Symmetric check
            continue
            
        # Apply switch
        adj_set[a].discard(b); adj_set[b].discard(a)
        adj_set[c].discard(d); adj_set[d].discard(c)
        adj_set[a].add(c); adj_set[c].add(a)
        adj_set[b].add(d); adj_set[d].add(b)
        
        new_len = _shortest_cycle_length(adj_set, g)
        
        if new_len > current_len:
            current_len = new_len
            if current_len > best_len:
                best_len = current_len
                best_adj = {v: list(adj_set[v]) for v in range(n)}
        else:
            # Accept with probability exp(-(new_len - current_len) / T) ?
            # We want to MAXIMIZE length.
            # Delta = current_len - new_len (positive if worse)
            delta = current_len - new_len
            if delta > 0:
                prob = math.exp(-delta / T)
                if rng.random() > prob:
                    # Reject, revert
                    adj_set[a].discard(c); adj_set[c].discard(a)
                    adj_set[b].discard(d); adj_set[d].discard(b)
                    adj_set[a].add(b); adj_set[b].add(a)
                    adj_set[c].add(d); adj_set[d].add(c)
            # If delta == 0, accept (or random walk)
            # If delta < 0, it's an improvement, handled above.
            
        T *= cooling
        
    return best_adj

def _verify(adj, k, g):
    n = len(adj)
    if n < k + 1:
        return False
    for v in range(n):
        if len(adj[v]) != k:
            return False
        if len(set(adj[v])) != k:
            return False
    for v in range(n):
        for u in adj[v]:
            if u == v:
                return False
            if v not in adj[u]:
                return False
    if _shortest_cycle_length(adj, g) < g:
        return False
    return True

def construct(k, g):
    if k != 4:
        return _fallback(k, g)

    rng = random.Random(54321)
    # Try to find a graph with n close to Moore bound
    # Moore bound 341. Known cage 320.
    # We try 320, 324, etc.
    candidates = [320, 324, 328, 332, 336, 340, 344, 348, 352, 360]
    
    for n in candidates:
        # Run SA for a few seconds?
        # 120s limit.
        # We can run SA for 320.
        adj = _random_4_regular(n, rng)
        if adj is None:
            continue
        adj = _sa_improve(adj, g, rng, steps=20000)
        if _verify(adj, k, g):
            edges = []
            for v in range(n):
                for u in adj[v]:
                    if v < u:
                        edges.append((v, u))
            return n, edges
            
    return _fallback(k, g)

def _fallback(k, g):
    # Same fallback as before
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
