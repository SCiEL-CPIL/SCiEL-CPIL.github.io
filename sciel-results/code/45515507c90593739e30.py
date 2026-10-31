"""
Construct a k-regular graph with girth at least g.
Structural redesign: algebraic circulant search with polynomial cycle filters + LUW fallback.
"""
import time
from collections import deque
from math import gcd

# EVOLVE-BLOCK-START


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Algebraic circulant search C_n(1,a,b)
        result = _find_circulant_algebraic(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: LUW fallback
        for d in [11, 9, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None:
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _find_circulant_algebraic(k, g, deadline):
    """
    Search for C_n(1,a,b) with girth >= g using algebraic cycle filters.
    For a 3-regular circulant on Z_n, the generators are {1, a, b} with their negatives.
    A cycle of length L corresponds to a sequence of L steps from {±1, ±a, ±b} summing to 0 mod n,
    with no immediate backtracking (consecutive steps not being negatives of each other).
    """
    min_n = 512  # Moore bound for (3,16) is 2 + 3*(1+2+4+8+16+32) = 2 + 3*63 = 191... 
    # Actually Moore bound for (3,g) with g even: 2*(2^(g/2-1) - 1) = 2*(2^7-1) = 254 for g=16
    # Wait: Moore bound for (d,g) with g even is 2 * sum_{i=0}^{g/2-1} (d-1)^i
    # For d=3, g=16: 2 * sum_{i=0}^{7} 2^i = 2 * (2^8 - 1) = 2*255 = 510
    min_n = 510
    max_n = 940

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            return None

        half = n // 2

        # a and b must be in {2, ..., half-1} (1 is already used)
        # a != b, a != n-b, b != n-a (i.e., a+b != n)
        # Also need: 2a != 0, 2b != 0, a+b != 0, a-b != 0 mod n (no 2-cycles or 4-cycles from pairs)
        # For no 4-cycles: no two distinct generators sum to 0 or equal another generator
        # Generators: 1, -1, a, -a, b, -b
        # 4-cycle conditions: g1+g2+g3+g4 = 0 with no backtracking
        # This means g1+g2 = -(g3+g4), so we need g1+g2 != 0 for all pairs
        
        for a in range(2, half):
            if time.time() > deadline - 3:
                return None

            # Basic checks for a
            if 2 * a % n == 0:
                continue
            if a % n in (1, n - 1):
                continue

            for b in range(a + 1, half):
                if time.time() > deadline - 2:
                    return None

                # Check b
                if 2 * b % n == 0:
                    continue

                # Check no 4-cycles: all pair sums of distinct generators must be nonzero
                # and not equal to ±1, ±a, ±b (which would give a 4-cycle)
                if _has_4cycle_algebraic(n, a, b):
                    continue

                # Check no 6-cycles
                if _has_6cycle_algebraic(n, a, b):
                    continue

                # Full girth check
                if _girth_check_nb(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _has_4cycle_algebraic(n, a, b):
    """
    Check for 4-cycles in C_n(1,a,b) algebraically.
    A 4-cycle is a sequence s1,s2,s3,s4 from G={±1,±a,±b} with:
    - s1+s2+s3+s4 = 0 mod n
    - s2 != -s1, s3 != -s2, s4 != -s3, s1 != -s4 (no backtracking)
    - Not all si equal (would be a 2-cycle repeated, but that's already excluded)
    
    For 4-cycles, the structure is: s1, s2, -s2, -s1 (backtracking pair) - excluded
    Or: s1, s2, s3, s4 where s1+s2 = -(s3+s4) and s2 != -s1, s3 != -s2, etc.
    
    Actually for a simple 4-cycle v0-v1-v2-v3-v0:
    steps: s1, s2, s3, s4 with s1+s2+s3+s4=0
    No backtracking: s2 != -s1, s3 != -s2, s4 != -s3, s1 != -s4
    Also v0 != v2 (s1+s2 != 0) and v1 != v3 (s2+s3 != 0)
    
    The simplest 4-cycles come from: s1 + s2 + s3 + s4 = 0
    where {s1,s2,s3,s4} are from G with no adjacent pair being negatives.
    
    Key observation: if s1+s2 = s3+s4 = some value, or s1+s2 = -(s3+s4).
    Let's enumerate: pick s1 from G, s2 from G\{-s1}, then s3+s4 = -(s1+s2).
    For each s3 in G\{-s2}, s4 = -(s1+s2)-s3. Check s4 in G, s4 != -s3, s1 != -s4.
    """
    G = [1, n - 1, a, n - a, b, n - b]
    G_set = set(G)
    
    for s1 in G:
        for s2 in G:
            if s2 == (-s1) % n:
                continue
            partial = (s1 + s2) % n
            if partial == 0:
                continue  # v0 = v2, not a simple cycle
            for s3 in G:
                if s3 == (-s2) % n:
                    continue
                s4 = (-partial - s3) % n
                if s4 not in G_set:
                    continue
                if s4 == (-s3) % n:
                    continue
                if s1 == (-s4) % n:
                    continue
                # Also check v1 != v3: s2+s3 != 0
                if (s2 + s3) % n == 0:
                    continue
                # Found a 4-cycle
                return True
    return False


def _has_6cycle_algebraic(n, a, b):
    """
    Check for 6-cycles in C_n(1,a,b) using BFS to depth 3.
    A 6-cycle exists if two non-adjacent vertices at distance 3 from 0 are adjacent.
    """
    G = [1, n - 1, a, n - a, b, n - b]
    
    # BFS from 0 to depth 3, tracking parents to avoid backtracking
    dist = {0: 0}
    parent = {0: -1}
    level = [0]
    
    for depth in range(3):
        next_level = []
        for u in level:
            for step in G:
                v = (u + step) % n
                if v == parent.get(u, -1):
                    continue  # backtracking
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level
    
    # Check for edges between vertices at depth 3 (would form 6-cycles)
    # Also check edges between depth 2 and depth 3 that aren't parent-child
    for v in level:
        for step in G:
            w = (v + step) % n
            if w == parent.get(v, -1):
                continue
            if w in dist:
                if dist[w] == 3 and parent.get(w, -1) != v and parent.get(v, -1) != w:
                    # Two depth-3 vertices connected: 6-cycle
                    return True
                elif dist[w] == 2 and parent.get(w, -1) != v and parent.get(v, -1) != w:
                    # depth 2 and depth 3 connected (not parent-child): could form shorter cycle
                    # Actually this would be a 5-cycle or 6-cycle
                    # dist[v]=3, dist[w]=2, edge between them: cycle length = 3+2+1 = 6? 
                    # No: path 0->...->w (len 2), w->v (len 1), v->...->0 (len 3)
                    # But v is at depth 3, so path from 0 to v is length 3
                    # Cycle: 0 to w (2), w to v (1), v back to 0 (3) = 6? 
                    # Wait, that's not a simple cycle necessarily.
                    # Actually if dist[w]=2 and dist[v]=3 and there's an edge w-v,
                    # and w is not the parent of v, then we have a cycle of length 2+1+3=6
                    # But we need to check it's simple. For our purposes, flag it.
                    return True
    return False


def _girth_check_nb(n, a, b, g):
    """
    Non-backtracking BFS from vertex 0 to check girth >= g.
    State: (current_vertex, previous_vertex)
    If we reach a state where current == 0 and depth < g, we found a cycle of length depth.
    """
    G = [1, n - 1, a, n - a, b, n - b]
    max_depth = g - 1
    dist = {}
    queue = deque()
    
    start_state = (0, -1)
    dist[start_state] = 0
    queue.append(start_state)
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_depth:
            continue
        for step in G:
            v = (u + step) % n
            if v == prev:
                continue
            nd = d + 1
            if nd > max_depth:
                continue
            # Check if we've returned to start
            if v == 0 and u != -1 and nd < g:
                return False
            ns = (v, u)
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                queue.append(ns)
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
    q = k

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


# EVOLVE-BLOCK-END
