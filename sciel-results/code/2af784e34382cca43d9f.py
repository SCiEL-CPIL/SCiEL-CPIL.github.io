import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def luw_construct(k, g):
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

    def verify_girth_nonbacktracking(adj, n, min_girth):
        """
        Verify girth >= min_girth using non-backtracking walks.
        A cycle of length L exists iff there is a closed non-backtracking walk of length L.
        We check: for every vertex, do BFS on the non-backtracking state space up to
        depth (min_girth - 1) // 2, and detect if any state revisits a vertex at distance
        that would close a cycle shorter than min_girth.
        
        State: (vertex, parent) where parent is the vertex we came from (or -1 for start).
        """
        max_depth = (min_girth - 1) // 2
        for start in range(n):
            # BFS on non-backtracking states
            # dist[(v, p)] = distance from start to state (v, p)
            dist = {}
            from collections import deque
            queue = deque()
            for nb in adj[start]:
                state = (nb, start)
                dist[state] = 1
                queue.append(state)
            while queue:
                (v, p) = queue.popleft()
                d = dist[(v, p)]
                if d > max_depth:
                    continue
                for nb in adj[v]:
                    if nb == p:
                        continue  # non-backtracking
                    ns = (nb, v)
                    if ns in dist:
                        # We've seen this state before.
                        # A cycle is formed: the path from start to (v,p) of length d,
                        # plus the edge to nb, plus the path from start to (nb, v) of length dist[ns].
                        # Actually, the cycle length is d + 1 + dist[ns]... no.
                        # Let me think again.
                        # State (v, p) at depth d means we walked d steps from start to reach v, coming from p.
                        # State (nb, v) at depth dist[ns] means we walked dist[ns] steps from start to reach nb, coming from v.
                        # The edge (v, nb) connects them.
                        # The cycle would be: path1 (start->...->v) + edge(v,nb) + path2(nb->...->start)
                        # But path2 goes from nb back toward start. The length of path2 is dist[ns].
                        # So cycle length = d + 1 + dist[ns].
                        # However, this might not be a simple cycle. But for girth checking,
                        # if any non-backtracking closed walk of length < min_girth exists, girth < min_girth.
                        cycle_len = d + 1 + dist[ns]
                        if cycle_len < min_girth:
                            return False
                    else:
                        dist[ns] = d + 1
                        queue.append(ns)
        return True

    if k == 3 and g == 16:
        # Deterministic algebraic construction: Cayley graph on Z_m with generators {a, b, -a, -b, c, -c}
        # where a + b + c = 0 mod m, ensuring 3-regular.
        # Actually, for 3-regular, we need 3 pairs: {a, -a}, {b, -b}, {c, -c} with a+b+c=0.
        # This gives a 6-regular graph. Not what we want.
        
        # For 3-regular: use generators {a, b, a+b} on Z_m.
        # Edge from x to x+a, x to x+b, x to x+a+b.
        # This is 3-regular if a, b, a+b are all nonzero and distinct mod m.
        
        # Try various m and generator pairs.
        # Girth is related to the shortest nontrivial relation among a, b, a+b.
        
        def build_cayley(m, a, b):
            """Build 3-regular Cayley graph on Z_m with generators a, b, a+b."""
            c = (a + b) % m
            # Check that a, b, c are all nonzero and distinct
            if a == 0 or b == 0 or c == 0:
                return None
            if a == b or a == c or b == c:
                return None
            # Also need -a, -b, -c to not equal a, b, or c (to avoid loops/multi-edges)
            # Actually, in a Cayley graph on Z_m, the edge from x to x+a is the same as from x+a to x.
            # So we just need {a, b, c} to be a set of 3 distinct nonzero elements.
            # But we also need that a != -b, b != -c, c != -a to avoid the same edge being counted twice.
            # Actually, if a = -b, then edge x->x+a is the same as x->x-b, which is not in our generator set unless b=-a.
            # Let's just build it and check for simplicity.
            edges = []
            seen = set()
            for x in range(m):
                for gen in [a, b, c]:
                    y = (x + gen) % m
                    if x < y:
                        e = (x, y)
                    else:
                        e = (y, x)
                    if e not in seen:
                        seen.add(e)
                        edges.append(e)
            # Verify 3-regular
            adj = [[] for _ in range(m)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            for i in range(m):
                if len(adj[i]) != 3:
                    return None
            return m, edges, adj

        def find_good_cayley(m):
            """Try to find generators a, b for Cayley graph on Z_m with girth >= 16."""
            for a in range(1, m // 2 + 1):
                for b in range(a + 1, m // 2 + 1):
                    if time.time() > deadline - 5:
                        return None
                    c = (a + b) % m
                    if c == 0 or c == a or c == b:
                        continue
                    # Also check that -a, -b, -c don't coincide with a, b, c in a way that creates issues
                    # For simplicity, just build and check
                    result = build_cayley(m, a, b)
                    if result is None:
                        continue
                    n, edges, adj = result
                    if verify_girth_nonbacktracking(adj, n, g):
                        return n, edges
            return None

        # Try various m values. Smaller m is better (fewer vertices).
        # Moore bound for (3,16) is 512. So m >= 512.
        # Try m = 512, 513, 514, ... up to some limit.
        best_result = None
        for m in range(512, 600):
            if time.time() > deadline - 10:
                break
            result = find_good_cayley(m)
            if result is not None:
                n, edges = result
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                    # Found a valid graph; try to find smaller one
                    if n <= 512:
                        break
        
        if best_result is not None:
            return best_result
        
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
