import time
import itertools
import numpy as np
import math

def construct(k, g):
    _t0 = time.time()
    _DEADLINE = 115.0

    # ---- Fallback: Lazebnik-Ustimenko-Woldar graph CD(d, q) ----
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

    def luw_ok(k, g):
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        return d + 5 >= g

    # ---- Fast girth check for circulants ----
    def circ_girth_fast(n, a, b, g):
        # Check if there is a cycle of length < g
        # Steps are +/-a, +/-b
        steps = [a, -a, b, -b]
        # Remove duplicates if a==b or a==-b mod n
        # For simplicity, assume a != b, a != -b, etc.
        # We check non-backtracking walks of length L
        # A walk is a sequence of steps s_1...s_L
        # Non-backtracking: s_i != -s_{i-1}
        # Cycle: sum(s_i) == 0 mod n
        # We can do BFS on the state (current_pos, last_step)
        # But n is small (<= 320).
        # States: n * 4 = 1280.
        # We can do BFS up to depth g-1.
        # If we return to (0, step) with length < g, we have a cycle.
        # Actually, we just need to check if there is ANY cycle < g.
        # This is equivalent to checking if the graph has a cycle < g.
        # For circulants, we can just check the distances.
        # But the simple step sum check is:
        # Is there a sequence of L steps in {a, -a, b, -b} such that sum=0 and no consecutive opposites?
        # We can iterate L from 3 to g-1.
        # For each L, iterate all 4^L sequences?
        # 4^9 is 262k. For each n, this is fast.
        # But we do this for many a,b.
        # Better: Precompute all possible sums of length L?
        # No, just iterate.
        
        # Optimization: If a==b, it's 2-regular.
        # If a==-b, it's 2-regular.
        if a == b or a == n - b:
            return False # 2-regular, girth is n/2?
            
        # Check for small cycles
        # 3-cycle: a+b-a=0? No. a+a+b=0?
        # We can just use the matrix method for the specific n if it's fast enough.
        # For n=320, matrix mult is ~1ms.
        # 12800 pairs * 1ms = 12s.
        # This is acceptable.
        
        A = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            A[i, (i + a) % n] = 1
            A[i, (i + b) % n] = 1
            A[i, (i - a) % n] = 1
            A[i, (i - b) % n] = 1
        
        deg = A.sum(axis=1)
        if not np.all(deg == 4):
            return False
            
        B = A.astype(np.int64)
        for L in range(3, g):
            newB = B @ A - B @ np.diag(deg)
            if int(np.trace(newB)) > 0:
                return False
            B = newB
        return True

    # ---- Main logic ----
    if k == 4 and g == 10:
        # Search for n from Moore bound to 320
        moore = 1 + 4 * sum(3 ** i for i in range(4)) # 161
        # Lower bound is 243
        start_n = 243
        end_n = 320
        
        for n in range(start_n, end_n + 1):
            if time.time() - _t0 > _DEADLINE:
                break
            # Search a, b
            found = False
            for a in range(1, n // 2 + 1):
                if time.time() - _t0 > _DEADLINE:
                    break
                for b in range(a + 1, n // 2 + 1):
                    if circ_girth_fast(n, a, b, g):
                        edges = []
                        for i in range(n):
                            edges.append((i, (i + a) % n))
                            edges.append((i, (i + b) % n))
                        return n, edges
            if time.time() - _t0 > _DEADLINE:
                break

    # Fallback to LUW
    if luw_ok(k, g):
        return luw_construct(k, g)
        
    return luw_construct(k, g)
