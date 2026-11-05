"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    deadline = time.time() + 140.0
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def _luw_graph(k, g):
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
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Primary Strategy: Search for a Circulant Graph
    # Target: 4-regular, girth >= 10
    # Known optimum: 320. Moore bound: 243.
    if k == 4 and g == 10:
        def check_girth_circulant(N, S):
            # S is a set of 4 generators
            # Check for cycles of length < 10
            # BFS from 0
            dist = [-1] * N
            dist[0] = 0
            queue = [0]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                du = dist[u]
                if du >= 5: # We only care about cycles < 10, so depth up to 4 is enough to detect them?
                    # Actually, if we reach depth 4, and find an edge to a node with dist <= 4, 
                    # the cycle length is at most 4+1+4 = 9.
                    # If dist[u] is 4, and neighbor v has dist 4, cycle <= 9.
                    # If dist[u] is 4, and neighbor v has dist 3, cycle <= 8.
                    # So we must check edges from nodes with dist <= 4.
                    pass
                for s in S:
                    v1 = (u + s) % N
                    v2 = (u - s) % N
                    for v in (v1, v2):
                        if dist[v] == -1:
                            dist[v] = du + 1
                            queue.append(v)
                        else:
                            # Found a cycle
                            # Length is at most du + 1 + dist[v]
                            # But we need to ensure it's a simple cycle.
                            # In a BFS tree, if u and v are not parent-child, it's a cycle.
                            # The length is dist[u] + dist[v] + 1?
                            # Actually, the path 0->...->u->v->...->0 is not necessarily simple.
                            # However, if dist[u] + dist[v] + 1 < 10, it's a strong indicator.
                            # For strict correctness, we can just check if dist[u] + dist[v] + 1 < 10.
                            # If this condition holds, we have a cycle of length <= 9.
                            if dist[u] + dist[v] + 1 < 10:
                                return False
            return True

        def generate_circulant(N):
            # Try to find a valid set of 4 generators
            # Use random sampling
            for _ in range(2000):
                if time.time() > deadline:
                    return None
                # Pick 4 distinct non-zero elements from 1..N//2
                # Ensure s != -s (i.e., 2s != 0 mod N)
                # If N is even, N/2 is self-inverse. Avoid it?
                # If s = N/2, then s = -s. This creates a 2-cycle? No, (i, i+N/2) is same as (i+N/2, i).
                # It's a valid edge. But s and -s are same.
                # We need 4 distinct "directions".
                # So we should pick s such that s != -s.
                candidates = [i for i in range(1, N//2) if 2*i % N != 0]
                if len(candidates) < 4:
                    continue
                S = random.sample(candidates, 4)
                # Quick check: no 4-cycles
                # s1+s2 == s3+s4?
                # We can just use the BFS check
                if check_girth_circulant(N, S):
                    # Construct edges
                    edges = set()
                    for i in range(N):
                        for s in S:
                            j1 = (i + s) % N
                            j2 = (i - s) % N
                            if i < j1:
                                edges.add((i, j1))
                            if i < j2:
                                edges.add((i, j2))
                    # Verify degree
                    # Each vertex has 4 neighbors (s, -s for 4 s's).
                    # If s1 = s2, we have duplicates. But sample ensures distinct.
                    # If s1 = -s2, we have duplicates.
                    # candidates ensures s != -s.
                    # But s1 could be -s2?
                    # If s1 = 1, s2 = N-1.
                    # We picked from 1..N//2.
                    # If N is even, N//2 is not in candidates if 2*(N//2)=N=0.
                    # So s is in 1..N//2 - 1.
                    # -s is in N//2 + 1 .. N-1.
                    # So s1 != -s2 for distinct s1, s2 in 1..N//2.
                    # So edges are distinct.
                    return N, list(edges)
            return None

        # Try sizes from 320 down to 243
        # 320 is known to work.
        for N in range(320, 242, -5):
            if time.time() > deadline:
                break
            res = generate_circulant(N)
            if res:
                return res

    # Fallback to LUW
    return _luw_graph(k, g)
# EVOLVE-BLOCK-END
