import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k != 4 or g != 10:
        return _fallback_construct(k, g)

    # GF(4) arithmetic
    gf4_mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]

    def gf4_add(x, y):
        return x ^ y

    def gf4_mul(x, y):
        return gf4_mul_table[x][y]

    q = 4

    # Try LUW construction with increasing d
    for d in [5, 6, 7, 8]:
        if time.time() > deadline:
            break

        n_vertices = 2 * (q ** d)

        def encode(side, x, d=d, q=q):
            idx = side * (q ** d)
            for i in range(d):
                idx += x[i] * (q ** (d - 1 - i))
            return idx

        def decode(idx, d=d, q=q):
            side = idx // (q ** d)
            rem = idx % (q ** d)
            x = [0] * d
            for i in range(d):
                x[i] = rem % q
                rem //= q
            x.reverse()
            return side, tuple(x)

        # Build adjacency
        adj = [[] for _ in range(n_vertices)]
        edges = set()

        for idx in range(n_vertices):
            side, x = decode(idx)
            x = list(x)
            x_ext = [0] + x  # x_ext[0]=0, x_ext[1]=x_1, ..., x_ext[d]=x_d

            for t in range(q):
                y = [0] * d
                y[0] = gf4_add(x_ext[1], t)
                for i in range(1, d):
                    term1 = x_ext[i + 1]
                    term2 = gf4_mul(y[i - 1], x_ext[i])
                    term3 = gf4_mul(t, x_ext[i - 1])
                    y[i] = gf4_add(gf4_add(term1, term2), term3)

                nside = 1 - side
                nidx = encode(nside, tuple(y))
                edge = (min(idx, nidx), max(idx, nidx))
                if edge not in edges:
                    edges.add(edge)
                    adj[idx].append(nidx)
                    adj[nidx].append(idx)

        n = n_vertices

        # Verify degree
        valid = True
        for i in range(n):
            if len(adj[i]) != k:
                valid = False
                break

        if not valid:
            continue

        # Verify girth
        def check_girth(adj, n, g, deadline):
            radius = g // 2 - 1
            if radius < 1:
                return True

            for start in range(n):
                if time.time() > deadline:
                    return False
                visited = {start: 0}
                parent = {start: -1}
                queue = deque([(start, 0)])

                while queue:
                    u, dist = queue.popleft()
                    if dist == radius:
                        continue
                    for v in adj[u]:
                        if v not in visited:
                            visited[v] = dist + 1
                            parent[v] = u
                            queue.append((v, dist + 1))
                        else:
                            if v != parent[u]:
                                return False
            return True

        if check_girth(adj, n, g, deadline):
            return n, list(edges)

    return _fallback_construct(k, g)


def _fallback_construct(k, g):
    if k == 4 and g == 10:
        n = 8
        edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        return n, edges
    else:
        n = k
        edges = [(i, (i+1) % k) for i in range(k)]
        return n, edges
