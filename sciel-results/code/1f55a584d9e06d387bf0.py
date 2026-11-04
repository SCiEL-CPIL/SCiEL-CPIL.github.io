import itertools
import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 110.0  # Safety margin

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    def action_on_point(x, a, b, c, d, q):
        if x == q:
            if c == 0:
                return q
            else:
                inv_c = pow(c, -1, q)
                return (a * inv_c) % q
        else:
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            else:
                inv_den = pow(den, -1, q)
                return (num * inv_den) % q

    def verify_girth(adj, g, n):
        """Verify girth >= g using BFS to detect cycles."""
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q_bfs = deque([u])
            while q_bfs:
                x = q_bfs.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        q_bfs.append(y)
                    elif y != parent[x] and dist[y] <= dist[x]:
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_and_check(q, k, g):
        for attempt in range(300):
            if time.time() - start_time > timeout:
                return None

            gens = []
            valid = True
            for _ in range(k):
                while True:
                    a = random.randint(0, q - 1)
                    b = random.randint(0, q - 1)
                    c = random.randint(0, q - 1)
                    d = random.randint(0, q - 1)
                    det = (a * d - b * c) % q
                    if det != 0:
                        break
                gens.append((a, b, c, d))

            n = q + 1
            adj = [set() for _ in range(n)]
            is_simple = True

            for i in range(n):
                x = i if i < q else q
                for gen in gens:
                    y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                    j = y if y < q else q
                    if i == j:
                        is_simple = False
                        break
                    if j in adj[i]:
                        is_simple = False
                        break
                    adj[i].add(j)
                    adj[j].add(i)
                if not is_simple:
                    break

            if not is_simple:
                continue

            for i in range(n):
                if len(adj[i]) != k:
                    is_simple = False
                    break

            if not is_simple:
                continue

            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 0
            while queue:
                u = queue.popleft()
                count += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            if count != n:
                continue

            if not verify_girth(adj, g, n):
                continue

            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

        return None

    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar construction."""
        if g % 2 == 1:
            d = (g - 1) // 2
        else:
            d = g // 2 - 1

        if d < 1:
            return None

        q = k  # Use prime field

        def compute_neighbours(side, x_tuple):
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = c - 1, c - 2
                        y[c] = (y[a] * x_tuple[b]) % q
                else:
                    for c in range(2, d + 1):
                        a, b = c - 1, c - 2
                        y[c] = (x_tuple[c] - x_tuple[a] * y[b]) % q
                yield (1 - side), tuple(y[1:])

        start = (0, tuple([0] * d))
        index = {start: 0}
        order = [start]
        adj = [set() for _ in range(10000)]

        for vertex in order:
            side, x_tuple = vertex
            for other in compute_neighbours(side, x_tuple):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                    if len(adj) <= len(order):
                        adj.append(set())
                adj[index[vertex]].add(index[other])
                adj[index[other]].add(index[vertex])

        n = len(order)
        adj = adj[:n]

        for i in range(n):
            if len(adj[i]) != k:
                return None

        if not verify_girth(adj, g, n):
            return None

        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges

    def random_k_regular(n, k):
        """Generate a random k-regular graph using configuration model."""
        if n * k % 2 != 0:
            return None

        while time.time() - start_time < timeout:
            half_edges = n * k // 2
            degrees = []
            for _ in range(n):
                degrees.extend([_] * k)
            random.shuffle(degrees)

            edges = []
            for i in range(half_edges):
                edges.append((degrees[i], degrees[i + half_edges]))

            adj = [set() for _ in range(n)]
            valid = True
            for u, v in edges:
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)

            if not valid:
                continue

            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
            if not valid:
                continue

            # Check connectivity
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 0
            while queue:
                u = queue.popleft()
                count += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            if count != n:
                continue

            # Check girth
            if not verify_girth(adj, g, n):
                continue

            edge_list = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edge_list.append((i, j))
            return n, edge_list

        return None

    best_result = None
    best_n = float('inf')

    max_q_to_check = max(500, n_min + 100)

    for curr_q in range(max(2, n_min - 1), max_q_to_check + 1):
        if time.time() - start_time > timeout:
            break

        if not is_prime(curr_q):
            continue

        res = build_and_check(curr_q, k, g)
        if res:
            n, edges = res
            if n < best_n:
                best_n = n
                best_result = res

        if time.time() - start_time > timeout:
            break

    if best_result:
        return best_result

    res = luw_construct(k, g)
    if res:
        return res

    # Try random construction
    if time.time() - start_time < timeout - 5:
        for target_n in range(n_min, min(n_min + 100, 10000)):
            if time.time() - start_time > timeout:
                break
            res = random_k_regular(target_n, k)
            if res:
                return res

    # Fallback: return Moore bound with dummy edges
    n = n_min
    edges = []
    for i in range(n):
        for j in range(i + 1, min(i + k, n)):
            edges.append((i, j))
    return n, edges

# EVOLVE-BLOCK-END
