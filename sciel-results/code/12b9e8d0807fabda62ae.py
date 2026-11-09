import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145

    best_n = 39
    best_edges = []

    def can_add_fast(adj, u, v):
        """Check if adding edge (u,v) creates a C4.
        C4 u-v-w-x-u: need w in N(v), x in N(u), w~x, w!=x, w!=u, x!=v.
        """
        nv = adj[v] & ~(1 << u)
        nu = adj[u] & ~(1 << v)
        c = nv
        while c:
            a_bit = c & -c
            a = a_bit.bit_length() - 1
            c ^= a_bit
            if (adj[a] & nu) != 0:
                return False
        return True

    def try_construct(n, seed_val, deadline):
        min_deg_needed = n - 39
        if min_deg_needed <= 0:
            return None

        rng = random.Random(seed_val)
        adj = [0] * n
        deg = [0] * n
        edges = []
        in_graph = [[False] * n for _ in range(n)]

        heap = []
        for u in range(n):
            for v in range(u + 1, n):
                pri = -(2 * min_deg_needed) + rng.random() * 0.01
                heapq.heappush(heap, (pri, u, v))

        edge_count = 0
        max_edges = n * (n - 1) // 2

        while heap and edge_count < max_edges:
            if time.time() > deadline - 3:
                break

            pri, u, v = heapq.heappop(heap)

            if in_graph[u][v]:
                continue

            expected_pri = -(min_deg_needed - deg[u] + min_deg_needed - deg[v])
            if pri < expected_pri - 0.1:
                new_pri = expected_pri + rng.random() * 0.01
                heapq.heappush(heap, (new_pri, u, v))
                continue

            if can_add_fast(adj, u, v):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
                in_graph[u][v] = True
                in_graph[v][u] = True
                edge_count += 1

                for w in range(n):
                    if w == u or w == v:
                        continue
                    if not in_graph[u][w]:
                        new_pri = -(min_deg_needed - deg[u] + min_deg_needed - deg[w]) + rng.random() * 0.01
                        heapq.heappush(heap, (new_pri, u, w))
                    if not in_graph[v][w]:
                        new_pri = -(min_deg_needed - deg[v] + min_deg_needed - deg[w]) + rng.random() * 0.01
                        heapq.heappush(heap, (new_pri, v, w))

        if min(deg) >= min_deg_needed:
            return edges

        if time.time() > deadline - 5:
            return None

        for _ in range(50):
            if time.time() > deadline - 5:
                break

            still_deficient = [i for i in range(n) if deg[i] < min_deg_needed]
            if not still_deficient:
                break

            progress = False
            for u in still_deficient:
                if deg[u] >= min_deg_needed:
                    continue
                candidates = []
                for v in range(n):
                    if v == u or in_graph[u][v]:
                        continue
                    if can_add_fast(adj, u, v):
                        candidates.append(v)

                candidates.sort(key=lambda v: deg[v])
                for v in candidates:
                    if deg[u] >= min_deg_needed:
                        break
                    if can_add_fast(adj, u, v):
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        deg[u] += 1
                        deg[v] += 1
                        edges.append([u, v])
                        in_graph[u][v] = True
                        in_graph[v][u] = True
                        progress = True
                        break

            if not progress:
                break

        if min(deg) >= min_deg_needed:
            return edges

        return None

    for target_n in range(47, 39, -1):
        if time.time() > deadline - 10:
            break

        found = False
        for restart in range(300):
            if time.time() > deadline - 10:
                break

            seed_val = restart * 7919 + 13 * target_n + 1000
            result = try_construct(target_n, seed_val, deadline)

            if result is not None:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = result[:]
                found = True
                break

        if best_n >= target_n:
            break

    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
