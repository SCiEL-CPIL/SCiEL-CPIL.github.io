"""Bootstrap search program for Large graphs with no regular induced subgraph on six vertices."""
import time
import random
import itertools
import math

DEADLINE = 150
SAFETY_MARGIN = 10
START_TIME = None


def find_violation(adj, n):
    """Find any violating subset (size >= 6) that is regular, or None."""
    for k in range(6, n + 1):
        for combo in itertools.combinations(range(n), k):
            m = k
            degs = [0] * m
            for i in range(m):
                vi = combo[i]
                row = adj[vi]
                for j in range(i + 1, m):
                    if row[combo[j]]:
                        degs[i] += 1
                        degs[j] += 1
            d0 = degs[0]
            for d in degs[1:]:
                if d != d0:
                    break
            else:
                return combo
    return None


def random_graph(n, p=0.5, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    adj = [[False] * n for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
    return adj, edges


def get_edges_from_adj(adj, n):
    e = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                e.append([u, v])
    return e


def local_search(adj, n, max_iters=100, deadline=None):
    rng = random.Random()
    current = [row[:] for row in adj]

    viol = find_violation(current, n)
    if viol is None:
        return n, get_edges_from_adj(current, n), True

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        viol = find_violation(current, n)
        if viol is None:
            return n, get_edges_from_adj(current, n), True

        vlist = list(viol)
        if len(vlist) >= 2:
            u, v = rng.sample(vlist, 2)
        else:
            u, v = rng.choice([(a, b) for a in range(n) for b in range(a + 1, n)])

        current[u][v] = not current[u][v]
        current[v][u] = current[u][v]

    viol = find_violation(current, n)
    return n, get_edges_from_adj(current, n), viol is None


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(12345)

    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        attempts = 0
        max_attempts = 10 if n >= 18 else 40

        while time.time() < deadline and attempts < max_attempts:
            attempts += 1

            p = rng.random() * 0.3 + 0.35

            adj, edges = random_graph(n, p=p, seed=None)

            viol = find_violation(adj, n)

            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                    break
            else:
                new_n, new_edges, new_valid = local_search(
                    adj, n,
                    max_iters=40 if n <= 15 else 15,
                    deadline=deadline
                )
                if new_valid and new_n > best_n:
                    best_n = new_n
                    best_edges = new_edges
                    best_valid = True
                    break

        if best_valid and best_n >= n:
            continue

    if not best_valid or best_n < 5:
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
