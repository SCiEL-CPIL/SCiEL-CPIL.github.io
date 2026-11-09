"""Decompose each fixed instance graph into edge-disjoint cycles + single edges."""
import time
import random
from random import Random


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    avail = set(tuple(sorted(e)) for e in edges)
    pieces = []

    def try_greedy_cycle(start, max_attempts=3):
        """Try to extend from start to form a cycle back to start. Dry run."""
        for _ in range(max_attempts):
            path = [start]
            visited = {start}
            cur = start
            for _ in range(len(adj) * 2):
                if time.time() > deadline:
                    return None
                nbrs = [nb for nb in adj[cur] if (min(cur, nb), max(cur, nb)) in avail]
                if not nbrs:
                    return None
                if len(path) >= 3 and start in nbrs:
                    return path[:]
                unvisited = [nb for nb in nbrs if nb not in visited]
                if unvisited:
                    weights = [len(adj[nb]) + 1 for nb in unvisited]
                    total = sum(weights)
                    r = random.random() * total
                    cum = 0
                    chosen = unvisited[0]
                    for nb, w in zip(unvisited, weights):
                        cum += w
                        if r <= cum:
                            chosen = nb
                            break
                else:
                    if start in nbrs and len(path) >= 3:
                        return path[:]
                    return None
                path.append(chosen)
                visited.add(chosen)
                cur = chosen
        return None

    while time.time() < deadline:
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break

        best_cycle = None
        best_len = 0

        rng_local = Random()
        num_starts = min(len(candidates), 20)
        starts = rng_local.sample(candidates, num_starts) if num_starts < len(candidates) else candidates

        for s in starts:
            if time.time() > deadline:
                break
            cycle = try_greedy_cycle(s, max_attempts=3)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle

        if not best_cycle:
            for s in starts[:5]:
                if time.time() > deadline:
                    break
                cycle = try_greedy_cycle(s, max_attempts=5)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle

        if not best_cycle:
            break

        # Commit the best cycle found
        for i in range(len(best_cycle)):
            a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = (min(a, b), max(a, b))
            avail.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        pieces.append(best_cycle)

    if avail:
        rem_adj = {}
        for (u, v) in avail:
            rem_adj.setdefault(u, set()).add(v)
            rem_adj.setdefault(v, set()).add(u)

        for u in list(rem_adj.keys()):
            if time.time() > deadline:
                break
            nbrs_u = list(rem_adj[u])
            for i in range(len(nbrs_u)):
                for j in range(i + 1, len(nbrs_u)):
                    v, w = nbrs_u[i], nbrs_u[j]
                    if w in rem_adj.get(v, set()):
                        e1 = (min(u, v), max(u, v))
                        e2 = (min(v, w), max(v, w))
                        e3 = (min(u, w), max(u, w))
                        if e1 in avail and e2 in avail and e3 in avail:
                            cycle = [u, v, w]
                            for e in [e1, e2, e3]:
                                avail.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
                            adj[v].discard(w)
                            adj[w].discard(v)
                            adj[u].discard(w)
                            adj[w].discard(u)
                            pieces.append(cycle)
                            break
                else:
                    continue
                break

    for e in sorted(avail):
        pieces.append(list(e))

    return pieces


def construct():
    deadline = time.time() + 140.0
    random.seed(42)
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    total_size = sum(n * n for n, _, _ in specs)
    for i, spec in enumerate(specs):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            results.append([list(e) for e in edges])
            continue
        frac = (n * n) / total_size
        graph_deadline = time.time() + max(5.0, remaining_time * frac * 0.9)
        results.append(_decompose(edges, graph_deadline))
    return results
