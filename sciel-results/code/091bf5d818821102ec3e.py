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

    def edge_key(a, b):
        return (min(a, b), max(a, b))

    def find_cycle_dfs(start, max_len, visited, path, avail_set, adj_map):
        """DFS to find a cycle back to start."""
        if time.time() > deadline:
            return None
        if len(path) >= 3 and start in adj_map.get(path[-1], set()):
            if edge_key(start, path[-1]) in avail_set:
                return path[:]
        if len(path) >= max_len:
            return None
        cur = path[-1]
        nbrs = [nb for nb in adj_map.get(cur, set()) if edge_key(cur, nb) in avail_set]
        unvisited = [nb for nb in nbrs if nb not in visited]
        unvisited.sort(key=lambda x: len(adj_map.get(x, set())), reverse=True)
        for nb in unvisited:
            visited.add(nb)
            path.append(nb)
            result = find_cycle_dfs(start, max_len, visited, path, avail_set, adj_map)
            if result:
                return result
            path.pop()
            visited.discard(nb)
        return None

    def greedy_cycle(start, max_attempts=5, max_len=None):
        """Try to find a cycle from start using greedy + DFS fallback."""
        if max_len is None:
            max_len = min(len(adj), 15)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            path = [start]
            visited = {start}
            cur = start
            for _ in range(max_len):
                if time.time() > deadline:
                    return None
                nbrs = [nb for nb in adj[cur] if edge_key(cur, nb) in avail]
                if not nbrs:
                    break
                if len(path) >= 3 and start in nbrs:
                    return path[:]
                unvisited = [nb for nb in nbrs if nb not in visited]
                if not unvisited:
                    if start in nbrs and len(path) >= 3:
                        return path[:]
                    break
                weights = [len(adj.get(nb, set())) + 1 for nb in unvisited]
                total = sum(weights)
                r = random.random() * total
                cum = 0
                chosen = unvisited[0]
                for nb, w in zip(unvisited, weights):
                    cum += w
                    if r <= cum:
                        chosen = nb
                        break
                path.append(chosen)
                visited.add(chosen)
                cur = chosen
        # Fallback to DFS
        path = [start]
        visited = {start}
        result = find_cycle_dfs(start, max_len, visited, path, avail, adj)
        if result:
            return result
        return None

    # Phase 1: Extract longer cycles
    while time.time() < deadline:
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break

        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        
        best_cycle = None
        best_len = 0

        num_starts = min(len(candidates), 15)
        starts = candidates[:num_starts]

        for s in starts:
            if time.time() > deadline:
                break
            cycle = greedy_cycle(s, max_attempts=4, max_len=12)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
            if best_len >= 10:
                break

        if not best_cycle:
            for s in starts[:8]:
                if time.time() > deadline:
                    break
                path = [s]
                visited = {s}
                cycle = find_cycle_dfs(s, 10, visited, path, avail, adj)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle

        if not best_cycle:
            break

        for i in range(len(best_cycle)):
            a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = edge_key(a, b)
            avail.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        pieces.append(best_cycle)

    # Phase 2: Extract 3-cycles from remaining edges
    if avail and time.time() < deadline:
        rem_adj = {}
        for (u, v) in avail:
            rem_adj.setdefault(u, set()).add(v)
            rem_adj.setdefault(v, set()).add(u)
        
        three_cycles = []
        for u in sorted(rem_adj.keys()):
            nbrs_u = sorted(rem_adj[u])
            for i in range(len(nbrs_u)):
                v = nbrs_u[i]
                if v <= u:
                    continue
                for w in nbrs_u[i+1:]:
                    if w in rem_adj.get(v, set()):
                        three_cycles.append([u, v, w])
        
        for cycle in three_cycles:
            if time.time() > deadline:
                break
            u, v, w = cycle
            e1 = edge_key(u, v)
            e2 = edge_key(v, w)
            e3 = edge_key(u, w)
            if e1 in avail and e2 in avail and e3 in avail:
                avail.discard(e1)
                avail.discard(e2)
                avail.discard(e3)
                adj[u].discard(v)
                adj[v].discard(u)
                adj[v].discard(w)
                adj[w].discard(v)
                adj[u].discard(w)
                adj[w].discard(u)
                pieces.append(cycle)

    # Phase 3: Remaining edges as singletons
    for e in sorted(avail):
        pieces.append(list(e))

    return pieces


def construct():
    deadline = time.time() + 145.0
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
        graph_deadline = time.time() + max(5.0, remaining_time * frac * 0.95)
        results.append(_decompose(edges, graph_deadline))
    return results
