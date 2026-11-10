import time
import random
from collections import defaultdict


def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _edge_set(edges):
    return {tuple(sorted(e)) for e in edges}


def _is_simple(path):
    return len(set(path)) == len(path)


def _path_edges_set(path):
    return {tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)}


def _try_merge(paths, i, j):
    """Try to merge paths[i] and paths[j] by connecting endpoints. Returns merged path or None."""
    P1 = list(paths[i])
    P2 = list(paths[j])

    if P1[-1] == P2[0]:
        merged = P1 + P2[1:]
        if _is_simple(merged):
            return merged

    if P1[-1] == P2[-1]:
        P2r = P2[::-1]
        merged = P1 + P2r[1:]
        if _is_simple(merged):
            return merged

    if P1[0] == P2[0]:
        P1r = P1[::-1]
        merged = P1r + P2[1:]
        if _is_simple(merged):
            return merged

    if P1[0] == P2[-1]:
        P1r = P1[::-1]
        merged = P1r + P2[1:]
        if _is_simple(merged):
            return merged

    return None


def _improve_by_merge(paths, max_iters=500):
    """Repeatedly try to merge any two paths until no more merges possible."""
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(paths)
        for i in range(n):
            for j in range(i + 1, n):
                merged = _try_merge(paths, i, j)
                if merged is not None:
                    paths[i] = merged
                    paths.pop(j)
                    improved = True
                    break
            if improved:
                break
    return paths


def _simple_paths_from_trail(trail, adj, t0, tlimit):
    """Decompose a vertex trail (may repeat vertices/edges) into simple paths.

    Recursively extract cycles: the first repeated vertex delimits a cycle,
    which we peel off as one simple path; the remaining prefix and the
    suffix (which share the repeated vertex) are handled recursively.
    Falls back to a greedy cover of the trail's edges if extraction stalls
    (e.g. a duplicate edge makes a 'cycle' non-simple).
    """
    if len(trail) < 2:
        return []
    if _is_simple(trail):
        return [list(trail)]

    # Find first repeated vertex
    seen = {}
    split_idx = None
    for i, v in enumerate(trail):
        if v in seen:
            split_idx = i
            break
        seen[v] = i

    if split_idx is None:
        return [list(trail)]

    prefix = trail[:split_idx + 1]
    suffix = trail[split_idx:]
    cycle = prefix

    if _is_simple(cycle):
        rest = suffix[1:]
        rest_paths = _simple_paths_from_trail(rest, adj, t0, tlimit) if len(rest) >= 2 else []
        return [cycle] + rest_paths

    # Extraction stalled (e.g. duplicate edge). Fallback: greedily cover the
    # trail's distinct edges with simple paths.
    e_set = _path_edges_set(trail)
    if not e_set:
        return [list(trail)]
    rem = set(e_set)
    adj2 = defaultdict(set)
    for u, v in rem:
        adj2[u].add(v)
        adj2[v].add(u)
    out = []
    while rem and (time.time() - t0) < tlimit:
        starts = [v for v in adj2 if adj2[v]]
        if not starts:
            break
        start = min(starts, key=lambda v: (len(adj2[v]), v))
        path = [start]
        in_path = {start}
        while True:
            v = path[-1]
            nxts = [u for u in adj2[v] if tuple(sorted((v, u))) in rem and u not in in_path]
            if not nxts:
                break
            nxt = min(nxts, key=lambda u: (len(adj2[u]), u))
            rem.discard(tuple(sorted((v, nxt))))
            adj2[v].discard(nxt)
            adj2[nxt].discard(v)
            path.append(nxt)
            in_path.add(nxt)
        if len(path) >= 2:
            out.append(path)
    for e in rem:
        out.append([e[0], e[1]])
    return out


def _eulerian_decompose(edges, adj, rng, t0, tlimit):
    """Pair odd-degree vertices, build an Eulerian circuit, split at the dummy
    edges into trails, then decompose each trail into simple paths."""
    remaining = set(_edge_set(edges))
    if not remaining:
        return []

    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1

    odd = [v for v, d in deg.items() if d % 2 == 1]
    n_odd = len(odd)

    dummy_edges = []
    if n_odd > 0:
        odd = list(odd)
        rng.shuffle(odd)
        for i in range(0, n_odd, 2):
            if i + 1 < n_odd:
                dummy_edges.append((odd[i], odd[i + 1]))

    aug_edges = list(remaining) + [(a, b) for a, b in dummy_edges]

    edge_index = defaultdict(list)
    for idx, (u, v) in enumerate(aug_edges):
        edge_index[u].append((v, idx))
        edge_index[v].append((u, idx))

    # Hierholzer's algorithm for the Eulerian circuit
    circuit = []
    stack = [aug_edges[0][0]]
    used_edges = set()
    while stack:
        v = stack[-1]
        found = False
        while edge_index.get(v):
            nb, idx = edge_index[v].pop()
            if idx not in used_edges:
                used_edges.add(idx)
                stack.append(nb)
                found = True
                break
        if not found:
            circuit.append(stack.pop())
    circuit.reverse()

    dummy_set = {tuple(sorted((a, b))) for a, b in dummy_edges}

    # Split the circuit into trails at the dummy edges
    trails = []
    current = [circuit[0]]
    for i in range(len(circuit) - 1):
        u, v = circuit[i], circuit[i + 1]
        e = tuple(sorted((u, v)))
        if e in dummy_set:
            if len(current) >= 2:
                trails.append(current)
            current = [v]
        else:
            current.append(v)
    if len(current) >= 2:
        trails.append(current)

    # Decompose each trail into simple paths
    paths = []
    for tr in trails:
        paths.extend(_simple_paths_from_trail(tr, adj, t0, tlimit))
    return paths


def _greedy_path_cover(edges, adj, rng):
    """Greedy path cover with degree-based heuristics (always keeps paths simple)."""
    remaining = set(_edge_set(edges))
    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1

    paths = []
    while remaining:
        candidates = [v for v, d in deg.items() if d > 0]
        if not candidates:
            break

        odd = [v for v in candidates if deg[v] % 2 == 1]
        leaves = [v for v in candidates if deg[v] == 1]

        if odd and rng.random() < 0.7:
            start = rng.choice(odd)
        elif leaves and rng.random() < 0.5:
            start = rng.choice(leaves)
        else:
            min_d = min(deg[v] for v in candidates)
            low = [v for v in candidates if deg[v] == min_d]
            start = rng.choice(low)

        path = [start]
        in_path = {start}

        while True:
            v = path[-1]
            nbrs = [u for u in adj[v] if tuple(sorted((v, u))) in remaining]
            if not nbrs:
                break
            unvis = [u for u in nbrs if u not in in_path]
            if unvis:
                min_d = min(deg[u] for u in unvis)
                best = [u for u in unvis if deg[u] == min_d]
                nxt = rng.choice(best)
            else:
                break
            e = tuple(sorted((v, nxt)))
            remaining.discard(e)
            deg[v] -= 1
            deg[nxt] -= 1
            if deg[v] <= 0:
                del deg[v]
            if deg[nxt] <= 0:
                del deg[nxt]
            path.append(nxt)
            in_path.add(nxt)

        if len(path) >= 2:
            paths.append(path)

    for e in remaining:
        paths.append([e[0], e[1]])
    return paths


def _solve_one(edges, time_limit=40.0):
    """Solve one graph instance, combining the Gallai construction with
    greedy restarts and endpoint-merge refinement."""
    adj = _build_adj(edges)
    best = None
    best_count = float('inf')
    t0 = time.time()

    def consider(paths):
        nonlocal best, best_count
        if not paths:
            return
        paths = _improve_by_merge(list(paths))
        cnt = len(paths)
        if cnt < best_count:
            best_count = cnt
            best = [list(p) for p in paths]

    # Strategy 1: Gallai construction with several odd-pairings
    for trial in range(60):
        if time.time() - t0 > time_limit - 2.0:
            break
        rng = random.Random(90000 + trial * 7919)
        paths = _eulerian_decompose(edges, adj, rng, t0, time_limit - 0.5)
        consider(paths)
        if best_count <= 1:
            break

    # Strategy 2: Greedy restarts (different seeds)
    for trial in range(400):
        if time.time() - t0 > time_limit - 1.0:
            break
        seed = trial * 7919 + 13
        rng = random.Random(seed)
        paths = _greedy_path_cover(edges, adj, rng)
        consider(paths)
        if best_count <= 1:
            break

    # Strategy 3: Leaf-biased greedy restarts
    for trial in range(200):
        if time.time() - t0 > time_limit - 0.5:
            break
        seed = 100000 + trial * 6151
        rng = random.Random(seed)
        remaining = set(_edge_set(edges))
        deg = defaultdict(int)
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        paths = []
        while remaining:
            candidates = [v for v, d in deg.items() if d > 0]
            if not candidates:
                break
            leaves = [v for v in candidates if deg[v] == 1]
            odd = [v for v in candidates if deg[v] % 2 == 1]
            if leaves and rng.random() < 0.8:
                start = rng.choice(leaves)
            elif odd and rng.random() < 0.6:
                start = rng.choice(odd)
            else:
                min_d = min(deg[v] for v in candidates)
                low = [v for v in candidates if deg[v] == min_d]
                start = rng.choice(low)
            path = [start]
            in_path = {start}
            while True:
                v = path[-1]
                nbrs = [u for u in adj[v] if tuple(sorted((v, u))) in remaining]
                if not nbrs:
                    break
                unvis = [u for u in nbrs if u not in in_path]
                if unvis:
                    min_d = min(deg[u] for u in unvis)
                    best_n = [u for u in unvis if deg[u] == min_d]
                    nxt = rng.choice(best_n)
                else:
                    break
                e = tuple(sorted((v, nxt)))
                remaining.discard(e)
                deg[v] -= 1
                deg[nxt] -= 1
                if deg[v] <= 0:
                    del deg[v]
                if deg[nxt] <= 0:
                    del deg[nxt]
                path.append(nxt)
                in_path.add(nxt)
            if len(path) >= 2:
                paths.append(path)
        for e in remaining:
            paths.append([e[0], e[1]])
        consider(paths)
        if best_count <= 1:
            break

    if best is None:
        best = _greedy_path_cover(edges, adj, random.Random(42))
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 148.0
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    n_graphs = len(specs)

    for i, (n, p, seed) in enumerate(specs):
        remaining = deadline - time.time()
        if remaining < 3:
            edges = _instance(n, p, seed)
            adj = _build_adj(edges)
            results.append(_greedy_path_cover(edges, adj, random.Random(1)))
            continue
        alloc = max(2.0, (remaining - 1.0) / (n_graphs - i))
        edges = _instance(n, p, seed)
        paths = _solve_one(edges, time_limit=alloc)
        results.append(paths)

    return results


if __name__ == "__main__":
    res = construct()
    for i, paths in enumerate(res):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in res)}")
