"""Decompose each fixed instance graph into cycles + leftover edges.
Strategy: repeatedly find long cycles using randomized DFS with restarts,
then fall back to shorter cycles, then edges."""
import time
import sys
from random import Random

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycle_greedy(adj, n, rng, max_attempts=30, min_len=3):
    """Find a cycle of at least min_len vertices using randomized greedy extension."""
    best = None
    best_len = 0

    for _ in range(max_attempts):
        # Pick a random start vertex with degree >= 2
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            break
        start = rng.choice(candidates)

        # Greedy path extension: prefer neighbors that keep options open
        path = [start]
        visited = {start}
        cur = start

        while True:
            nbrs = [nb for nb in adj[cur] if nb not in visited]
            if not nbrs:
                break
            # Prefer low-degree neighbors (to avoid trapping ourselves)
            nbrs.sort(key=lambda x: len(adj[x]))
            # But with some randomness
            if rng.random() < 0.3:
                nxt = rng.choice(nbrs)
            else:
                nxt = nbrs[0]
            path.append(nxt)
            visited.add(nxt)
            cur = nxt

        # Check if we can close the cycle
        if len(path) >= min_len and adj[path[-1]] and start in adj[path[-1]]:
            if len(path) > best_len:
                best = path[:]
                best_len = len(path)

    return best


def _find_cycle_dfs(adj, n, target_len, rng, max_nodes=50000):
    """Try to find a cycle of exactly target_len vertices using randomized DFS."""
    if target_len < 3:
        return None

    nodes_visited = 0
    candidates = [v for v in range(n) if len(adj[v]) >= 2]
    if not candidates:
        return None

    # Shuffle candidates for randomness
    rng.shuffle(candidates)

    for start in candidates:
        if nodes_visited > max_nodes:
            break
        # DFS from start, looking for a path of target_len vertices ending at a neighbor of start
        stack = [(start, [start])]
        visited_stack = {start}

        while stack:
            cur, path = stack.pop()
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return None

            if len(path) == target_len:
                if start in adj[cur]:
                    return path[:]
                continue

            nbrs = [nb for nb in adj[cur] if nb not in visited_stack]
            # Shuffle for randomness
            rng.shuffle(nbrs)
            for nb in nbrs:
                new_path = path + [nb]
                new_visited = visited_stack | {nb}
                # Prune: if nb == start and len(new_path) < target_len, skip (can't close early)
                if nb == start:
                    continue
                stack.append((nb, new_path))

    return None


def _decompose(n, edges, time_budget, global_deadline):
    adj = _build_adjacency(n, edges)
    remaining = set(edges)
    pieces = []
    rng = Random(42)
    t0 = time.time()

    # Phase 1: Greedy long cycles
    max_cycle_len = min(n, 15)
    while remaining:
        if time.time() > global_deadline or time.time() - t0 > time_budget:
            break

        # Try to find a cycle of decreasing length
        found = None
        for target in range(max_cycle_len, 2, -1):
            if time.time() > global_deadline or time.time() - t0 > time_budget:
                break
            # Use greedy first (fast), then DFS if greedy fails
            found = _find_cycle_greedy(adj, n, rng, max_attempts=20, min_len=target)
            if found is not None:
                break
            # Try DFS for shorter targets
            if target <= 10:
                found = _find_cycle_dfs(adj, n, target, rng, max_nodes=30000)
                if found is not None:
                    break

        if found is None:
            break

        # Verify cycle edges are in remaining
        ok = True
        for i in range(len(found)):
            a = found[i]
            b = found[(i + 1) % len(found)]
            e = (a, b) if a < b else (b, a)
            if e not in remaining:
                ok = False
                break
        if not ok:
            break

        # Commit cycle
        for i in range(len(found)):
            a = found[i]
            b = found[(i + 1) % len(found)]
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        pieces.append(found)

    # Leftover edges
    for e in sorted(remaining):
        pieces.append(list(e))
    return pieces


def _validate(n, edges, pieces):
    seen = set()
    for piece in pieces:
        if len(piece) == 2:
            u, v = piece
            e = (u, v) if u < v else (v, u)
            if e in seen:
                return False
            seen.add(e)
        elif len(piece) >= 3:
            verts = piece
            if len(set(verts)) != len(verts):
                return False
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                e = (a, b) if a < b else (b, a)
                if e in seen:
                    return False
                seen.add(e)
        else:
            return False
    if seen != set(edges):
        return False
    return True


def construct():
    global_deadline = time.time() + 140.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        elapsed = time.time() - (global_deadline - 140.0)
        remaining_time = 140.0 - elapsed - 5.0
        budget = max(1.0, min(40.0, remaining_time))
        try:
            pieces = _decompose(n, edges, budget, global_deadline)
            if not _validate(n, edges, pieces):
                pieces = [list(e) for e in edges]
        except Exception:
            pieces = [list(e) for e in edges]
        results.append(pieces)
    return results
