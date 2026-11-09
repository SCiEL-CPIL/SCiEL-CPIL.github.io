import random
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def greedy_trails(edges, adj, max_time=30.0):
        """Greedy trail decomposition: repeatedly walk along unused edges."""
        remaining = set(edges)
        trails = []
        t0 = time.time()
        while remaining:
            if time.time() - t0 > max_time:
                # Emergency: make each remaining edge its own trail
                for e in remaining:
                    trails.append(list(e))
                break
            # Pick a starting edge: prefer high-degree endpoints to build long trails
            best_edge = None
            best_key = -1
            for e in remaining:
                u, v = e
                # count unused incident edges
                du = sum(1 for nb in adj[u] if tuple(sorted((u, nb))) in remaining)
                dv = sum(1 for nb in adj[v] if tuple(sorted((v, nb))) in remaining)
                k = du + dv
                if k > best_key:
                    best_key = k
                    best_edge = e
            if best_edge is None:
                break
            u, v = best_edge
            trail = [u, v]
            remaining.discard(best_edge)
            # Extend from v end
            while True:
                cur = trail[-1]
                cands = [nb for nb in adj[cur] if tuple(sorted((cur, nb))) in remaining]
                if not cands:
                    break
                # pick neighbor with most remaining options (longest trail)
                cands.sort(key=lambda nb: -sum(1 for x in adj[nb] if tuple(sorted((nb, x))) in remaining))
                nxt = cands[0]
                trail.append(nxt)
                remaining.discard(tuple(sorted((cur, nxt))))
            # Extend from u end
            while True:
                cur = trail[0]
                cands = [nb for nb in adj[cur] if tuple(sorted((cur, nb))) in remaining]
                if not cands:
                    break
                cands.sort(key=lambda nb: -sum(1 for x in adj[nb] if tuple(sorted((nb, x))) in remaining))
                nxt = cands[0]
                trail.insert(0, nxt)
                remaining.discard(tuple(sorted((cur, nxt))))
            trails.append(trail)
        return trails

    def split_trails_to_paths(trails):
        """Split each trail into simple paths at the first repeated vertex."""
        paths = []
        for trail in trails:
            # Find first repeated vertex
            seen = {}
            split_idx = None
            for i, vtx in enumerate(trail):
                if vtx in seen:
                    split_idx = i
                    break
                seen[vtx] = i
            if split_idx is None:
                # Already a simple path
                paths.append(trail)
            else:
                # trail[0..split_idx] and trail[split_idx..end]
                part1 = trail[:split_idx + 1]
                part2 = trail[split_idx:]
                # Recursively split each part (they may still have repeats)
                paths.extend(split_trails_to_paths([part1]))
                paths.extend(split_trails_to_paths([part2]))
        return paths

    def merge_paths(paths, edges_set, adj, max_time=30.0):
        """Local search: try to merge adjacent paths by finding a short connecting path."""
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < max_time:
            improved = False
            n = len(paths)
            for i in range(n):
                if time.time() - t0 > max_time:
                    break
                for j in range(n):
                    if i == j:
                        continue
                    a = paths[i]
                    b = paths[j]
                    if not a or not b:
                        continue
                    # Try to connect end of a to start of b (or other orientations)
                    candidates = [
                        (a[-1], b[0], list(reversed(a)) + list(b)),
                        (a[0], b[0], list(a) + list(reversed(b))),
                        (a[-1], b[-1], list(reversed(a)) + list(reversed(b))),
                        (a[0], b[-1], list(a) + list(b)),
                    ]
                    for start, end, merged in candidates:
                        # Check simple
                        if len(set(merged)) != len(merged):
                            continue
                        # Check all edges exist in graph
                        ok = True
                        for k in range(len(merged) - 1):
                            e = tuple(sorted((merged[k], merged[k + 1])))
                            if e not in edges_set:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Check edge-disjoint: merged edges must be subset of a's + b's edges
                        a_edges = set()
                        for k in range(len(a) - 1):
                            a_edges.add(tuple(sorted((a[k], a[k + 1]))))
                        b_edges = set()
                        for k in range(len(b) - 1):
                            b_edges.add(tuple(sorted((b[k], b[k + 1]))))
                        m_edges = set()
                        for k in range(len(merged) - 1):
                            m_edges.add(tuple(sorted((merged[k], merged[k + 1]))))
                        if m_edges == (a_edges | b_edges):
                            # Valid merge
                            paths[i] = merged
                            paths[j] = []
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # Remove empty paths
            paths = [p for p in paths if p]
        return paths

    def decompose(edges):
        if not edges:
            return []
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        edges_set = set(edges)

        # Phase 1: greedy trail decomposition
        trails = greedy_trails(edges, adj, max_time=25.0)

        # Phase 2: split trails into simple paths
        paths = split_trails_to_paths(trails)

        # Phase 3: local search merge
        paths = merge_paths(paths, edges_set, adj, max_time=35.0)

        # Verify coverage; if any edge missing, add as single-edge paths
        covered = set()
        for p in paths:
            for k in range(len(p) - 1):
                covered.add(tuple(sorted((p[k], p[k + 1]))))
        for e in edges_set:
            if e not in covered:
                paths.append(list(e))

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > time_limit - 10:
            # Not enough time; return empty for remaining (will be invalid but avoids timeout)
            results.append([])
            continue
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
