"""Triangle-free diameter-four augmentation: greedy diameter reduction + aggressive local search."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _build_adj(n, base_edges, added):
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _diameter(adj, n):
    """Return graph diameter, or -1 if disconnected."""
    best = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u] + 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = du
                    q.append(v)
        if any(d < 0 for d in dist):
            return -1
        m = max(dist)
        if m > best:
            best = m
    return best


def _valid(added, n, base_edges, base_set):
    for u, v in added:
        if u == v:
            return False
        if u % 2 == v % 2:
            return False
        if (u, v) in base_set or (v, u) in base_set:
            return False
    adj = _build_adj(n, base_edges, added)
    d = _diameter(adj, n)
    return d >= 0 and d <= 4


def _fallback(n, base_set):
    edges = []
    for v in [1, 3, 5, 7, 9]:
        if (12, v) not in base_set and (v, 12) not in base_set:
            edges.append((12, v))
    for v in [2, 4, 6, 8, 10]:
        if (11, v) not in base_set and (v, 11) not in base_set:
            edges.append((11, v))
    return edges


def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    base_set = set()
    for u, v in base_edges:
        base_set.add((u, v))
        base_set.add((v, u))

    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]

    candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (u, v) not in base_set:
                candidates.append((u, v))

    def diam_of(added):
        adj = _build_adj(n, base_edges, added)
        return _diameter(adj, n)

    # Phase 1: Greedy diameter reduction from empty
    cur = []
    for _ in range(50):
        if time.time() > DEADLINE:
            break
        d = diam_of(cur)
        if d <= 4:
            break
        best_e = None
        best_d = d
        for e in candidates:
            if e in cur:
                continue
            trial = cur + [e]
            td = diam_of(trial)
            if td < 0:
                continue
            if td < best_d:
                best_d = td
                best_e = e
        if best_e is None:
            break
        cur.append(best_e)

    # Phase 2: Aggressive minimization - repeated single removals
    improved = True
    while improved:
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(len(cur)):
            if time.time() > DEADLINE:
                break
            trial = cur[:i] + cur[i + 1:]
            if _valid(trial, n, base_edges, base_set):
                cur = trial
                improved = True
                break

    # Phase 3: 1-for-1 swaps to enable further removals
    for _ in range(60):
        if time.time() > DEADLINE:
            break
        swapped = False
        for i in range(len(cur)):
            if time.time() > DEADLINE:
                break
            for e in candidates:
                if time.time() > DEADLINE:
                    break
                if e in cur:
                    continue
                trial = cur[:i] + [e] + cur[i + 1:]
                if _valid(trial, n, base_edges, base_set):
                    cur = trial
                    swapped = True
                    break
            if swapped:
                break
        if not swapped:
            break
        # After swap, try to remove again
        improved = True
        while improved:
            if time.time() > DEADLINE:
                break
            improved = False
            for i in range(len(cur)):
                if time.time() > DEADLINE:
                    break
                trial = cur[:i] + cur[i + 1:]
                if _valid(trial, n, base_edges, base_set):
                    cur = trial
                    improved = True
                    break

    # Phase 4: Try 2-for-1 moves (remove 2, add 1) to reduce count
    for _ in range(30):
        if time.time() > DEADLINE:
            break
        found = False
        for i in range(len(cur)):
            if time.time() > DEADLINE:
                break
            for j in range(i + 1, len(cur)):
                if time.time() > DEADLINE:
                    break
                for e in candidates:
                    if time.time() > DEADLINE:
                        break
                    if e in cur:
                        continue
                    trial = cur[:i] + cur[i + 1:j] + cur[j + 1:] + [e]
                    if len(trial) < len(cur) and _valid(trial, n, base_edges, base_set):
                        cur = trial
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
        # After 2-for-1, try single removals again
        improved = True
        while improved:
            if time.time() > DEADLINE:
                break
            improved = False
            for i in range(len(cur)):
                if time.time() > DEADLINE:
                    break
                trial = cur[:i] + cur[i + 1:]
                if _valid(trial, n, base_edges, base_set):
                    cur = trial
                    improved = True
                    break

    # Phase 5: Try 2-for-2 swaps (remove 2, add 2) to restructure
    for _ in range(20):
        if time.time() > DEADLINE:
            break
        found = False
        for i in range(len(cur)):
            if time.time() > DEADLINE:
                break
            for j in range(i + 1, len(cur)):
                if time.time() > DEADLINE:
                    break
                for e1 in candidates:
                    if time.time() > DEADLINE:
                        break
                    if e1 in cur:
                        continue
                    for e2 in candidates:
                        if time.time() > DEADLINE:
                            break
                        if e2 in cur or e2 == e1:
                            continue
                        trial = cur[:i] + cur[i + 1:j] + cur[j + 1:] + [e1, e2]
                        if _valid(trial, n, base_edges, base_set):
                            cur = trial
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
        # After 2-for-2, try to remove
        improved = True
        while improved:
            if time.time() > DEADLINE:
                break
            improved = False
            for i in range(len(cur)):
                if time.time() > DEADLINE:
                    break
                trial = cur[:i] + cur[i + 1:]
                if _valid(trial, n, base_edges, base_set):
                    cur = trial
                    improved = True
                    break

    # Guarantee validity
    if not _valid(cur, n, base_edges, base_set):
        cur = _fallback(n, base_set)
        improved = True
        while improved:
            if time.time() > DEADLINE:
                break
            improved = False
            for i in range(len(cur)):
                if time.time() > DEADLINE:
                    break
                trial = cur[:i] + cur[i + 1:]
                if _valid(trial, n, base_edges, base_set):
                    cur = trial
                    improved = True
                    break

    # Final verification
    if not _valid(cur, n, base_edges, base_set):
        cur = _fallback(n, base_set)
        if not _valid(cur, n, base_edges, base_set):
            cur = []
            for v in odd_verts:
                if (12, v) not in base_set:
                    cur.append((12, v))

    # Type-safe output
    result = []
    seen = set()
    base_keys = set()
    for a, b in base_set:
        if a < b:
            base_keys.add((a, b))
    for e in cur:
        u, v = e
        u = int(u)
        v = int(v)
        key = (min(u, v), max(u, v))
        if key in seen or key in base_keys:
            continue
        seen.add(key)
        result.append([u, v])
    return result
# EVOLVE-BLOCK-END
