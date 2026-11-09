"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = 24

# Precompute path edges as a set of (min, max) tuples
PATH_SET = set()
for _u, _v in P['edges']:
    PATH_SET.add((min(_u, _v), max(_u, _v)))

# All candidate edges not in the path
ALL_NONPATH = []
for _u in range(N):
    for _v in range(_u + 1, N):
        if (_u, _v) not in PATH_SET:
            ALL_NONPATH.append((_u, _v))


def _build_adj(added):
    adj = [[] for _ in range(N)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_diameter(adj):
    """Return graph diameter, or -1 if disconnected. Early-stop at dist > 4."""
    maxd = 0
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du > 4:
                return -1  # exceeds budget
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if dist[v] > maxd:
                        maxd = dist[v]
                    q.append(v)
        if -1 in dist:
            return -1  # disconnected
    return maxd


def _quick_valid(added):
    """Fast validity: no dup/path edges, triangle-free, diameter <= 4."""
    es = set(PATH_SET)
    for u, v in added:
        k = (min(u, v), max(u, v))
        if k in es:
            return False
        es.add(k)
    adj = _build_adj(added)
    adj_set = [set(a) for a in adj]
    # triangle-free check
    for u in range(N):
        au = adj[u]
        for i in range(len(au)):
            v = au[i]
            for w in au[i + 1:]:
                if w in adj_set[v]:
                    return False
    # diameter check with early stop
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du > 4:
                return False
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if dist[v] > 4:
                        return False
                    q.append(v)
        if -1 in dist:
            return False
    return True


def _initial_solution():
    # Guaranteed valid 10-edge star at vertex 12 to odd vertices on each side
    return [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
            [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]


def _greedy_removal(start_edges, max_time=5.0):
    """Repeatedly try removing any edge that keeps validity."""
    start_time = time.time()
    cur = [list(e) for e in start_edges]
    if not _quick_valid(cur):
        return start_edges
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for i in range(len(cur)):
            if time.time() - start_time > max_time:
                break
            trial = cur[:i] + cur[i + 1:]
            if _quick_valid(trial):
                cur = trial
                improved = True
                break
    return cur


def _swap_polish(added_edges, max_time=5.0):
    """1-for-1 swap polish: try replacing one edge with another to keep validity
    and (after removal) reduce count. Focus on swaps that enable a removal."""
    start_time = time.time()
    cur = [list(e) for e in added_edges]
    best = cur
    best_len = len(cur)
    if not _quick_valid(cur):
        return added_edges

    while time.time() - start_time < max_time:
        improved = False
        # Try 1-for-1 swaps
        for i in range(len(cur)):
            if time.time() - start_time > max_time:
                break
            u, v = cur[i]
            for (nu, nv) in ALL_NONPATH:
                if (nu, nv) == (u, v) or (nu, nv) == (v, u):
                    continue
                # skip if already present
                present = False
                for j, e in enumerate(cur):
                    if j == i:
                        continue
                    if (e[0] == nu and e[1] == nv) or (e[0] == nv and e[1] == nu):
                        present = True
                        break
                if present:
                    continue
                trial = cur[:i] + [[nu, nv]] + cur[i + 1:]
                if not _quick_valid(trial):
                    continue
                # try to remove an edge from the swapped solution
                for k in range(len(trial)):
                    if time.time() - start_time > max_time:
                        break
                    t2 = trial[:k] + trial[k + 1:]
                    if _quick_valid(t2):
                        cur = t2
                        if len(cur) < best_len:
                            best = [list(e) for e in cur]
                            best_len = len(cur)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # try pure removals once more
            for i in range(len(cur)):
                if time.time() - start_time > max_time:
                    break
                trial = cur[:i] + cur[i + 1:]
                if _quick_valid(trial):
                    cur = trial
                    if len(cur) < best_len:
                        best = [list(e) for e in cur]
                        best_len = len(cur)
                    improved = True
                    break
            if not improved:
                break
    return best


def _random_restart_greedy(restarts=30, max_time=30.0):
    """Randomized greedy: shuffle candidate edges, add while valid, then trim."""
    start_time = time.time()
    rng = random.Random(7777)
    best = None
    best_len = 999
    pool = list(ALL_NONPATH)

    for _ in range(restarts):
        if time.time() - start_time > max_time:
            break
        cur = []
        cur_set = set()
        shuffled = list(pool)
        rng.shuffle(shuffled)
        # Add edges greedily in shuffled order while valid, cap size
        for (u, v) in shuffled:
            if time.time() - start_time > max_time:
                break
            if (u, v) in cur_set:
                continue
            trial = cur + [[u, v]]
            if _quick_valid(trial):
                cur = trial
                cur_set.add((u, v))
                if len(cur) >= 14:
                    break
        # Trim
        remaining = max_time - (time.time() - start_time)
        if remaining > 1.0:
            cur = _greedy_removal(cur, max_time=min(3.0, remaining * 0.5))
        if _quick_valid(cur) and len(cur) < best_len:
            best = [list(e) for e in cur]
            best_len = len(cur)
            if best_len <= 6:
                break
    return best


def _hand_crafted_candidates():
    """Return a list of hand-crafted candidate edge sets to try polishing from."""
    cands = []
    # 10-edge star (guaranteed)
    cands.append(_initial_solution())
    # Two-hub 7-edge: hub 6 and 17
    cands.append([[6, 17], [6, 0], [6, 11], [6, 13], [6, 15], [17, 2], [17, 23]])
    # Two-hub 8-edge: hub 12 and 18
    cands.append([[12, 18], [12, 0], [12, 2], [12, 4], [12, 6], [12, 8], [12, 10], [18, 23]])
    # Asymmetric 8-edge: hub 12 with 6 + hub 18 with 1
    cands.append([[12, 18], [12, 1], [12, 3], [12, 5], [12, 7], [12, 9], [12, 11], [18, 23]])
    # Asymmetric 7-edge: hub 12 with 5 + hub 17 with 1
    cands.append([[12, 17], [12, 1], [12, 3], [12, 5], [12, 7], [12, 9], [17, 23]])
    return cands


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Guaranteed valid fallback
    fallback = _initial_solution()
    if not _quick_valid(fallback):
        # Should never happen, but keep a minimal valid set
        fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                    [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]

    best = fallback
    best_len = len(fallback)

    # 1) Polish hand-crafted candidates (fast, high-quality starting points)
    for cand in _hand_crafted_candidates():
        if time.time() > deadline:
            break
        if not _quick_valid(cand):
            continue
        if len(cand) < best_len:
            best = cand
            best_len = len(cand)
        # Polish from this candidate
        remaining = deadline - time.time()
        if remaining > 2.0:
            polished = _swap_polish(cand, max_time=min(4.0, remaining * 0.4))
            if _quick_valid(polished) and len(polished) < best_len:
                best = polished
                best_len = len(polished)

    # 2) Randomized restart greedy for diverse exploration
    remaining = deadline - time.time()
    if remaining > 5.0 and best_len > 6:
        rr = _random_restart_greedy(restarts=20, max_time=min(25.0, remaining * 0.6))
        if rr is not None and _quick_valid(rr) and len(rr) < best_len:
            best = rr
            best_len = len(rr)

    # 3) Final polish on the best solution found
    remaining = deadline - time.time()
    if remaining > 3.0 and best_len > 6:
        final = _swap_polish(best, max_time=min(8.0, remaining * 0.7))
        if _quick_valid(final) and len(final) < best_len:
            best = final
            best_len = len(final)

    # Ensure return type is correct: list of [int, int]
    result = []
    for e in best:
        result.append([int(e[0]), int(e[1])])

    # Final sanity: if somehow invalid, return fallback
    if not _quick_valid(result):
        result = []
        for e in fallback:
            result.append([int(e[0]), int(e[1])])

    return result
# EVOLVE-BLOCK-END
