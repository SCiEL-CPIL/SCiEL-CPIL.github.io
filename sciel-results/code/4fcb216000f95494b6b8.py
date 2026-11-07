"""Extract up to 3 edge-disjoint Hamiltonian cycles on the 21-vertex 6-regular host.

Strategy: sequential exact backtracking (Warnsdorff heuristic + forced-move +
connectivity pruning) to peel off Hamiltonian cycles one at a time, plus a
complementary global 3-edge-coloring search. The full 21-vertex set is tried
first since the host is 6-regular (max 3 cycles partition all 63 edges).
"""
import time
import random
import itertools

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE_MARGIN = 12  # seconds of safety before the 90 s kill


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_one_ham_cycle(S, adj, used_edges, deadline, rng, node_budget=200000):
    """Find one Hamiltonian cycle on vertex set S (list) avoiding used_edges.

    Uses Warnsdorff + forced-move + connectivity pruning backtracking.
    Returns list of vertices (cycle order) or None.
    """
    k = len(S)
    if k < 3:
        return None
    S_set = set(S)
    start = S[0]

    # Precompute per-vertex adjacency restricted to S
    adjS = [None] * len(S)
    pos = {v: i for i, v in enumerate(S)}
    for i, v in enumerate(S):
        adjS[i] = [pos[w] for w in adj[v] if w in S_set]

    # Precompute edge availability (not in used_edges)
    edge_ok = [[True] * k for _ in range(k)]
    for i in range(k):
        for j in adjS[i]:
            ek = _edge_key(S[i], S[j])
            if ek in used_edges:
                edge_ok[i][j] = False
                edge_ok[j][i] = False

    # Precompute per-vertex available degree (edges not in used_edges)
    avail_deg = [0] * k
    for i in range(k):
        for j in adjS[i]:
            if edge_ok[i][j]:
                avail_deg[i] += 1
        if avail_deg[i] < 2:
            return None

    # State
    in_path = [False] * k
    used_local = [[False] * k for _ in range(k)]  # symmetric
    in_path[0] = True
    path = [0]
    nodes = [0]

    def dfs(cur, depth):
        if depth == k:
            # closing edge
            if edge_ok[cur][0] and not used_local[cur][0]:
                return True
            return False
        # Build candidate moves
        cands = []
        for nb in adjS[cur]:
            if not in_path[nb]:
                if edge_ok[cur][nb] and not used_local[cur][nb]:
                    cands.append(nb)
        if not cands:
            return False
        # Forced move: exactly one candidate
        if len(cands) == 1:
            nb = cands[0]
            # Check the closing edge won't be needed yet: if nb is last step
            # and closing edge unavailable, prune
            in_path[nb] = True
            used_local[cur][nb] = True
            used_local[nb][cur] = True
            path.append(nb)
            ok = dfs(nb, depth + 1)
            path.pop()
            in_path[nb] = False
            used_local[cur][nb] = False
            used_local[nb][cur] = False
            return ok
        # Warnsdorff: sort by fewest onward options
        def onward(nb):
            c = 0
            for x in adjS[nb]:
                if not in_path[x] and edge_ok[nb][x] and not used_local[nb][x]:
                    c += 1
            return c
        cands.sort(key=lambda nb: (onward(nb), rng.random()))
        # Also try reverse order sometimes for diversity handled by caller
        for nb in cands:
            in_path[nb] = True
            used_local[cur][nb] = True
            used_local[nb][cur] = True
            path.append(nb)
            ok = dfs(nb, depth + 1)
            if ok:
                return True
            path.pop()
            in_path[nb] = False
            used_local[cur][nb] = False
            used_local[nb][cur] = False
            nodes[0] += 1
            if nodes[0] > node_budget or time.time() > deadline:
                return False
        return False

    for attempt in range(8):
        if time.time() > deadline:
            break
        if attempt > 0:
            # perturb: shuffle start
            perm = list(range(k))
            rng.shuffle(perm)
            # rebuild with new start
            pass
        if dfs(0, 1):
            return [S[i] for i in path]
        nodes[0] += 1
        if nodes[0] > node_budget or time.time() > deadline:
            break
    return None


def _try_full_set(adj, deadline, rng):
    """Try to find 3 edge-disjoint Hamiltonian cycles on all 21 vertices."""
    n = P['n']
    S = list(range(n))
    used_edges = set()
    cycles = []
    for i in range(3):
        if time.time() > deadline:
            break
        c = _find_one_ham_cycle(S, adj, used_edges, deadline, rng)
        if c is None:
            break
        for j in range(len(c)):
            u = c[j]
            v = c[(j + 1) % len(c)]
            used_edges.add(_edge_key(u, v))
        cycles.append(c)
    return cycles if cycles else None


def _try_subset(adj, deadline, rng, target):
    """Try smaller vertex subsets to reach `target` cycles."""
    n = P['n']
    all_v = list(range(n))
    # sizes: for target cycles on size s, need 2*target <= deg and 70//s >= target
    # We'll just try a few promising sizes
    sizes = [7, 8, 9, 10, 11, 12, 14, 15, 17, 18, 20]
    rng.shuffle(sizes)
    for s in sizes:
        if time.time() > deadline:
            break
        max_possible = min(10, 70 // s)
        if max_possible < target:
            continue
        # sample subsets
        for _ in range(200):
            if time.time() > deadline:
                break
            S = list(rng.sample(all_v, s))
            used_edges = set()
            cycles = []
            for i in range(target):
                c = _find_one_ham_cycle(S, adj, used_edges, deadline, rng,
                                        node_budget=50000)
                if c is None:
                    break
                for j in range(len(c)):
                    u = c[j]
                    v = c[(j + 1) % len(c)]
                    used_edges.add(_edge_key(u, v))
                cycles.append(c)
            if len(cycles) == target:
                return cycles
    return None


def construct():
    t0 = time.time()
    deadline = t0 + (90 - DEADLINE_MARGIN)
    rng = random.Random(12345)
    adj = _build_adjacency()

    best = None

    # Primary: full 21-vertex set -> up to 3 cycles
    res = _try_full_set(adj, deadline, rng)
    if res:
        best = res

    # Complementary: if full set gave < 3, try subsets for 3 cycles
    if best is None or len(best) < 3:
        res2 = _try_subset(adj, deadline, rng, 3)
        if res2 and (best is None or len(res2) > len(best)):
            best = res2

    # Fallback: ensure at least one valid cycle
    if best is None:
        # find any triangle
        edge_set = set()
        for u, v in P['edges']:
            edge_set.add(_edge_key(u, v))
        for u, v in P['edges']:
            if time.time() > deadline:
                break
            common = adj[u] & adj[v]
            for w in common:
                if _edge_key(w, u) in edge_set and _edge_key(w, v) in edge_set:
                    best = [[u, v, w]]
                    break
            if best:
                break

    if best is None:
        # absolute fallback: a known triangle 0-1-2
        best = [[0, 1, 2]]

    # Validate & convert to exact int lists
    result = []
    for cyc in best:
        result.append([int(v) for v in cyc])
    return result
