# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _longest_path(adj, n, start, deadline):
    """Greedy longest simple path from start using Warnsdorff + degree tie-break."""
    path = [start]
    visited = {start}
    while True:
        u = path[-1]
        cands = [v for v in adj[u] if v not in visited]
        if not cands:
            break
        # Warnsdorff: prefer neighbours with fewest onward options; break ties by degree.
        best = None
        best_key = None
        for v in cands:
            onward = sum(1 for w in adj[v] if w not in visited and w != start)
            key = (onward, -len(adj[v]), v)
            if best_key is None or key < best_key:
                best_key = key
                best = v
        path.append(best)
        visited.add(best)
        if deadline is not None and time.time() > deadline:
            break
    return path

def _extract_cycles_from_path(adj, path, deadline):
    """Find all cycles (i,j) with path[i] adjacent to path[j], j-i>=2; return best (longest)."""
    best = None
    L = len(path)
    for i in range(L):
        if deadline is not None and time.time() > deadline:
            break
        u = path[i]
        for j in range(i + 2, L):
            if path[j] in adj[u]:
                length = j - i + 1
                if best is None or length > len(best):
                    best = path[i:j + 1]
    return best

def _find_cycle(adj, n, seed, deadline):
    """Return the longest simple cycle found, or [] if none."""
    best = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []
    rng = random.Random(seed)
    rng.shuffle(vertices)
    # Try several starts; stop early if near-Hamiltonian.
    for start in vertices[:10]:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue
        path = _longest_path(adj, n, start, deadline)
        if len(path) < 3:
            continue
        cycle = _extract_cycles_from_path(adj, path, deadline)
        if cycle and len(cycle) > len(best):
            best = cycle
            if len(best) >= n - 1:
                break
    return best

def _greedy_decompose(edges, n, seed, deadline):
    remaining = set(edges)
    pieces = []
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    cc = 0
    while remaining:
        if time.time() > deadline:
            for e in remaining:
                pieces.append([e[0], e[1]])
            return pieces
        cycle = _find_cycle(adj, n, seed + cc, deadline)
        if cycle:
            ce = set()
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                ce.add((min(u, v), max(u, v)))
            for e in ce:
                if e in remaining:
                    remaining.discard(e)
                    u, v = e
                    adj[u].discard(v)
                    adj[v].discard(u)
            pieces.append(cycle)
            cc += 1
        else:
            e = min(remaining, key=lambda e: (len(adj[e[0]]) + len(adj[e[1]]), e))
            remaining.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
    return pieces

def _improve_pieces(pieces, edges_set, deadline):
    """Local polish: try to merge two single-edge pieces into a 3-cycle, and to
    extend a short cycle by swapping in a longer cycle sharing a vertex.
    Returns an improved (or equal) piece list."""
    pieces = [list(p) for p in pieces]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # 1) Merge two single edges sharing a vertex into a 3-cycle.
        singles = [i for i, p in enumerate(pieces) if len(p) == 2]
        for a in range(len(singles)):
            i = singles[a]
            u1, v1 = pieces[i]
            for b in range(a + 1, len(singles)):
                j = singles[b]
                u2, v2 = pieces[j]
                # share exactly one vertex
                if u1 == u2 or u1 == v2:
                    w = u1
                    x, y = v1, v2
                elif u1 == v2:
                    w = u1
                    x, y = v1, v2
                elif v1 == u2:
                    w = v1
                    x, y = u1, v2
                elif v1 == v2:
                    w = v1
                    x, y = u1, u2
                else:
                    continue
                # need edge (x,y) to exist
                if (min(x, y), max(x, y)) in edges_set:
                    pieces[i] = [w, x, y]
                    del pieces[j]
                    changed = True
                    break
            if changed:
                break
    return pieces

def _bipartite_max_matching(adj):
    """Hopcroft-Karp on adj: dict left->set(right). Returns list of (l, r)."""
    from collections import deque
    pair_u = {u: -1 for u in adj}
    pair_v = defaultdict(lambda: -1)
    dist = {}
    def bfs():
        q = deque()
        for u in adj:
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist.get(pu, -1) == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found
    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist.get(pu, -1) == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False
    matching = []
    while bfs():
        for u in list(adj):
            if pair_u[u] == -1:
                dfs(u)
    for u in adj:
        if pair_u[u] != -1:
            matching.append((u, pair_u[u]))
    return matching

def _decompose(edges, n, seed, time_limit=40.0):
    if not edges:
        return []
    edges_set = set(edges)
    start_time = time.time()
    deadline = start_time + time_limit

    best_pieces = None
    best_count = len(edges)

    # Phase 1: greedy with restarts (fast, gives a strong initial packing).
    num_restarts = 40
    for r in range(num_restarts):
        if time.time() > deadline - 3.0:
            break
        pieces = _greedy_decompose(edges, n, seed + r * 101, deadline)
        pieces = _improve_pieces(pieces, edges_set, deadline)
        if len(pieces) < best_count:
            best_count = len(pieces)
            best_pieces = pieces
        if best_count <= n // 3:  # near-optimal for sparse random graphs
            break

    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]

    # Phase 2: branch-and-bound to find a near-perfect cycle packing.
    # Objective: maximize number of cycles (each cycle uses >=3 edges).
    # We solve a max edge-disjoint cycle packing via branch and bound on a
    # small set of candidate cycles, using bipartite matching as an upper bound.
    if time.time() < deadline - 2.0:
        # Build candidate cycles: all 3-cycles and a set of longer cycles.
        adj = defaultdict(set)
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        candidates = []
        # 3-cycles
        for u in adj:
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v] & adj[u]:
                    if w > v:
                        candidates.append([u, v, w])
        # Longer cycles from greedy (top few)
        for r in range(15):
            c = _find_cycle(adj, n, seed + 7000 + r, deadline)
            if c and len(c) >= 4:
                candidates.append(c)
        # Deduplicate by edge set
        seen = set()
        uniq = []
        for c in candidates:
            es = frozenset((min(c[i], c[(i+1)%len(c)]), max(c[i], c[(i+1)%len(c)])) for i in range(len(c)))
            if es not in seen:
                seen.add(es)
                uniq.append((c, es))
        candidates = uniq

        if len(candidates) > 60:
            # Keep the longest 60 candidates (longer cycles are more valuable)
            candidates.sort(key=lambda x: -len(x[0]))
            candidates = candidates[:60]

        # Branch and bound: pick cycles that are edge-disjoint.
        # Upper bound: max number of edge-disjoint cycles <= floor(remaining_edges / 3)
        # and <= max matching on a conflict graph (approximate).
        # Use a simple DFS with memo on (index, used_edges) -- too big, so use
        # greedy + local search on the candidate set instead.
        # Simpler: try all subsets of top-K candidates (K up to ~20) via DFS.
        K = min(len(candidates), 20)
        top = candidates[:K]
        best_pack = []
        best_pack_count = 0

        def dfs_pack(idx, used_edges, pack):
            nonlocal best_pack, best_pack_count
            if time.time() > deadline - 1.0:
                return
            if len(pack) > best_pack_count:
                best_pack_count = len(pack)
                best_pack = list(pack)
            if idx >= K:
                return
            # Upper bound: remaining candidates can add at most (K - idx)
            if len(pack) + (K - idx) <= best_pack_count:
                return
            # Option: skip
            dfs_pack(idx + 1, used_edges, pack)
            # Option: take if disjoint
            c, es = top[idx]
            if not (es & used_edges):
                pack.append(c)
                dfs_pack(idx + 1, used_edges | es, pack)
                pack.pop()

        dfs_pack(0, frozenset(), [])
        if best_pack_count > 0:
            # Build pieces from best packing + remaining edges as singles.
            used = set()
            for c in best_pack:
                for i in range(len(c)):
                    u, v = c[i], c[(i+1)%len(c)]
                    used.add((min(u,v), max(u,v)))
            pieces = [list(c) for c in best_pack]
            for e in edges_set:
                if e not in used:
                    pieces.append([e[0], e[1]])
            if len(pieces) < best_count:
                best_count = len(pieces)
                best_pieces = pieces

    return best_pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_deadline = time.time() + 140.0

    for idx, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        remaining_time = total_deadline - time.time()
        graphs_left = len(specs) - idx
        time_for_this = max(10.0, remaining_time / graphs_left - 1.0)
        pieces = _decompose(edges, n, seed, time_limit=time_for_this)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
