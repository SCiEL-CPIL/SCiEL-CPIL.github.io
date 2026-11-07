"""Rewritten program: find up to 3 edge-disjoint Hamiltonian cycles on the 21-vertex host."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
DEADLINE = time.time() + 50  # safety margin before 60s/90s kill

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def _build_adjacency(edges):
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _find_hamiltonian_cycle(adj, start=0, time_limit=None):
    """Backtracking search for a Hamiltonian cycle starting/ending at `start`.
    Uses degree pruning: a vertex not yet visited with only one available
    neighbor (other than the current) must be visited next if that neighbor is
    its only option; also if current vertex has only one unvisited neighbor,
    it is forced."""
    n = N
    visited = [False] * n
    visited[start] = True
    path = [start]
    remaining = set(range(n))
    remaining.discard(start)

    # Precompute neighbor sets
    adj_list = {v: set(adj[v]) for v in range(n)}

    # Count of remaining (unvisited) neighbors for each vertex, dynamically.
    # We'll maintain unvisited neighbor count.
    unvisited_nbr = {}
    for v in range(n):
        unvisited_nbr[v] = len(adj_list[v] & remaining) if v != start else len(adj_list[v] & remaining)

    def dfs(cur):
        if time_limit is not None and time.time() > time_limit:
            return None
        if len(path) == n:
            # check closing edge
            if start in adj_list[cur]:
                return list(path)
            return None
        # gather unvisited neighbors of cur
        nbrs = [v for v in adj_list[cur] if not visited[v]]
        if not nbrs:
            return None
        # forced moves
        # 1) current vertex: if only one unvisited neighbor, must go there
        # 2) any unvisited vertex (not the last one we'll close) with exactly one
        #    unvisited neighbor -> that neighbor must be visited next (unless it is cur)
        # Prune: if any unvisited vertex v (v != cur) has unvisited_nbr[v] == 1,
        # then its only unvisited neighbor must be visited next. If that neighbor
        # is not cur, and cur has other options, we may still be okay, but if cur
        # is not that neighbor we must visit it now.
        forced = None
        for v in remaining:
            if v == cur:
                continue
            cnt = unvisited_nbr[v]
            if cnt == 1:
                # v's only unvisited neighbor
                only = [u for u in adj_list[v] if not visited[u]][0]
                if only != cur:
                    # we must visit `only` next, but cur is not `only` -> impossible
                    # unless only == cur handled above. So prune.
                    return None
                # else only == cur, fine
        if len(nbrs) == 1:
            forced = nbrs[0]
        else:
            # check if some v forces us to go to a specific neighbor of cur
            # i.e., v has unvisited_nbr[v]==1 and its only unvisited neighbor is in nbrs
            # then we must pick that one.
            for v in remaining:
                if v == cur:
                    continue
                if unvisited_nbr[v] == 1:
                    only = [u for u in adj_list[v] if not visited[u]][0]
                    if only in nbrs:
                        if forced is not None and forced != only:
                            return None
                        forced = only
                        break
        if forced is not None:
            nbrs = [forced]
        for nxt in nbrs:
            # try
            visited[nxt] = True
            remaining.discard(nxt)
            path.append(nxt)
            # update unvisited_nbr for neighbors of nxt
            changed = []
            for u in adj_list[nxt]:
                if not visited[u] and u != start:
                    unvisited_nbr[u] -= 1
                    changed.append(u)
            res = dfs(nxt)
            if res is not None:
                return res
            # undo
            path.pop()
            remaining.add(nxt)
            visited[nxt] = False
            for u in changed:
                unvisited_nbr[u] += 1
        return None

    return dfs(start)


def _cycle_edges(cycle):
    edges = []
    n = len(cycle)
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        edges.append(_edge_key(a, b))
    return edges


def _try_decompose(seed_order, rng, time_limit):
    """Try to find up to 3 edge-disjoint Hamiltonian cycles.
    seed_order: list of edges in original order for adjacency building order.
    Returns list of cycles (each a list of vertices) or None if fails to get >=1."""
    # build full edge set
    all_edges = set()
    for a, b in P['edges']:
        all_edges.add(_edge_key(a, b))
    available = set(all_edges)
    cycles = []
    for _ in range(3):
        if time.time() > time_limit:
            break
        # build adjacency from available edges
        adj = defaultdict(set)
        for (a, b) in available:
            adj[a].add(b)
            adj[b].add(a)
        # try to find a Hamiltonian cycle
        # try a few random starts? keep start=0 for determinism but vary neighbor order
        found = None
        # We'll do a randomized backtracking by shuffling neighbor order each attempt.
        # Implement a local randomized search by repeatedly calling with shuffled adj order.
        attempts = 0
        while found is None and attempts < 40 and time.time() < time_limit:
            attempts += 1
            # shuffle adjacency lists to vary the search
            adj_shuf = {v: list(adj[v]) for v in adj}
            for v in adj_shuf:
                rng.shuffle(adj_shuf[v])
            # run dfs with shuffled order
            found = _find_hamiltonian_cycle_shuf(adj_shuf, start=0, time_limit=time_limit)
            if found is None:
                # try a different start vertex occasionally
                if attempts % 5 == 0:
                    s = rng.randrange(N)
                    found = _find_hamiltonian_cycle_shuf(adj_shuf, start=s, time_limit=time_limit)
        if found is None:
            break
        # remove its edges
        ce = set(_cycle_edges(found))
        if not ce.issubset(available):
            break
        available -= ce
        cycles.append(found)
    return cycles if cycles else None


def _find_hamiltonian_cycle_shuf(adj_shuf, start=0, time_limit=None):
    """Randomized backtracking using pre-shuffled neighbor lists."""
    n = N
    adj_list = {v: list(adj_shuf[v]) for v in adj_shuf}
    visited = [False] * n
    visited[start] = True
    path = [start]
    remaining = set(range(n))
    remaining.discard(start)
    unvisited_nbr = {v: len(adj_list[v] & remaining) for v in range(n)}

    def dfs(cur):
        if time_limit is not None and time.time() > time_limit:
            return None
        if len(path) == n:
            if start in adj_list[cur]:
                return list(path)
            return None
        nbrs = [v for v in adj_list[cur] if not visited[v]]
        if not nbrs:
            return None
        # pruning: any unvisited v (v!=cur) with unvisited_nbr[v]==1 forces its only nbr
        for v in remaining:
            if v == cur:
                continue
            if unvisited_nbr[v] == 1:
                only = [u for u in adj_list[v] if not visited[u]][0]
                if only != cur:
                    return None
        if len(nbrs) == 1:
            nbrs = nbrs
        else:
            forced = None
            for v in remaining:
                if v == cur:
                    continue
                if unvisited_nbr[v] == 1:
                    only = [u for u in adj_list[v] if not visited[u]][0]
                    if only in nbrs:
                        forced = only
                        break
            if forced is not None:
                nbrs = [forced]
        for nxt in nbrs:
            visited[nxt] = True
            remaining.discard(nxt)
            path.append(nxt)
            changed = []
            for u in adj_list[nxt]:
                if not visited[u] and u != start:
                    unvisited_nbr[u] -= 1
                    changed.append(u)
            res = dfs(nxt)
            if res is not None:
                return res
            path.pop()
            remaining.add(nxt)
            visited[nxt] = False
            for u in changed:
                unvisited_nbr[u] += 1
        return None

    return dfs(start)


def construct():
    rng = random.Random(12345)
    best = None
    # First try deterministic natural order (start=0)
    natural = _try_decompose(P['edges'], rng, DEADLINE)
    if natural:
        best = natural
    if best is not None and len(best) >= 3:
        return [list(c) for c in best]
    # Try many randomized restarts
    for it in range(200):
        if time.time() > DEADLINE - 2:
            break
        # shuffle edge list to vary adjacency build order (affects nothing structurally,
        # but we shuffle neighbor order inside _try_decompose anyway)
        edges_copy = list(P['edges'])
        rng.shuffle(edges_copy)
        res = _try_decompose(edges_copy, rng, DEADLINE - 1)
        if res is None:
            continue
        if best is None or len(res) > len(best):
            best = res
        if len(best) >= 3:
            break
    if best is None:
        # fallback: return the natural single cycle if we can find one
        adj = _build_adjacency(P['edges'])
        c = _find_hamiltonian_cycle(adj, start=0, time_limit=DEADLINE - 1)
        if c is not None:
            return [list(c)]
        return [list(range(N))]
    return [list(c) for c in best]
# EVOLVE-BLOCK-END
