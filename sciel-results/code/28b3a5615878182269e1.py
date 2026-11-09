"""Rewritten: max-clique-based branch set + randomized disjoint path assignment."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 85.0  # safety margin before 90s kill

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    edge_set = set()
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((a, b))
        edge_set.add((b, a))
    return adj, edge_set

def _bron_kerbosch(adj, n):
    """Bron-Kerbosch with pivot and degeneracy ordering; returns max clique."""
    best = []
    # degeneracy ordering
    order = list(range(n))
    # sort by degree descending for a simple degeneracy-like heuristic
    order.sort(key=lambda v: len(adj[v]), reverse=True)
    pos = {v: i for i, v in enumerate(order)}

    def bk(R, P, X):
        nonlocal best
        if not P and not X:
            if len(R) > len(best):
                best = list(R)
            return
        # pivot
        candidates = list(P) + list(X)
        if candidates:
            pivot = max(candidates, key=lambda v: len(adj[v] & (P | X)))
        else:
            pivot = None
        to_iter = list(P)
        if pivot is not None:
            to_iter = [v for v in to_iter if v not in adj[pivot]]
        for v in to_iter:
            if v not in P:
                continue
            R.add(v)
            Pn = P & adj[v]
            Xn = X & adj[v]
            P -= {v}
            X -= {v}
            bk(R, Pn, Xn)
            R.remove(v)
            X.add(v)

    bk(set(), set(range(n)), set())
    return best

def _bfs_path(adj, s, t, block, rng=None):
    """Find a shortest path from s to t avoiding 'block' (interior vertices)."""
    if s == t:
        return [s]
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == t:
            break
        nbs = list(adj[cur])
        if rng is not None:
            rng.shuffle(nbs)
        else:
            nbs.sort()
        for nb in nbs:
            if nb in prev:
                continue
            if nb in block:
                continue
            prev[nb] = cur
            q.append(nb)
    if t not in prev:
        return None
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def _try_assign_paths(adj, branches, order, block_extra, rng):
    """Assign disjoint paths for pairs in 'order'. Returns dict pair->path or None."""
    used = set(block_extra)
    all_b = set(branches)
    paths = {}
    for (u, v) in order:
        # block = used interiors + all branches except u and v (as interior)
        block = set(used)
        for b in all_b:
            if b != u and b != v:
                block.add(b)
        path = _bfs_path(adj, u, v, block, rng)
        if path is None:
            return None
        interior = set(path[1:-1])
        if interior & used:
            return None
        used |= interior
        paths[(u, v)] = path
    return paths

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']
    adj, edge_set = _build_adj(n, edges)

    def time_up():
        return time.time() - t0 > DEADLINE - 1.5

    # Trivial fallback: first edge
    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    # Find max clique
    max_clique = _bron_kerbosch(adj, n)
    # Sort for determinism
    max_clique.sort()

    best = None
    best_size = 0

    def finalize(branches, paths_dict, order):
        nonlocal best, best_size
        if paths_dict is None:
            return
        b = list(branches)
        p = [paths_dict[(u, v)] for (u, v) in order]
        if len(b) > best_size:
            best = {'branches': [int(x) for x in b],
                    'paths': [[int(x) for x in pp] for pp in p]}
            best_size = len(b)

    # Try the max clique with multiple random orderings and rng seeds
    def orderings_for(branches):
        pairs = list(combinations(sorted(branches), 2))
        return pairs

    # First try the max clique with several randomized attempts
    if len(max_clique) >= 2:
        base_pairs = orderings_for(max_clique)
        # deterministic first
        rng = random.Random(0)
        res = _try_assign_paths(adj, max_clique, base_pairs, set(), rng)
        finalize(max_clique, res, base_pairs)

        # randomized orderings: shuffle pair order, and rng for tie-breaking
        attempts = 0
        max_attempts = 400
        while attempts < max_attempts and not time_up():
            attempts += 1
            rng = random.Random(attempts)
            shuffled = base_pairs[:]
            rng.shuffle(shuffled)
            res = _try_assign_paths(adj, max_clique, shuffled, set(), rng)
            if res is not None:
                finalize(max_clique, res, shuffled)
                # found a valid assignment for the max clique; good enough, but
                # we still try to expand beyond it below.
                break

    # Ensure we have at least a 2-branch solution
    if best is None:
        u, v = edges[0]
        best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        best_size = 2

    # Try to EXPAND beyond the max clique: greedily add vertices one at a time
    # and re-solve. This can yield more branches than any clique if paths are
    # long enough to be disjoint.
    current_branches = list(best['branches'])
    vertices = list(range(n))

    # Greedy expansion with randomized re-solves
    improved = True
    while improved and not time_up():
        improved = False
        cur_sorted = sorted(current_branches)
        non_branch = [x for x in vertices if x not in set(cur_sorted)]
        # Try each candidate vertex; for each, attempt several random orderings
        best_add = None
        for vb in non_branch:
            if time_up():
                break
            new_bs = cur_sorted + [vb]
            new_bs.sort()
            pairs = list(combinations(new_bs, 2))
            found = False
            for seed in range(30):
                if time_up():
                    break
                rng = random.Random(seed)
                shuffled = pairs[:]
                rng.shuffle(shuffled)
                res = _try_assign_paths(adj, new_bs, shuffled, set(), rng)
                if res is not None:
                    # valid!
                    b = list(new_bs)
                    p = [res[(u, v)] for (u, v) in shuffled]
                    if len(b) > best_size:
                        best = {'branches': [int(x) for x in b],
                                'paths': [[int(x) for x in pp] for pp in p]}
                        best_size = len(b)
                        current_branches = b
                        improved = True
                        best_add = vb
                        found = True
                    break
            if found:
                break
        if best_add is None:
            # no single vertex can be added; try a randomized larger jump
            pass

    # Randomized search for larger sets (jump beyond greedy)
    if not time_up():
        base_rng = random.Random(12345)
        max_tries = 600
        for trial in range(max_tries):
            if time_up():
                break
            # target size: try to beat best_size
            size = base_rng.randint(best_size + 1, min(best_size + 2, n))
            if size < 2 or size > n:
                continue
            subset = base_rng.sample(vertices, size)
            subset.sort()
            pairs = list(combinations(subset, 2))
            for seed in range(15):
                if time_up():
                    break
                rng = random.Random(seed * 1000 + trial)
                shuffled = pairs[:]
                rng.shuffle(shuffled)
                res = _try_assign_paths(adj, subset, shuffled, set(), rng)
                if res is not None:
                    b = list(subset)
                    p = [res[(u, v)] for (u, v) in shuffled]
                    if len(b) > best_size:
                        best = {'branches': [int(x) for x in b],
                                'paths': [[int(x) for x in pp] for pp in p]}
                        best_size = len(b)
                    break

    # If we somehow lost a valid result, fallback
    if best is None or best_size < 2:
        return fallback()

    return best
# EVOLVE-BLOCK-END
