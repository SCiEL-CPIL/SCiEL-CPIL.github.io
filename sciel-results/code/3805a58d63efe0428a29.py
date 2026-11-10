"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    deadline = _time.time() + 84  # safety margin under the 90s kill
    rng = random.Random(12345)

    def find_path(src, dst, forbidden, min_interior=0):
        """Shortest path src->dst avoiding forbidden (except src/dst).
        Among shortest, prefer the one with the fewest interior vertices
        (i.e. prefer direct edges) to conserve interior budget.
        Returns list of vertices or None."""
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        q = deque([src])
        found = False
        while q:
            u = q.popleft()
            if u == dst:
                found = True
                break
            for v in adj[u]:
                if v in visited:
                    continue
                if v != dst and v in forbidden:
                    continue
                visited.add(v)
                parent[v] = u
                q.append(v)
        if not found:
            return None
        path = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        if len(path) - 2 < min_interior:
            return None
        return path

    def check_branch_set(branches, restarts=160):
        """Try to assign disjoint interior paths for all pairs of branches.
        Randomized greedy + path-swap local search. Returns dict {(u,v):path}
        or None."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        if not pairs:
            return {}

        best = None

        for r in range(restarts):
            if _time.time() > deadline:
                break
            order = list(pairs)
            rng.shuffle(order)
            paths = {}
            used = set()
            ok = True
            for (u, v) in order:
                forb = (bset - {u, v}) | used
                p = find_path(u, v, forb)
                if p is None:
                    ok = False
                    break
                interior = set(p[1:-1])
                if interior & bset or interior & used:
                    ok = False
                    break
                used |= interior
                paths[(u, v)] = p
            if not ok:
                continue

            # local search: swap interior vertices between two paths to free space
            improved = True
            guard = 0
            while improved and guard < 200 and _time.time() < deadline:
                improved = False
                guard += 1
                pl = list(paths.items())
                rng.shuffle(pl)
                for i in range(len(pl)):
                    for j in range(i + 1, len(pl)):
                        (a1, a2), pa = pl[i]
                        (b1, b2), pb = pl[j]
                        ia = pa[1:-1]
                        ib = pb[1:-1]
                        if not ia or not ib:
                            continue
                        # candidate swap: move one vertex from ia into ib's place and vice versa
                        for va in ia:
                            for vb in ib:
                                na = pa[:pa.index(va)] + [vb] + pa[pa.index(va) + 1:]
                                nb = pb[:pb.index(vb)] + [va] + pb[pb.index(vb) + 1:]
                                # check adjacency for both modified paths
                                if not all(x in adj[y] or y in adj[x] for x, y in zip(na, na[1:])):
                                    continue
                                if not all(x in adj[y] or y in adj[x] for x, y in zip(nb, nb[1:])):
                                    continue
                                nia = set(na[1:-1])
                                nib = set(nb[1:-1])
                                if nia & bset or nib & bset:
                                    continue
                                if nia & nib:
                                    continue
                                # ensure no interior collides with other paths' interiors
                                other = set()
                                for (k1, k2), pk in paths.items():
                                    if (k1, k2) == (a1, a2) or (k1, k2) == (b1, b2):
                                        continue
                                    other |= set(pk[1:-1])
                                if (nia | nib) & other:
                                    continue
                                # accept the swap
                                paths[(a1, a2)] = na
                                paths[(b1, b2)] = nb
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

            if best is None or len(paths) > len(best):
                best = paths
            if len(paths) == len(pairs):
                break

        if best is not None and len(best) == len(pairs):
            return best
        return None

    def try_grow(initial_branches):
        """Local search over branch sets: add / remove / swap moves."""
        cur = list(initial_branches)
        cur_set = set(cur)
        best_size = len(cur)
        improved = True
        stall = 0
        while improved and _time.time() < deadline:
            improved = False
            stall += 1
            if stall > 60:
                break
            # ADD move
            cands = [c for c in range(n) if c not in cur_set]
            rng.shuffle(cands)
            for c in cands:
                if _time.time() > deadline:
                    break
                new_set = cur + [c]
                paths = check_branch_set(new_set)
                if paths is not None:
                    cur = new_set
                    cur_set = set(cur)
                    best_size = len(cur)
                    improved = True
                    break
            if improved:
                continue
            # SWAP move (remove one, add one)
            for rm in list(cur_set):
                if _time.time() > deadline:
                    break
                base = [x for x in cur if x != rm]
                for c in cands:
                    if c == rm:
                        continue
                    new_set = base + [c]
                    paths = check_branch_set(new_set)
                    if paths is not None:
                        cur = new_set
                        cur_set = set(cur)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # REMOVE move (only if it helps future adds is hard; skip to avoid shrinking)
        return cur

    def find_clique():
        deg = [len(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: -deg[x])
        clique = []
        for v in order:
            if all(v in adj[u] for u in clique):
                clique.append(v)
        return clique

    best_branches = None
    best_paths = None

    def consider(bs):
        nonlocal best_branches, best_paths
        if bs is None or len(bs) < 2:
            return
        paths = check_branch_set(bs)
        if paths is not None and (best_branches is None or len(bs) > len(best_branches)):
            best_branches = list(bs)
            best_paths = paths

    # Seed 1: clique
    clique = find_clique()
    if len(clique) >= 2:
        consider(try_grow(clique))

    # Seed 2: top-degree
    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])[:6]
    consider(try_grow(list(top)))

    # Seed 3: random triples
    for _ in range(40):
        if _time.time() > deadline:
            break
        start = rng.sample(range(n), min(3, n))
        consider(try_grow(start))

    # Seed 4: random pairs
    for _ in range(60):
        if _time.time() > deadline:
            break
        a, b = rng.sample(range(n), 2)
        consider(try_grow([a, b]))

    # Fallback: at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}

    branches = [int(b) for b in best_branches]
    paths_list = []
    for (u, v) in combinations(branches, 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            path = [u, v]
        paths_list.append([int(x) for x in path])

    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
