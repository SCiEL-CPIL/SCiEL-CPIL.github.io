import random
import time
import sys

sys.setrecursionlimit(100000)


def _is_non_2_colorable(n, edges):
    """
    Exact check (backtracking + unit propagation) that the 5-uniform hypergraph
    on vertices 0..n-1 has NO proper 2-coloring.
    Returns True  => NOT 2-colorable (no valid 2-coloring exists).
    Returns False => a valid 2-coloring exists.
    """
    if not edges:
        return False

    m = len(edges)
    E = [0] * m
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        E[i] = mask

    # vertex degree (for variable ordering)
    deg = [0] * n
    for mask in E:
        mm = mask
        while mm:
            lsb = mm & -mm
            deg[lsb.bit_length() - 1] += 1
            mm ^= lsb

    # adjacency: incident edges per vertex
    adj = [[] for _ in range(n)]
    for i in range(m):
        mm = E[i]
        while mm:
            lsb = mm & -mm
            adj[lsb.bit_length() - 1].append(i)
            mm ^= lsb

    # per-vertex critical-edge sets, indexed by k = unassigned count in edge (1..5)
    csets = [[set() for _ in range(6)] for _ in range(n)]
    for i in range(m):
        mm = E[i]
        while mm:
            lsb = mm & -mm
            csets[lsb.bit_length() - 1][5].add(i)
            mm ^= lsb

    # scratch arrays for the DFS
    cnt = [0] * n
    assigned = [0] * n

    def dfs(color1, unassigned, depth):
        # ---- unit propagation to fixpoint ----
        while True:
            forced = False
            for v in range(n):
                if unassigned & (1 << v):
                    continue
                # k=1: edge fully determined -> must not be monochromatic
                for i in csets[v][1]:
                    if cnt[i] == 1 and (E[i] & unassigned) == 0:
                        if (E[i] & color1) == 0 or (E[i] & color1) == E[i]:
                            return False
                # k=2: the two unassigned vertices must differ
                for i in csets[v][2]:
                    if cnt[i] == 2:
                        u = E[i] & unassigned
                        a = u & -u
                        b = u ^ a
                        if a == b:
                            continue
                        ia = a.bit_length() - 1
                        ib = b.bit_length() - 1
                        if assigned[ia] == assigned[ib]:
                            return False
                # k=3: the three unassigned vertices must not all be equal
                for i in csets[v][3]:
                    if cnt[i] == 3:
                        u = E[i] & unassigned
                        a = u & -u; u ^= a
                        b = u & -u; u ^= b
                        c = u & -u
                        ia = a.bit_length() - 1
                        ib = b.bit_length() - 1
                        ic = c.bit_length() - 1
                        if assigned[ia] == assigned[ib] == assigned[ic]:
                            return False
                # k=4: force the 4th unassigned vertex to the opposite color
                for i in csets[v][4]:
                    if cnt[i] == 4:
                        u = E[i] & unassigned
                        a = u & -u
                        ia = a.bit_length() - 1
                        rest = u ^ a
                        if (rest & color1) == 0:
                            color1 |= a
                            unassigned ^= a
                            assigned[ia] = 1
                            cnt[ia] = 1
                            for w in range(n):
                                csets[w][5].discard(i)
                            for w in range(n):
                                if w != ia and (E[i] & (1 << w)):
                                    csets[w][4].add(i)
                            forced = True
                        elif (rest & color1) == rest:
                            unassigned ^= a
                            assigned[ia] = 0
                            cnt[ia] = 1
                            for w in range(n):
                                csets[w][5].discard(i)
                            for w in range(n):
                                if w != ia and (E[i] & (1 << w)):
                                    csets[w][4].add(i)
                            forced = True
            if not forced:
                break
        # ---- base case: all assigned ----
        if unassigned == 0:
            return True
        # ---- choose highest-degree unassigned vertex ----
        best = -1
        bestv = -1
        mm = unassigned
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            if deg[v] > best:
                best = deg[v]
                bestv = v
            mm ^= lsb
        a = 1 << bestv

        # ---- branch: color 1 ----
        assigned[bestv] = 1
        color1 |= a
        unassigned ^= a
        cnt[bestv] = 1
        for w in range(n):
            csets[w][5].discard(bestv)
        for i in adj[bestv]:
            csets[bestv][4].add(i)
            for w in range(n):
                if w != bestv and (E[i] & (1 << w)):
                    csets[w][4].add(i)
        if dfs(color1, unassigned, depth + 1):
            return True

        # ---- undo color-1 branch ----
        for i in adj[bestv]:
            csets[bestv][4].discard(i)
            for w in range(n):
                if w != bestv and (E[i] & (1 << w)):
                    csets[w][4].discard(i)
        for w in range(n):
            csets[w][5].add(bestv)
        color1 ^= a
        cnt[bestv] = 0

        # ---- branch: color 0 ----
        assigned[bestv] = 0
        unassigned ^= a
        cnt[bestv] = 1
        for w in range(n):
            csets[w][5].discard(bestv)
        for i in adj[bestv]:
            csets[bestv][4].add(i)
            for w in range(n):
                if w != bestv and (E[i] & (1 << w)):
                    csets[w][4].add(i)
        if dfs(color1, unassigned, depth + 1):
            return True

        # ---- undo color-0 branch ----
        for i in adj[bestv]:
            csets[bestv][4].discard(i)
            for w in range(n):
                if w != bestv and (E[i] & (1 << w)):
                    csets[w][4].discard(i)
        for w in range(n):
            csets[w][5].add(bestv)
        cnt[bestv] = 0
        assigned[bestv] = 0
        return False

    # symmetry breaking: fix vertex 0 to color 0
    assigned[0] = 0
    cnt[0] = 1
    unassigned = ((1 << n) - 1) ^ 1
    for w in range(n):
        csets[w][5].discard(0)
    for i in adj[0]:
        csets[0][4].add(i)
        for w in range(n):
            if w != 0 and (E[i] & (1 << w)):
                csets[w][4].add(i)

    if dfs(0, unassigned, 0):
        return False  # a valid coloring was found
    return True        # no valid coloring


def _greedy_minimize(n, edges):
    """Remove edges while the hypergraph stays non-2-colorable (random order)."""
    cur = list(edges)
    improved = True
    while improved:
        improved = False
        random.shuffle(cur)
        for i in range(len(cur) - 1, -1, -1):
            cand = cur[:i] + cur[i + 1:]
            if _is_non_2_colorable(n, cand):
                cur = cand
                improved = True
    return cur


def _complement_search(n, start, deadline):
    """
    Complement family: start from a sparse random 5-uniform hypergraph, then
    repeatedly add random edges until the graph becomes non-2-colorable.
    This is a structurally different search direction from greedy removal.
    """
    import itertools
    all5 = list(itertools.combinations(range(n), 5))
    rng = random.Random(start)
    best = None
    best_len = 10 ** 9
    while time.time() < deadline:
        # random sparse starting set
        k = rng.randint(24, 40)
        chosen = rng.sample(all5, min(k, len(all5)))
        chosen_set = set(map(tuple, chosen))
        if _is_non_2_colorable(n, chosen):
            # already non-2-colorable; minimize it
            red = _greedy_minimize(n, chosen)
            if len(red) < best_len:
                best_len = len(red)
                best = red
            continue
        # add random edges until non-2-colorable
        cur = list(chosen)
        pool = [e for e in all5 if e not in chosen_set]
        rng.shuffle(pool)
        added = 0
        while added < len(pool) and time.time() < deadline:
            e = pool[added]
            added += 1
            if _is_non_2_colorable(n, cur + [e]):
                cur.append(e)
                break
        # found a non-2-colorable set; minimize
        if _is_non_2_colorable(n, cur):
            red = _greedy_minimize(n, cur)
            if len(red) < best_len:
                best_len = len(red)
                best = red
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 105  # safety margin under the 120/150 s limit
    random.seed(12345)

    # ---- Phase 1: minimize K9^(5) (the proven ~52-edge baseline) ----
    import itertools
    n = 9
    all5 = list(itertools.combinations(range(n), 5))
    best = _greedy_minimize(n, all5)
    best_len = len(best)

    # ---- Phase 2: complement-based search to try to beat the baseline ----
    seed = 0
    while time.time() < deadline - 15 and best_len > 51:
        res = _complement_search(n, seed, min(deadline - 15, start_time + 30))
        seed += 1
        if res is not None and len(res) < best_len:
            best = res
            best_len = len(res)

    # ---- Final verification (guaranteed valid result) ----
    if not _is_non_2_colorable(n, best):
        # fall back to the complete hypergraph (definitely non-2-colorable)
        best = [list(e) for e in all5]

    return {'n': n, 'edges': [list(e) for e in best]}
