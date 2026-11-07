"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    nodes = list(range(n))
    edge_set = set()
    for a, b in P['edges']:
        edge_set.add((min(a, b), max(a, b)))

    # adjacency as dict of sets
    adj = {v: set() for v in nodes}
    for (a, b) in edge_set:
        adj[a].add(b)
        adj[b].add(a)

    deadline = _time.time() + 70  # safety margin vs 90s wall-clock

    # ---------- find one Hamiltonian cycle in the available subgraph ----------
    def find_hamiltonian(avail):
        """avail: dict v->set of neighbors (available edges). Returns list of
        n distinct vertices (cycle) or None."""
        # pick a start vertex with smallest degree for determinism
        start = min(nodes, key=lambda v: (len(avail[v]), v))
        if len(avail[start]) < 2:
            return None
        # order neighbors deterministically
        nbrs = sorted(avail[start])
        # symmetry: fix the second vertex to the smallest neighbor to break reflection
        second = nbrs[0]
        if len(nbrs) < 2:
            return None
        # path from start -> second, must end at a neighbor of start (not second)
        start_nbrs = set(nbrs)  # possible closing neighbors

        path = [start, second]
        used = {start, second}
        # available edges remaining (mutable sets)
        rem = {v: set(avail[v]) for v in nodes}
        # remove used endpoints' used edges
        rem[start].discard(second)
        rem[second].discard(start)

        def propagate(rem, path, used):
            """Enforce degree-1 forcing and 2-edge-bridge. Return False on
            contradiction, else True (modifying rem)."""
            changed = True
            while changed:
                changed = False
                # degree-1 forcing: a vertex with exactly 1 available edge must use it
                for v in nodes:
                    if v in used:
                        continue
                    deg = len(rem[v])
                    if deg == 0:
                        return False
                    if deg == 1:
                        w = next(iter(rem[v]))
                        # if w already used and w != path[-1], then w must be the
                        # other endpoint of the forced edge... but w is used, so the
                        # edge v-w is the only way out of v. If w is not the current
                        # path end, then v must connect to w, but w is used -> v can
                        # only be placed adjacent to w. This is only OK if v becomes
                        # the new end adjacent to w, i.e. w == path[-1].
                        if w not in used:
                            # force edge v-w: remove all other edges incident to v
                            other = rem[v] - {w}
                            for x in other:
                                rem[x].discard(v)
                            rem[v] = {w}
                            changed = True
                        else:
                            # w is used; the only edge out of v goes to a used vertex.
                            # That used vertex must be the current path end so v can be
                            # appended. If w is an interior used vertex, impossible.
                            if w != path[-1]:
                                return False
                            # else v must be appended next to w (handled by search)
                # 2-edge-bridge detection
                for v in nodes:
                    if v in used:
                        continue
                    if len(rem[v]) == 2:
                        a, b = tuple(rem[v])
                        # if neither a nor b is the current path end, then v is a
                        # 2-edge bridge that must be entered from one and exited the
                        # other; but since both endpoints are 'free' (unused) it's a
                        # cut of size 2. For a Hamiltonian cycle, a 2-edge bridge
                        # where both endpoints are unused is fine only if the cycle
                        # passes v between them. We can force that v's two edges are
                        # both used, which means we cannot drop either. This is
                        # already true. To prune: if a and b are both unused and
                        # neither equals path[-1], the subgraph on the side of the
                        # bridge must be fully traversed; hard to check cheaply.
                        # Skip heavy bridge logic; degree forcing is the main win.
                        pass
            return True

        # iterative DFS with backtracking, using a stack of states
        # state = (path, used, rem, choice_index)
        # To keep it fast, implement recursive with sys.setrecursionlimit.
        import sys
        sys.setrecursionlimit(10000)

        # precompute neighbor ordering
        def dfs(path, used, rem):
            if len(path) == n:
                last = path[-1]
                if last in start_nbrs and last != second:
                    return path
                return None
            cur = path[-1]
            cand = [x for x in rem[cur] if x not in used]
            # order candidates: prefer those with fewer remaining options
            cand.sort(key=lambda x: len(rem[x]))
            for x in cand:
                # try edge cur-x
                # snapshot
                rem[cur].discard(x)
                rem[x].discard(cur)
                new_used = used | {x}
                new_path = path + [x]
                if propagate(rem, new_path, new_used):
                    res = dfs(new_path, new_used, rem)
                    if res is not None:
                        return res
                # undo
                rem[cur].add(x)
                rem[x].add(cur)
            return None

        # initial propagate
        if not propagate(rem, path, used):
            return None
        res = dfs(path, used, rem)
        return res

    # ---------- main: repeatedly find disjoint cycles ----------
    avail = {v: set(adj[v]) for v in nodes}
    cycles = []

    while True:
        if _time.time() > deadline:
            break
        if len(cycles) >= 3:
            break
        cyc = find_hamiltonian(avail)
        if cyc is None:
            break
        # validate
        if len(set(cyc)) != n or len(cyc) != n:
            break
        ok = True
        for i in range(n):
            a, b = cyc[i], cyc[(i + 1) % n]
            e = (min(a, b), max(a, b))
            if e not in avail[a]:
                ok = False
                break
        if not ok:
            break
        # remove used edges
        for i in range(n):
            a, b = cyc[i], cyc[(i + 1) % n]
            avail[a].discard(b)
            avail[b].discard(a)
        cycles.append(list(cyc))

    # convert to plain lists of ints
    out = [[int(v) for v in c] for c in cycles]
    if not out:
        # fallback: return a single valid cycle if we can find one
        avail = {v: set(adj[v]) for v in nodes}
        cyc = find_hamiltonian(avail)
        if cyc is not None:
            out = [[int(v) for v in cyc]]
        else:
            # last resort: any 3-cycle
            for (a, b) in edge_set:
                for c in adj[a]:
                    if c != b and (min(b, c), max(b, c)) in edge_set:
                        out = [[int(a), int(b), int(c)]]
                        break
                if out:
                    break
    return out
# EVOLVE-BLOCK-END
