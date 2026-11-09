"""Strong edge colourings of three fixed bounded-degree graphs.

Rewritten solver: bitmask conflict checks + DSATUR greedy + local search
(move-to-lower and 2-swap) with multi-restart.
"""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random

_DEADLINE = 135.0  # wall-clock kill is 150s; leave a safety margin.


def construct():
    t0 = time.time()

    def _instance(n, d, seed):
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def _solve(n, d, seed, budget):
        edges = _instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Adjacency as bitmask (n <= 80 fits in a Python int).
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # forbid[i]: vertices an edge in the same colour class must NOT use.
        # = {u,v} union N(u) union N(v).
        forbid = [0] * m
        for i, (u, v) in enumerate(edges):
            forbid[i] = ((1 << u) | (1 << v) | adj[u] | adj[v])

        # degree per vertex
        deg = [adj[v].bit_count() for v in range(n)]

        def _dsatur_greedy(shuffle_seed, force_natural=False):
            colour = [-1] * m
            colour_to_edges = []          # list of lists of edge indices
            uncoloured = set(range(m))
            # constraint score: number of colours that edge i cannot use.
            constraint = [0] * m

            while uncoloured:
                # pick most constrained edge (ties broken randomly / naturally)
                if force_natural:
                    i = min(uncoloured)
                else:
                    best = None
                    best_key = None
                    for i in uncoloured:
                        key = (constraint[i], i)
                        if best_key is None or key > best_key:
                            best_key = key
                            best = i
                    i = best

                # find smallest colour c where no conflict
                placed = False
                for c in range(len(colour_to_edges)):
                    ok = True
                    for j in colour_to_edges[c]:
                        if (forbid[i] & (1 << edges[j][0])) or \
                           (forbid[i] & (1 << edges[j][1])):
                            ok = False
                            break
                    if ok:
                        colour[i] = c
                        colour_to_edges[c].append(i)
                        placed = True
                        break
                if not placed:
                    c = len(colour_to_edges)
                    colour[i] = c
                    colour_to_edges.append([i])

                # update constraints for remaining edges
                for k in uncoloured:
                    if k == i:
                        continue
                    fu, fv = edges[k]
                    for j in colour_to_edges[colour[i]]:
                        a, b = edges[j]
                        if a == fu or a == fv or b == fu or b == fv:
                            continue
                        if (adj[fu] & (1 << a)) or (adj[fu] & (1 << b)) or \
                           (adj[fv] & (1 << a)) or (adj[fv] & (1 << b)):
                            constraint[k] += 1
                uncoloured.discard(i)

            return colour, colour_to_edges

        def _can_place(i, c, colour_to_edges, colour):
            """Can edge i be (re)assigned to colour c?"""
            for j in colour_to_edges[c]:
                if j == i:
                    continue
                a, b = edges[j]
                if (forbid[i] & (1 << a)) or (forbid[i] & (1 << b)):
                    return False
            return True

        def _compact(colour, colour_to_edges):
            present = [c for c in range(len(colour_to_edges)) if colour_to_edges[c]]
            if not present:
                return colour, colour_to_edges
            if present == list(range(len(present))):
                return colour, colour_to_edges
            mapping = {old: new for new, old in enumerate(present)}
            for i in range(m):
                colour[i] = mapping[colour[i]]
            new_cte = defaultdict(list)
            for i, c in enumerate(colour):
                new_cte[c].append(i)
            colour_to_edges = [new_cte[c] for c in range(len(present))]
            return colour, colour_to_edges

        def _local_search(colour, colour_to_edges, time_limit):
            t_start = time.time()
            while True:
                if time.time() - t_start > time_limit:
                    break
                improved = False

                # Phase 1: move each edge to the lowest possible colour.
                for i in range(m):
                    cur_c = colour[i]
                    for c in range(cur_c):
                        if _can_place(i, c, colour_to_edges, colour):
                            colour_to_edges[cur_c].remove(i)
                            colour_to_edges[c].append(i)
                            colour[i] = c
                            improved = True
                            break

                # Phase 2: 2-swap. For edge i in colour c1, try swapping with
                # edge j in colour c2 (c2 != c1) so that i -> c2 and j -> c1.
                for i in range(m):
                    c1 = colour[i]
                    # candidate target colours: any colour != c1
                    for c2 in range(len(colour_to_edges)):
                        if c2 == c1:
                            continue
                        # i must fit in c2 without j (j will leave)
                        ok_i = True
                        for j2 in colour_to_edges[c2]:
                            if j2 == i:
                                continue
                            a, b = edges[j2]
                            if (forbid[i] & (1 << a)) or (forbid[i] & (1 << b)):
                                ok_i = False
                                break
                        if not ok_i:
                            continue
                        # find j in c2 that fits in c1 without i
                        for j in colour_to_edges[c2]:
                            if j == i:
                                continue
                            ok_j = True
                            for k in colour_to_edges[c1]:
                                if k == j:
                                    continue
                                a, b = edges[k]
                                if (forbid[j] & (1 << a)) or (forbid[j] & (1 << b)):
                                    ok_j = False
                                    break
                            if ok_j:
                                # perform swap
                                colour_to_edges[c1].remove(i)
                                colour_to_edges[c2].remove(j)
                                colour_to_edges[c1].append(j)
                                colour_to_edges[c2].append(i)
                                colour[i] = c2
                                colour[j] = c1
                                improved = True
                                break
                        if improved:
                            break

                # Compact
                colour, colour_to_edges = _compact(colour, colour_to_edges)

                if not improved:
                    break

            return colour, colour_to_edges

        # ---- Multi-restart: try several orderings, keep the best. ----
        best = None
        best_count = m + 1

        # Restart 0: natural DSATUR (deterministic).
        colour, cte = _dsatur_greedy(0, force_natural=True)
        colour, cte = _local_search(colour, cte, time_limit=3.0)
        cnt = len(cte)
        if cnt < best_count:
            best_count = cnt
            best = list(colour)

        # Randomised restarts.
        rng = Random(seed + 7)
        restarts = 4
        for r in range(restarts):
            if time.time() - t0 > _DEADLINE - 20:
                break
            colour, cte = _dsatur_greedy(seed + r * 131, force_natural=False)
            colour, cte = _local_search(colour, cte, time_limit=4.0)
            cnt = len(cte)
            if cnt < best_count:
                best_count = cnt
                best = list(colour)

        # Final extended local search on the best solution.
        if best is not None and time.time() - t0 < _DEADLINE - 10:
            cte = defaultdict(list)
            for i, c in enumerate(best):
                cte[c].append(i)
            cte = [cte[c] for c in range(len(cte))]
            best, cte = _local_search(best, cte, time_limit=10.0)

        return best if best is not None else [0] * m

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        # Allocate remaining budget proportionally to problem size.
        remaining = _DEADLINE - (time.time() - t0)
        budget = max(5.0, remaining / 3.0)
        out.append(_solve(n, d, seed, budget))

    return out
# EVOLVE-BLOCK-END
