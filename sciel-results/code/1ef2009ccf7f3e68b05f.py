"""Strong edge colourings of three fixed bounded-degree graphs.

Combined solver: bitmask conflict checks + DSATUR greedy + local search
(move-to-lower and 2-swap) with multi-restart, plus colour-elimination
pass from the inspiration to reduce total colour count.
"""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random

_DEADLINE = 138.0  # wall-clock kill is 150s; leave a safety margin.


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

        # Precompute conflict masks: for each edge i, which edges j conflict with it.
        # Two edges conflict if they share a vertex OR are joined by an edge.
        # We use bitmasks over edge indices (m can be up to ~240, so we use Python ints).
        # Actually m is at most ~2*n*d/2 = n*d, which for n=80,d=6 is 480.
        # Python ints handle arbitrary size, so this is fine.
        conflict_mask = [0] * m
        for i in range(m):
            u1, v1 = edges[i]
            mask = 0
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Share a vertex?
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    mask |= (1 << j)
                    continue
                # Joined by an edge?
                if (adj[u1] & (1 << u2)) or (adj[u1] & (1 << v2)) or \
                   (adj[v1] & (1 << u2)) or (adj[v1] & (1 << v2)):
                    mask |= (1 << j)
            conflict_mask[i] = mask
            # Also set the reverse direction
            if mask:
                # We'll handle this by building a full symmetric matrix
                pass

        # Build symmetric conflict masks
        for i in range(m):
            for j in range(i + 1, m):
                if (conflict_mask[i] >> j) & 1:
                    conflict_mask[j] |= (1 << i)

        def _dsatur_greedy(shuffle_seed, force_natural=False):
            rng_local = Random(shuffle_seed)
            colour = [-1] * m
            colour_to_edges = []
            uncoloured = set(range(m))
            # constraint score: number of colours that edge i cannot use.
            constraint = [0] * m

            while uncoloured:
                if force_natural:
                    i = min(uncoloured)
                else:
                    # Pick most constrained edge
                    best = None
                    best_key = None
                    for i in uncoloured:
                        key = (constraint[i], rng_local.random(), i)
                        if best_key is None or key > best_key:
                            best_key = key
                            best = i
                    i = best

                # Find smallest colour c where no conflict
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

                # Update constraints for remaining edges
                for k in uncoloured:
                    if k == i:
                        continue
                    # Check if k conflicts with i
                    if (conflict_mask[k] >> i) & 1:
                        constraint[k] += 1
                uncoloured.discard(i)

            return colour, colour_to_edges

        def _can_place(i, c, colour_to_edges, colour):
            """Can edge i be (re)assigned to colour c?"""
            for j in colour_to_edges[c]:
                if j == i:
                    continue
                if (conflict_mask[i] >> j) & 1:
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

        def _eliminate_colours(colour, colour_to_edges, deadline):
            """Try to eliminate the highest colours by moving edges to lower colours."""
            used = len(colour_to_edges)
            for c in range(used - 1, 0, -1):
                if time.time() > deadline:
                    break
                edges_of_c = list(colour_to_edges[c])
                if not edges_of_c:
                    continue

                # Try to move each edge to an existing lower colour
                success = True
                for i in edges_of_c:
                    if time.time() > deadline:
                        success = False
                        break
                    found = False
                    for c2 in range(c):
                        if _can_place(i, c2, colour_to_edges, colour):
                            colour_to_edges[c].remove(i)
                            colour_to_edges[c2].append(i)
                            colour[i] = c2
                            found = True
                            break
                    if not found:
                        success = False
                        break

                if success:
                    # Colour c is now empty
                    colour_to_edges[c] = []
                else:
                    # Try swap-based approach
                    for i in edges_of_c:
                        if time.time() > deadline:
                            break
                        for c2 in range(c):
                            # Find blockers in c2
                            blockers = [j for j in colour_to_edges[c2]
                                        if (conflict_mask[i] >> j) & 1]
                            if len(blockers) == 1:
                                k = blockers[0]
                                # Check if k can move to c (after i moves to c2)
                                ok_k = True
                                for j2 in colour_to_edges[c]:
                                    if j2 == i:
                                        continue
                                    if (conflict_mask[k] >> j2) & 1:
                                        ok_k = False
                                        break
                                if ok_k:
                                    colour_to_edges[c].remove(i)
                                    colour_to_edges[c2].remove(k)
                                    colour_to_edges[c].append(k)
                                    colour_to_edges[c2].append(i)
                                    colour[i] = c2
                                    colour[k] = c
                                    break

                    # After swaps, try again to eliminate c
                    edges_of_c = [i for i in range(m) if colour[i] == c]
                    if edges_of_c:
                        success2 = True
                        for i in edges_of_c:
                            if time.time() > deadline:
                                success2 = False
                                break
                            found = False
                            for c2 in range(c):
                                if _can_place(i, c2, colour_to_edges, colour):
                                    colour_to_edges[c].remove(i)
                                    colour_to_edges[c2].append(i)
                                    colour[i] = c2
                                    found = True
                                    break
                            if not found:
                                success2 = False
                                break
                        if success2:
                            colour_to_edges[c] = []

            # Compact
            colour, colour_to_edges = _compact(colour, colour_to_edges)
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

                # Phase 2: 2-swap
                for i in range(m):
                    c1 = colour[i]
                    for c2 in range(len(colour_to_edges)):
                        if c2 == c1:
                            continue
                        # i must fit in c2 without j (j will leave)
                        ok_i = True
                        for j2 in colour_to_edges[c2]:
                            if (conflict_mask[i] >> j2) & 1:
                                ok_i = False
                                break
                        if not ok_i:
                            continue
                        # find j in c2 that fits in c1 without i
                        for j in colour_to_edges[c2]:
                            ok_j = True
                            for k in colour_to_edges[c1]:
                                if k == j:
                                    continue
                                if (conflict_mask[j] >> k) & 1:
                                    ok_j = False
                                    break
                            if ok_j:
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
        colour, cte = _eliminate_colours(colour, cte, t0 + _DEADLINE)
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
            colour, cte = _eliminate_colours(colour, cte, t0 + _DEADLINE)
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
            best, cte = _eliminate_colours(best, cte, t0 + _DEADLINE)

        return best if best is not None else [0] * m

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = _DEADLINE - (time.time() - t0)
        budget = max(5.0, remaining / 3.0)
        out.append(_solve(n, d, seed, budget))

    return out
# EVOLVE-BLOCK-END
