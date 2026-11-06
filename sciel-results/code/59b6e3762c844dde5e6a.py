"""Strong edge colourings via maximum induced matching extraction (branch-and-bound)."""
# EVOLVE-BLOCK-START
import time

_DEADLINE = 145.0
_start = time.time()


def _remaining():
    return _DEADLINE - time.time()


def construct():
    from random import Random

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def greedy_mis(available):
        # Greedy maximal induced matching, ordered by edge "exclusion weight" (conflicts).
        # available: list of edge indices
        # conflict[i]: bitmask of edges conflicting with i
        avail_set = set(available)
        # order by number of conflicts within available (higher first -> removes more)
        order = sorted(available, key=lambda i: -len(conflict[i] & avail_mask))
        chosen = []
        removed = 0  # bitmask of removed edges
        for i in order:
            if removed >> i & 1:
                continue
            chosen.append(i)
            removed |= conflict[i] | (1 << i)
            if removed == avail_mask:
                break
        return chosen

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        m = len(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # conflict[i] = bitmask of edges j (j != i) that share a vertex with i or are joined by an edge
        conflict = [0] * m
        for i in range(m):
            u, v = edges[i]
            near = {u, v} | adj[u] | adj[v]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                a, b = edges[j]
                if a in near or b in near:
                    mask |= 1 << j
            conflict[i] = mask

        # colour assignment
        colour = [-1] * m
        used_mask = 0
        ncolours = 0

        while used_mask != (1 << m) - 1:
            if _remaining() < 1.0:
                # Fallback: greedy colour the rest to guarantee validity
                for i in range(m):
                    if used_mask >> i & 1:
                        continue
                    near = {edges[i][0], edges[i][1]} | adj[edges[i][0]] | adj[edges[i][1]]
                    taken = set()
                    for j in range(m):
                        if used_mask >> j & 1:
                            continue
                        a, b = edges[j]
                        if a in near or b in near:
                            if colour[j] >= 0:
                                taken.add(colour[j])
                    c = 0
                    while c in taken:
                        c += 1
                    colour[i] = c
                    used_mask |= 1 << i
                ncolours = max(colour) + 1
                break

            # Find a large induced matching among remaining edges
            remaining = [i for i in range(m) if not (used_mask >> i & 1)]
            avail_mask = 0
            for i in remaining:
                avail_mask |= 1 << i

            # Branch-and-bound to find a large independent set in conflict graph
            # We want to maximize the number of chosen edges (all pairwise non-conflicting).
            # Use a simple branch and bound with bitsets.

            best_found = greedy_mis(remaining)
            best_size = len(best_found)

            # Try to improve via branch and bound (bounded effort)
            def bb_search(cand_mask, chosen, chosen_mask):
                nonlocal best_found, best_size
                if _remaining() < 0.2:
                    return
                # upper bound: number of bits in cand_mask
                if cand_mask.bit_count() + len(chosen) <= best_size:
                    return
                if cand_mask == 0:
                    if len(chosen) > best_size:
                        best_size = len(chosen)
                        best_found = list(chosen)
                    return
                # pick a vertex: the one with fewest conflicts within cand (degree)
                # to bound, pick vertex with max degree to prune
                # find a vertex in cand
                lb = cand_mask & (-cand_mask)
                v = lb.bit_length() - 1
                # Option 1: exclude v
                bb_search(cand_mask & ~lb, chosen, chosen_mask)
                # Option 2: include v
                new_cand = cand_mask & ~conflict[v] & ~(1 << v)
                chosen.append(v)
                bb_search(new_cand, chosen, chosen_mask | (1 << v))
                chosen.pop()

            # Limit B&B by node budget
            node_budget = 200000
            nodes = [0]
            def bb_search_limited(cand_mask, chosen):
                nonlocal best_found, best_size
                nodes[0] += 1
                if nodes[0] > node_budget or _remaining() < 0.2:
                    return
                if cand_mask.bit_count() + len(chosen) <= best_size:
                    return
                if cand_mask == 0:
                    if len(chosen) > best_size:
                        best_size = len(chosen)
                        best_found = list(chosen)
                    return
                # choose vertex with max degree in cand for tighter pruning
                # compute degrees
                best_v = -1
                best_deg = -1
                cm = cand_mask
                while cm:
                    lb = cm & (-cm)
                    v = lb.bit_length() - 1
                    deg = (conflict[v] & cand_mask).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    cm ^= lb
                v = best_v
                lb = 1 << v
                # exclude v
                bb_search_limited(cand_mask & ~lb, chosen)
                # include v
                new_cand = cand_mask & ~conflict[v] & ~lb
                chosen.append(v)
                bb_search_limited(new_cand, chosen)
                chosen.pop()

            chosen = []
            bb_search_limited(avail_mask, chosen)

            # Assign this colour class
            for i in best_found:
                colour[i] = ncolours
                used_mask |= 1 << i
            ncolours += 1

        out.append([colour[i] for i in range(m)])

    return out
# EVOLVE-BLOCK-END
