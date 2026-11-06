"""Strong edge colouring: randomized greedy restarts + local colour compression."""
import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
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

    def build_conflict(edges, adj):
        m = len(edges)
        # vertex -> list of edge indices incident
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)
        # conflict[i] = set of edge indices that cannot share a colour with i
        conflict = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            # edges sharing a vertex with i
            for w in (u, v):
                for j in v2e.get(w, []):
                    if j != i:
                        conflict[i].add(j)
            # edges joined by an edge (one endpoint of i adjacent to one endpoint of j)
            for w in (u, v):
                for nb in adj.get(w, ()):
                    for j in v2e.get(nb, []):
                        conflict[i].add(j)
        return conflict

    def greedy_first_fit(edges, conflict, order):
        m = len(edges)
        colour = [-1] * m
        used_per_colour = [set() for _ in range(m)]
        for idx in order:
            taken = set()
            for c in range(m):
                if colour[idx] != -1:
                    break
                # check if idx conflicts with any edge already in colour c
                ok = True
                for j in conflict[idx]:
                    if colour[j] == c:
                        ok = False
                        break
                if ok:
                    colour[idx] = c
                    used_per_colour[c].add(idx)
                    break
            if colour[idx] == -1:
                # find first free colour
                c = 0
                while c < m and len(used_per_colour[c]) > 0:
                    # check conflicts
                    ok = True
                    for j in conflict[idx]:
                        if colour[j] == c:
                            ok = False
                            break
                    if not ok:
                        c += 1
                        continue
                    ok = True
                    break
                if not ok:
                    c += 1
                colour[idx] = c
                used_per_colour[c].add(idx)
        return colour

    def greedy_first_fit_fast(edges, conflict, order):
        m = len(edges)
        colour = [-1] * m
        # For each colour, track which edges are in it
        colour_edges = []  # list of sets
        for idx in order:
            placed = False
            for c in range(len(colour_edges)):
                # check if idx conflicts with any edge in colour c
                ok = True
                for j in conflict[idx]:
                    if colour[j] == c:
                        ok = False
                        break
                if ok:
                    colour[idx] = c
                    colour_edges[c].add(idx)
                    placed = True
                    break
            if not placed:
                c = len(colour_edges)
                colour[idx] = c
                colour_edges.append({idx})
        return colour

    def local_compression(colour, conflict, edges, rng, max_iters=20000):
        m = len(edges)
        best = list(colour)
        best_num = max(best) + 1 if best else 0
        for _ in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            # pick a random colour class and try to move edges
            num_colours = max(best) + 1
            if num_colours == 0:
                break
            c = rng.randrange(num_colours)
            # get edges in colour c
            cls = [i for i in range(m) if best[i] == c]
            if not cls:
                continue
            # try to move a random edge from c to another colour
            i = cls[rng.randrange(len(cls))]
            # try to move i to a lower colour
            moved = False
            for c2 in range(c):
                ok = True
                for j in conflict[i]:
                    if best[j] == c2:
                        ok = False
                        break
                if ok:
                    best[i] = c2
                    moved = True
                    break
            if moved:
                continue
            # try to move i to a new colour (only if it reduces max)
            if c == num_colours - 1 and len(cls) > 1:
                # moving to new colour won't help
                pass
            # try swapping: find another edge j in a different colour that could take i's place
            # Simple: try to recolor i to any available colour < num_colours
            for c2 in range(num_colours):
                if c2 == c:
                    continue
                ok = True
                for j in conflict[i]:
                    if best[j] == c2:
                        ok = False
                        break
                if ok:
                    best[i] = c2
                    moved = True
                    break
            if not moved:
                # try to move some other edge from c to free up, then move i
                for k in cls:
                    if k == i:
                        continue
                    for c2 in range(num_colours):
                        if c2 == c:
                            continue
                        ok = True
                        for j in conflict[k]:
                            if best[j] == c2:
                                ok = False
                                break
                        if ok:
                            best[k] = c2
                            # now try to move i to c
                            ok_i = True
                            for j in conflict[i]:
                                if best[j] == c:
                                    ok_i = False
                                    break
                            if ok_i:
                                best[i] = c
                            # update best
                            new_num = max(best) + 1
                            if new_num < best_num:
                                best_num = new_num
                            break
                    else:
                        continue
                    break
            new_num = max(best) + 1
            if new_num < best_num:
                best_num = new_num
        return best, best_num

    def solve_instance(n, d, seed, time_budget):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        conflict = build_conflict(edges, adj)

        best_colour = None
        best_num = m  # worst case: each edge its own colour
        best_order = list(range(m))

        # Multiple random restarts with different edge orderings
        rng = Random(seed + 7919)
        num_restarts = 0
        while time.time() < t0 + time_budget:
            num_restarts += 1
            # Create a random order biased toward high-conflict edges
            order = list(range(m))
            # Weight by conflict size
            weights = [len(conflict[i]) for i in range(m)]
            # Random shuffle
            rng.shuffle(order)
            colour = greedy_first_fit_fast(edges, conflict, order)
            num = max(colour) + 1 if colour else 0

            # Local compression
            colour, num = local_compression(colour, conflict, edges, rng, max_iters=5000)

            if num < best_num:
                best_num = num
                best_colour = colour
                best_order = order
                # If we hit the conjectured bound, stop
                if best_num <= 5 * d * d // 4:
                    break
            if time.time() > DEADLINE - 5:
                break

        # Final local compression on best
        if best_colour is not None:
            best_colour, best_num = local_compression(best_colour, conflict, edges, rng, max_iters=20000)

        return [best_colour[i] for i in range(m)]

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    # Allocate time: give more to larger instances
    budgets = [35.0, 40.0, 40.0]
    for idx, spec in enumerate(specs):
        if time.time() > DEADLINE - 5:
            # Not enough time; return a trivial valid coloring
            edges = gen_instance(*spec)
            m = len(edges)
            out.append(list(range(m)))
            continue
        res = solve_instance(*spec, budgets[idx])
        out.append(res)

    # Validate and fix if needed (shouldn't happen, but safety)
    for idx, spec in enumerate(specs):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        m = len(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        colours = out[idx]
        # Check validity
        colour_classes = {}
        valid = True
        for i, c in enumerate(colours):
            if c < 0 or c >= m:
                valid = False
                break
            colour_classes.setdefault(c, []).append(i)
        if valid:
            for c, cls in colour_classes.items():
                for a in range(len(cls)):
                    for b in range(a + 1, len(cls)):
                        i, j = cls[a], cls[b]
                        u1, v1 = edges[i]
                        u2, v2 = edges[j]
                        # Check no shared vertex
                        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                            valid = False
                            break
                        # Check no joining edge
                        if (u1 in adj.get(v2, set()) or u1 in adj.get(u2, set()) or
                            v1 in adj.get(u2, set()) or v1 in adj.get(v2, set())):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
        if not valid:
            # Fallback: trivial coloring
            out[idx] = list(range(m))

    return out
